# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""DeepSeek V3 16B as torchtitan defines it, in plain PyTorch.

This is torchtitan's ``deepseek_v3`` "16B" flavor: 27 blocks (one dense, 26
MoE), multi-head latent attention, 64 routed experts with top-6 sigmoid routing
and two shared experts. It follows torchtitan's model op for op, so with the
same weights, batches and kernels it computes the same losses and gradients
bit for bit. Parameter names and order match torchtitan's, so its state dicts
load directly. The one buffer that differs is ``expert_bias``, which torchtitan
calls ``expert_bias_E``.

The model works on flattened tokens: a ``[B, S]`` batch becomes ``T = B * S``
rows, as torchtitan packs a microbatch.

Shape suffixes: B batch, S sequence, T tokens (B * S), D model dim, H heads,
K query/key head dim, V value head dim, R rotary dim / 2, E experts, F hidden
dim, N routed tokens (T * top_k), I/O grouped-matmul input/output features. In
the routed experts, R is instead the tokens this rank's experts process and e
those experts: N and E without expert parallelism (expert_parallel.py).
"""

import contextlib
import math
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.attention import sdpa_kernel, SDPBackend
from torch.nn.attention.flex_attention import (
    and_masks,
    BlockMask,
    create_block_mask,
    flex_attention,
)
from torch.utils.checkpoint import checkpoint


@dataclass
class DeepSeekV3Config:
    """torchtitan's DeepSeek V3 16B flavor (``deepseek_v3/flavors.py``)."""

    vocab_size: int = 102400
    dim: int = 2048
    n_layers: int = 27
    n_dense_layers: int = 1
    n_heads: int = 16
    kv_lora_rank: int = 512
    qk_nope_head_dim: int = 128
    qk_rope_head_dim: int = 64
    v_head_dim: int = 128
    mscale: float = 0.70
    dense_hidden_dim: int = 10944
    moe_hidden_dim: int = 1408
    num_experts: int = 64
    num_shared_experts: int = 2
    top_k: int = 6
    route_norm: bool = True
    route_scale: float = 1.0
    aux_loss_coeff: float = 1e-3
    load_balance_coeff: float = 1e-3
    norm_eps: float = 1e-6
    # YaRN rotary embedding; max_seq_len sizes the cache.
    max_seq_len: int = 4096
    rope_theta: float = 10000.0
    rope_factor: float = 40.0
    beta_fast: float = 32.0
    beta_slow: float = 1.0
    original_seq_len: int = 4096
    # "flex": FlexAttention, causal within each document, as torchtitan trains.
    # "sdpa": causal attention within each sequence, without compile.
    attention: str = "flex"
    activation_checkpointing: bool = True


# ---------------------------------------------------------------------------
# Rotary embedding (YaRN)
# ---------------------------------------------------------------------------


def precompute_rope_cache(config: DeepSeekV3Config) -> torch.Tensor:
    """Complex rotations ``[max_seq_len, qk_rope_head_dim / 2]`` with YaRN
    ("NTK-by-parts") frequency scaling, on the current default device."""
    dim, base = config.qk_rope_head_dim, config.rope_theta
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))

    def correction_dim(num_rotations: float) -> float:
        return (
            dim * math.log(config.original_seq_len / (num_rotations * 2 * math.pi))
        ) / (2 * math.log(base))

    low = max(math.floor(correction_dim(config.beta_fast)), 0)
    high = min(math.ceil(correction_dim(config.beta_slow)), dim - 1)
    if low == high:
        high += 0.001
    ramp = ((torch.arange(dim // 2, dtype=torch.float32) - low) / (high - low)).clamp(
        0, 1
    )
    freqs = inv_freq / config.rope_factor * ramp + inv_freq * (1 - ramp)
    t = torch.arange(config.max_seq_len, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    return torch.polar(torch.ones_like(freqs), freqs)


def apply_rotary_emb(
    q_THK: torch.Tensor, k_T1K: torch.Tensor, rope_T1R: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Rotate adjacent pairs of the query and key dims in FP32."""
    q_ = torch.view_as_complex(q_THK.float().reshape(*q_THK.shape[:-1], -1, 2))
    q_out = torch.view_as_real(q_ * rope_T1R).flatten(-2).type_as(q_THK)
    k_ = torch.view_as_complex(k_T1K.float().reshape(*k_T1K.shape[:-1], -1, 2))
    k_out = torch.view_as_real(k_ * rope_T1R).flatten(-2).type_as(k_T1K)
    return q_out, k_out


# ---------------------------------------------------------------------------
# Attention
# ---------------------------------------------------------------------------

# FlexAttention needs torch.compile. Without autotuning it compiles quickly and
# picks its kernel deterministically.
_compiled_flex_attention = torch.compile(
    flex_attention,
    options={
        "wrap_inductor_compiled_regions": True,
        "max_autotune": False,
        "coordinate_descent_tuning": False,
        "triton.cudagraphs": False,
    },
)


def document_causal_block_mask(positions_T: torch.Tensor) -> BlockMask:
    """FlexAttention mask: causal, within documents. A document starts where
    its position is 0."""
    document_T = torch.cumsum((positions_T == 0).int(), dim=0) - 1
    num_tokens = positions_T.shape[0]

    def causal(b, h, q_idx, kv_idx):
        return q_idx >= kv_idx

    def same_document(b, h, q_idx, kv_idx):
        return document_T[q_idx] == document_T[kv_idx]

    return create_block_mask(
        and_masks(causal, same_document),
        1,
        None,
        num_tokens,
        num_tokens,
        device=positions_T.device,
    )


class MultiHeadLatentAttention(nn.Module):
    """DeepSeek's MLA without query compression (``q_lora_rank = 0``): keys and
    values come from a 512-dim latent, plus a 64-dim rotary key shared by all
    heads."""

    def __init__(self, config: DeepSeekV3Config):
        super().__init__()
        self.kv_lora_rank = config.kv_lora_rank
        self.qk_nope_head_dim = config.qk_nope_head_dim
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.qk_head_dim = config.qk_nope_head_dim + config.qk_rope_head_dim
        self.v_head_dim = config.v_head_dim
        self.attention = config.attention
        dim = config.dim
        self.wq = nn.Linear(dim, config.n_heads * self.qk_head_dim, bias=False)
        self.wkv_a = nn.Linear(
            dim, config.kv_lora_rank + config.qk_rope_head_dim, bias=False
        )
        self.kv_norm = nn.RMSNorm(config.kv_lora_rank, eps=config.norm_eps)
        self.wkv_b = nn.Linear(
            config.kv_lora_rank,
            config.n_heads * (config.qk_nope_head_dim + config.v_head_dim),
            bias=False,
        )
        self.wo = nn.Linear(config.n_heads * config.v_head_dim, dim, bias=False)
        # YaRN's attention temperature, folded into the softmax scale.
        mscale = 0.1 * config.mscale * math.log(config.rope_factor) + 1.0
        self.softmax_scale = self.qk_head_dim**-0.5 * mscale * mscale

    def forward(
        self,
        x_TD: torch.Tensor,
        rope_T1R: torch.Tensor,
        batch_size: int,
        block_mask: BlockMask | None,
    ) -> torch.Tensor:
        num_tokens = x_TD.shape[0]
        q_THK = self.wq(x_TD).view(num_tokens, -1, self.qk_head_dim)
        q_nope, q_pe = torch.split(
            q_THK, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1
        )
        kv = self.wkv_a(x_TD)
        kv, k_pe = torch.split(kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        q_pe, k_pe = apply_rotary_emb(q_pe, k_pe.unsqueeze(1), rope_T1R)
        q_THK = torch.cat([q_nope, q_pe], dim=-1)

        kv = self.wkv_b(self.kv_norm(kv))
        kv = kv.view(num_tokens, -1, self.qk_nope_head_dim + self.v_head_dim)
        k_nope, v_THV = torch.split(
            kv, [self.qk_nope_head_dim, self.v_head_dim], dim=-1
        )
        k_THK = torch.cat([k_nope, k_pe.expand(-1, k_nope.size(1), -1)], dim=-1)

        if self.attention == "flex":
            out_1HTV = _compiled_flex_attention(
                q_THK.transpose(0, 1).unsqueeze(0),
                k_THK.transpose(0, 1).unsqueeze(0),
                v_THV.transpose(0, 1).unsqueeze(0),
                block_mask=block_mask,
                scale=self.softmax_scale,
            )
            out_THV = out_1HTV.squeeze(0).transpose(0, 1)
        else:
            # Flash attention needs equal query and value head dims, so MLA's
            # 192 and 128 run on cuDNN, or on the memory-efficient kernel when
            # torch.use_deterministic_algorithms rules cuDNN out.
            q, k, v = (
                t.view(batch_size, -1, *t.shape[1:]).transpose(1, 2)
                for t in (q_THK, k_THK, v_THV)
            )
            with sdpa_kernel(
                [SDPBackend.CUDNN_ATTENTION, SDPBackend.EFFICIENT_ATTENTION],
                set_priority=True,
            ):
                out_BHSV = F.scaled_dot_product_attention(
                    q, k, v, is_causal=True, scale=self.softmax_scale
                )
            out_THV = out_BHSV.transpose(1, 2).flatten(0, 1)
        return self.wo(out_THV.contiguous().view(num_tokens, -1))


# ---------------------------------------------------------------------------
# Feed-forward and experts
# ---------------------------------------------------------------------------


class GateUpLinear(nn.Module):
    """SwiGLU's gate and up projections, stored as one ``[2, F, D]`` weight."""

    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(2, hidden_dim, dim))

    def forward(self, x_TD: torch.Tensor) -> torch.Tensor:
        out_T2F = F.linear(x_TD, self.weight.flatten(0, -2))
        return out_T2F.unflatten(-1, self.weight.shape[:-1])


def swiglu(gate_up_T2F: torch.Tensor) -> torch.Tensor:
    gate_TF, up_TF = gate_up_T2F.unbind(-2)
    return F.silu(gate_TF) * up_TF


class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.w13 = GateUpLinear(dim, hidden_dim)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x_TD: torch.Tensor) -> torch.Tensor:
        return self.w2(swiglu(self.w13(x_TD)))


class GroupedLinear(nn.Module):
    """One linear per expert, applied to rows grouped by expert."""

    def __init__(self, num_experts: int, out_shape: tuple[int, ...], in_features: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(num_experts, *out_shape, in_features))

    def forward(self, x_NI: torch.Tensor, offsets_E: torch.Tensor) -> torch.Tensor:
        weight_EOI = self.weight.flatten(1, -2)
        out_NO = torch._grouped_mm(
            x_NI, weight_EOI.bfloat16().transpose(-2, -1), offs=offsets_E
        )
        return out_NO.reshape(*out_NO.shape[:-1], *self.weight.shape[1:-1])


class RoutedExperts(nn.Module):
    def __init__(self, config: DeepSeekV3Config):
        super().__init__()
        experts, dim, hidden = config.num_experts, config.dim, config.moe_hidden_dim
        self.top_k = config.top_k
        self.w13 = GroupedLinear(experts, (2, hidden), dim)
        self.w2 = GroupedLinear(experts, (dim,), hidden)
        # With expert parallelism, this rank holds a slice of the experts, and
        # the dispatcher moves the routed tokens to and from the ranks holding
        # theirs (expert_parallel.py).
        self.dispatcher = None

    def forward(
        self,
        x_TD: torch.Tensor,
        topk_scores_TK: torch.Tensor,
        topk_expert_ids_TK: torch.Tensor,
        num_tokens_per_expert_E: torch.Tensor,
    ) -> torch.Tensor:
        # Sort the (token, expert) pairs by expert, stably.
        sorted_N = torch.argsort(topk_expert_ids_TK.view(-1), stable=True)
        scores_N = topk_scores_TK.view(-1)[sorted_N]
        token_ids_N = sorted_N // self.top_k
        routed_ND = x_TD[token_ids_N]

        if self.dispatcher is None:
            routed_RD, num_tokens_per_local_expert_e = (
                routed_ND,
                num_tokens_per_expert_E,
            )
        else:
            routed_RD, num_tokens_per_local_expert_e, metadata = (
                self.dispatcher.dispatch(routed_ND, num_tokens_per_expert_E)
            )
        offsets_e = torch.cumsum(
            num_tokens_per_local_expert_e, dim=0, dtype=torch.int32
        )
        hidden_RF = swiglu(self.w13(routed_RD.bfloat16(), offsets_e))
        out_RD = self.w2(hidden_RF, offsets_e).type_as(routed_RD)
        if self.dispatcher is None:
            out_ND = out_RD
        else:
            out_ND = self.dispatcher.combine(out_RD, metadata)

        # Weight each expert output by its score, in FP32, and sum per token.
        out_ND = (out_ND * scores_N.reshape(-1, 1).to(torch.float32)).to(out_ND.dtype)
        return _ScatterAdd.apply(
            torch.zeros_like(x_TD),
            token_ids_N.reshape(-1, 1).expand(-1, x_TD.shape[-1]),
            out_ND,
        )


class _ScatterAdd(torch.autograd.Function):
    """``out.scatter_add(0, index, src)``, deterministic."""

    @staticmethod
    def forward(ctx, out, index, src):
        ctx.save_for_backward(index)
        previous = torch.are_deterministic_algorithms_enabled()
        previous_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
        torch.use_deterministic_algorithms(True)
        try:
            return out.scatter_add(0, index, src)
        finally:
            torch.use_deterministic_algorithms(previous, warn_only=previous_warn_only)

    @staticmethod
    def backward(ctx, grad):
        (index,) = ctx.saved_tensors
        return grad, None, torch.gather(grad, 0, index)


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------


class _Fp32OutputLinear(torch.autograd.Function):
    """``x @ weight.T`` from BF16 operands, accumulated and returned in FP32.

    The backward splits the FP32 output gradient into three BF16 pieces that
    sum to it exactly, so BF16 GEMMs give close to FP32 gradients (torchtitan's
    ``HiMidLoLinear`` with ``backward_mode="hi_mid_lo"``).
    """

    @staticmethod
    def forward(ctx, x_TD, weight_OD):
        ctx.use_bf16_gemm = (
            x_TD.is_cuda and x_TD.dtype == weight_OD.dtype == torch.bfloat16
        )
        ctx.save_for_backward(x_TD, weight_OD)
        if ctx.use_bf16_gemm:
            return torch.mm(x_TD, weight_OD.T, out_dtype=torch.float32)
        return torch.mm(x_TD.float(), weight_OD.float().T)

    @staticmethod
    def backward(ctx, grad_TO):
        x_TD, weight_OD = ctx.saved_tensors
        grad_TO = grad_TO.float()
        num_tokens, out_features = grad_TO.shape
        if not ctx.use_bf16_gemm:
            return (
                torch.mm(grad_TO, weight_OD.float()).to(x_TD.dtype),
                torch.mm(grad_TO.T, x_TD.float()),
            )
        if out_features > num_tokens:
            # Stack the pieces along the tokens.
            pieces_PTO = _split_into_bf16_pieces(grad_TO, dim=0)
            grad_x = torch.mm(pieces_PTO, weight_OD, out_dtype=torch.float32)
            grad_x = grad_x.unflatten(0, (3, num_tokens)).sum(dim=0).to(x_TD.dtype)
            grad_weight = torch.mm(
                pieces_PTO.T, torch.cat([x_TD] * 3), out_dtype=torch.float32
            )
            return grad_x, grad_weight
        # Stack the pieces along the outputs.
        pieces_TPO = _split_into_bf16_pieces(grad_TO, dim=1)
        grad_x = torch.mm(pieces_TPO, torch.cat([weight_OD] * 3))
        pieces = pieces_TPO.split(out_features, dim=1)
        grad_weight = torch.mm(pieces[0].T, x_TD, out_dtype=torch.float32)
        for piece in pieces[1:]:
            grad_weight += torch.mm(piece.T, x_TD, out_dtype=torch.float32)
        return grad_x, grad_weight


def _split_into_bf16_pieces(tensor: torch.Tensor, dim: int) -> torch.Tensor:
    """Split FP32 into BF16 ``hi``, ``mid`` and ``lo`` that sum to it, and
    concatenate them along ``dim``."""
    hi = tensor.to(torch.bfloat16)
    rest = tensor - hi.float()
    mid = rest.to(torch.bfloat16)
    lo = (rest - mid.float()).to(torch.bfloat16)
    return torch.cat([hi, mid, lo], dim=dim)


class RouterGate(nn.Module):
    def __init__(self, dim: int, num_experts: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(num_experts, dim))

    def forward(self, x_TD: torch.Tensor) -> torch.Tensor:
        out = _Fp32OutputLinear.apply(x_TD.reshape(-1, x_TD.shape[-1]), self.weight)
        return out.reshape(*x_TD.shape[:-1], -1)


class _InjectAuxLoss(torch.autograd.Function):
    """Return ``carrier`` unchanged; in backward, also send a gradient of one
    to ``aux_loss``, adding it to the training loss without a separate sum."""

    @staticmethod
    def forward(ctx, carrier, aux_loss):
        ctx.save_for_backward(aux_loss)
        return carrier

    @staticmethod
    def backward(ctx, grad):
        (aux_loss,) = ctx.saved_tensors
        return grad, torch.ones_like(aux_loss)


# True while torch.utils.checkpoint recomputes a block's forward.
_recomputing = False


@contextlib.contextmanager
def _recompute_context():
    global _recomputing
    _recomputing = True
    try:
        yield
    finally:
        _recomputing = False


class Router(nn.Module):
    """Top-k sigmoid routing with auxiliary-loss-free load balancing (an expert
    bias that steers selection only) and DeepSeek V3's sequence-wise balance
    loss (Eqs. 17-19), here over each microbatch."""

    def __init__(self, config: DeepSeekV3Config):
        super().__init__()
        self.gate = RouterGate(config.dim, config.num_experts)
        self.top_k = config.top_k
        self.route_norm = config.route_norm
        self.route_scale = config.route_scale
        self.aux_loss_coeff = config.aux_loss_coeff
        # Tokens each expert received since the last expert-bias update.
        self.register_buffer(
            "tokens_per_expert",
            torch.zeros(config.num_experts, dtype=torch.float32),
            persistent=False,
        )

    def forward(
        self,
        x_TD: torch.Tensor,
        expert_bias_E: torch.Tensor,
        aux_loss_denominator: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        scores_TE = torch.sigmoid(self.gate(x_TD))
        topk_expert_ids_TK = torch.topk(
            scores_TE + expert_bias_E, k=self.top_k, dim=-1, sorted=False
        ).indices
        topk_scores_TK = scores_TE.gather(dim=-1, index=topk_expert_ids_TK)
        if self.route_norm:
            topk_scores_TK = topk_scores_TK / (
                topk_scores_TK.sum(dim=-1, keepdim=True) + 1e-20
            )
        topk_scores_TK = topk_scores_TK * self.route_scale
        routing_map_TE = torch.zeros_like(scores_TE, dtype=torch.bool).scatter_(
            -1, topk_expert_ids_TK, True
        )
        if self.training:
            if not _recomputing:
                with torch.no_grad():
                    self.tokens_per_expert.add_(routing_map_TE.sum(dim=0))
            if self.aux_loss_coeff:
                topk_scores_TK = self._add_balance_loss(
                    scores_TE, routing_map_TE, topk_scores_TK, aux_loss_denominator
                )
        return topk_scores_TK, topk_expert_ids_TK, routing_map_TE

    def _add_balance_loss(
        self,
        scores_TE: torch.Tensor,
        routing_map_TE: torch.Tensor,
        topk_scores_TK: torch.Tensor,
        denominator: torch.Tensor,
    ) -> torch.Tensor:
        num_experts = scores_TE.size(-1)
        counts_E = routing_map_TE.to(scores_TE.dtype).sum(dim=0)
        f_E = F.normalize(counts_E, p=1, dim=0) * num_experts
        p_E = F.normalize(scores_TE, p=1, dim=-1).sum(dim=0)
        # Summed over the microbatch's tokens; divided below by the step's
        # token count over all ranks, like the main loss.
        balance_loss = (f_E * p_E).sum()
        scale = 1.0 / denominator.clamp_min(1)
        return _InjectAuxLoss.apply(
            topk_scores_TK, balance_loss * (self.aux_loss_coeff * scale)
        )


class MoE(nn.Module):
    def __init__(self, config: DeepSeekV3Config):
        super().__init__()
        self.routed_experts = RoutedExperts(config)
        self.router = Router(config)
        self.shared_experts = FeedForward(
            config.dim, config.moe_hidden_dim * config.num_shared_experts
        )
        self.load_balance_coeff = config.load_balance_coeff
        self.register_buffer(
            "expert_bias", torch.zeros(config.num_experts, dtype=torch.float32)
        )

    def forward(
        self, x_TD: torch.Tensor, aux_loss_denominator: torch.Tensor | None
    ) -> torch.Tensor:
        topk_scores_TK, topk_expert_ids_TK, routing_map_TE = self.router(
            x_TD, self.expert_bias, aux_loss_denominator
        )
        out_TD = self.routed_experts(
            x_TD, topk_scores_TK, topk_expert_ids_TK, routing_map_TE.sum(dim=0)
        )
        return torch.add(out_TD, self.shared_experts(x_TD))


@torch.no_grad()
def update_expert_biases(
    model: nn.Module, group: torch.distributed.ProcessGroup | None = None
) -> None:
    """Auxiliary-loss-free load balancing: move each MoE layer's expert bias by
    ``load_balance_coeff`` toward the experts that got fewer tokens than the
    mean, over all ranks, then reset the counts. Call it before each optimizer
    step."""
    moes = [module for module in model.modules() if isinstance(module, MoE)]
    if not moes:
        return
    tokens_per_expert_LE = torch.vstack([moe.router.tokens_per_expert for moe in moes])
    if torch.distributed.is_initialized():
        torch.distributed.all_reduce(tokens_per_expert_LE, group=group)
    for moe, tokens_per_expert_E in zip(moes, tokens_per_expert_LE, strict=True):
        delta_E = moe.load_balance_coeff * torch.sign(
            tokens_per_expert_E.mean() - tokens_per_expert_E
        )
        moe.expert_bias.add_(delta_E - delta_E.mean())
        moe.router.tokens_per_expert.zero_()


# ---------------------------------------------------------------------------
# Blocks and model
# ---------------------------------------------------------------------------


class TransformerBlock(nn.Module):
    def __init__(self, layer_id: int, config: DeepSeekV3Config):
        super().__init__()
        self.attention = MultiHeadLatentAttention(config)
        self.attention_norm = nn.RMSNorm(config.dim, eps=config.norm_eps)
        self.ffn_norm = nn.RMSNorm(config.dim, eps=config.norm_eps)
        self.moe_enabled = layer_id >= config.n_dense_layers
        if self.moe_enabled:
            self.moe = MoE(config)
        else:
            self.feed_forward = FeedForward(config.dim, config.dense_hidden_dim)
        self.activation_checkpointing = config.activation_checkpointing

    def forward(self, x_TD, rope_T1R, batch_size, block_mask, aux_loss_denominator):
        if not self.activation_checkpointing:
            return self._forward(
                x_TD, rope_T1R, batch_size, block_mask, aux_loss_denominator
            )
        return checkpoint(
            self._forward,
            x_TD,
            rope_T1R,
            batch_size,
            block_mask,
            aux_loss_denominator,
            use_reentrant=False,
            context_fn=lambda: (contextlib.nullcontext(), _recompute_context()),
        )

    def _forward(self, x_TD, rope_T1R, batch_size, block_mask, aux_loss_denominator):
        x_TD = x_TD + self.attention(
            self.attention_norm(x_TD), rope_T1R, batch_size, block_mask
        )
        if self.moe_enabled:
            ffn_TD = self.moe(self.ffn_norm(x_TD), aux_loss_denominator)
        else:
            ffn_TD = self.feed_forward(self.ffn_norm(x_TD))
        return torch.add(x_TD, ffn_TD)


class DeepSeekV3(nn.Module):
    def __init__(self, config: DeepSeekV3Config):
        super().__init__()
        self.config = config
        self.tok_embeddings = nn.Embedding(config.vocab_size, config.dim)
        self.layers = nn.ModuleList(
            TransformerBlock(layer_id, config) for layer_id in range(config.n_layers)
        )
        self.norm = nn.RMSNorm(config.dim, eps=config.norm_eps)
        self.lm_head = nn.Linear(config.dim, config.vocab_size, bias=False)
        self.register_buffer(
            "rope_cache", precompute_rope_cache(config), persistent=False
        )

    def forward(
        self,
        tokens_BS: torch.Tensor,
        positions_BS: torch.Tensor | None = None,
        aux_loss_denominator: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return logits ``[B * S, vocab_size]``.

        ``positions_BS`` index the rotary embedding; by default each sequence
        counts from 0. With FlexAttention, a position of 0 also starts a new
        document. ``aux_loss_denominator`` is the step's token count over all
        ranks, which the training loss is divided by too.
        """
        batch_size = tokens_BS.shape[0]
        if positions_BS is None:
            positions_BS = torch.arange(
                tokens_BS.shape[1], device=tokens_BS.device
            ).expand_as(tokens_BS)
        positions_T = positions_BS.flatten()
        num_tokens = positions_T.shape[0]
        rope_T1R = self.rope_cache[positions_T].view(num_tokens, 1, -1)
        block_mask = (
            document_causal_block_mask(positions_T)
            if self.config.attention == "flex"
            else None
        )
        h_TD = self.tok_embeddings(tokens_BS.flatten())
        for layer in self.layers:
            h_TD = layer(h_TD, rope_T1R, batch_size, block_mask, aux_loss_denominator)
        return self.lm_head(self.norm(h_TD))

    @torch.no_grad()
    def init_buffers(self, device: torch.device) -> None:
        """(Re)create the buffers on ``device``, as after ``to_empty``."""
        with torch.device(device):
            self.rope_cache = precompute_rope_cache(self.config)
        for module in self.modules():
            if isinstance(module, MoE):
                module.expert_bias.zero_()
                module.router.tokens_per_expert.zero_()


def param_init_fns(config: DeepSeekV3Config) -> dict[str, Callable]:
    """In-place initializers by parameter name, torchtitan's for DeepSeek V3.

    Each fills a full-shape tensor; output projections scale their standard
    deviation down with depth.
    """
    trunc_normal = nn.init.trunc_normal_
    linear = partial(trunc_normal, std=0.02)

    def depth(layer_id: int):
        return partial(trunc_normal, std=0.02 / (2 * (layer_id + 1)) ** 0.5)

    def gate_up(gate_init, up_init, experts: bool):
        def init(weight):
            gate_init(weight[:, 0] if experts else weight[0])
            up_init(weight[:, 1] if experts else weight[1])

        return init

    s = config.dim**-0.5
    fns = {
        "tok_embeddings.weight": partial(nn.init.normal_, std=1.0),
        "norm.weight": nn.init.ones_,
        "lm_head.weight": partial(trunc_normal, std=s, a=-3 * s, b=3 * s),
    }
    for layer_id in range(config.n_layers):
        prefix = f"layers.{layer_id}"
        fns.update(
            {
                f"{prefix}.attention.wq.weight": linear,
                f"{prefix}.attention.wkv_a.weight": linear,
                f"{prefix}.attention.kv_norm.weight": nn.init.ones_,
                f"{prefix}.attention.wkv_b.weight": linear,
                f"{prefix}.attention.wo.weight": depth(layer_id),
                f"{prefix}.attention_norm.weight": nn.init.ones_,
                f"{prefix}.ffn_norm.weight": nn.init.ones_,
            }
        )
        ffns = ["feed_forward"]
        if layer_id >= config.n_dense_layers:
            ffns = ["moe.shared_experts"]
            fns.update(
                {
                    f"{prefix}.moe.router.gate.weight": depth(layer_id),
                    f"{prefix}.moe.routed_experts.w13.weight": gate_up(
                        linear, depth(layer_id), experts=True
                    ),
                    f"{prefix}.moe.routed_experts.w2.weight": depth(layer_id),
                }
            )
        for ffn in ffns:
            fns[f"{prefix}.{ffn}.w13.weight"] = gate_up(
                linear, depth(layer_id), experts=False
            )
            fns[f"{prefix}.{ffn}.w2.weight"] = depth(layer_id)
    return fns
