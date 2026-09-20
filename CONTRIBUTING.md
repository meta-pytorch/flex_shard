# Contributing to FlexShard

We want to make contributing to this project as easy and transparent as
possible. Please follow our [Code of Conduct](CODE_OF_CONDUCT.md).

## Our development process

Submit contributions as GitHub pull requests against `main`. Maintainers review
pull requests and verify the relevant tests and required checks before merging.
External contributors do not need access to Meta's internal tools.

When GitHub merges are enabled, maintainers squash-merge approved pull requests.
Merged changes are synchronized into Meta's internal repository for internal
validation and landing. Until GitHub merges are enabled, maintainers import
pull requests and land them internally; synchronization then closes the linked
pull request.

Use ordinary GitHub pull requests; ghstack is not supported. Keep public code,
documentation, and configuration synchronized between the two repositories.

## Pull requests

We actively welcome your pull requests.

1. Fork the repository and create your branch from `main`.
2. Keep the change focused and explain the problem it solves.
3. Add tests for new behavior and update documentation for API changes.
4. Ensure the relevant tests pass and follow the existing coding style.
5. Include the test commands, dependency versions, and GPU count in your test
   results. Report skipped tests and tests you could not run.
6. If you have not already done so, complete the Contributor License Agreement
   (CLA).

## Development setup and tests

Use the Python, PyTorch, CUDA, and Triton requirements in [README.md](README.md).
From the repository root, install the development dependencies and run tests:

```bash
python -m pip install -e '.[test]'
python -m pytest
```

The suite includes CPU tests and CUDA/NCCL tests. Some distributed tests require
four GPUs and skip on hosts with insufficient hardware. A CPU-only run is
useful for applicable unit tests, but does not validate the training runtime.
Maintainers can help arrange GPU validation before landing a contribution.

Follow the surrounding Python style, use four spaces for indentation, and
preserve copyright and license headers. Maintainers preserve GitHub formatting
when importing changes and run the applicable internal build and test checks
before landing them.

## Contributor License Agreement (CLA)

In order to accept your pull request, we need you to submit a CLA. You only need
to do this once to work on any of Meta's open source projects.

Complete your CLA here: <https://code.facebook.com/cla>.

## Issues

We use GitHub issues to track public bugs. Include reproduction steps, expected
and actual behavior, dependency versions, and relevant hardware information.

Meta has a [bounty program](https://bugbounty.meta.com/) for the safe disclosure
of security bugs. In those cases, please follow that process and do not file a
public issue.

## License

By contributing to FlexShard, you agree that your contributions will be
licensed under the [LICENSE](LICENSE) file in the root directory of this source
tree.
