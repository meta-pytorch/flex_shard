# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Check that the wheel and source distribution contain the public project."""

from __future__ import annotations

import argparse
import tarfile
import zipfile
from collections.abc import Mapping
from pathlib import Path, PurePosixPath


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def check_paths(files: Mapping[str, bytes]) -> None:
    for name in files:
        path = PurePosixPath(name)
        require(
            not path.is_absolute() and ".." not in path.parts, f"Unsafe path: {name}"
        )
        require(
            not {
                "BUCK",
                "TARGETS",
                "PACKAGE",
                "fb",
                "fbcode",
                "fbsource",
                "__pycache__",
            }.intersection(path.parts)
            and path.suffix != ".pyc",
            f"Unexpected file in distribution: {name}",
        )


def read_archives(dist_dir: Path) -> tuple[dict[str, bytes], dict[str, bytes]]:
    artifacts = list(dist_dir.iterdir())
    wheels = [path for path in artifacts if path.name.endswith(".whl")]
    sdists = [path for path in artifacts if path.name.endswith(".tar.gz")]
    require(
        len(wheels) == 1 and len(sdists) == 1, "Expected exactly one wheel and sdist"
    )

    with zipfile.ZipFile(wheels[0]) as archive:
        wheel = {
            entry.filename: archive.read(entry)
            for entry in archive.infolist()
            if not entry.is_dir()
        }
    with tarfile.open(sdists[0]) as archive:
        sdist = {}
        for entry in archive.getmembers():
            if entry.isdir():
                continue
            require(entry.isfile(), f"Unexpected archive entry: {entry.name}")
            stream = archive.extractfile(entry)
            if stream is None:
                raise ValueError(f"Cannot read archive entry: {entry.name}")
            sdist[entry.name] = stream.read()

    check_paths(wheel)
    check_paths(sdist)
    roots = {PurePosixPath(name).parts[0] for name in sdist}
    require(len(roots) == 1, "Source distribution must have one root directory")
    root = roots.pop()
    sdist = {
        PurePosixPath(name).relative_to(root).as_posix(): contents
        for name, contents in sdist.items()
    }
    return wheel, sdist


def check_documentation(
    wheel: Mapping[str, bytes], sdist: Mapping[str, bytes], source_root: Path
) -> None:
    for name in (
        "LICENSE",
        "README.md",
        "CONTRIBUTING.md",
        "CODE_OF_CONDUCT.md",
        "pyproject.toml",
    ):
        require(bool(sdist.get(name)), f"Source distribution is missing {name}")
        require(
            sdist[name] == (source_root / name).read_bytes(),
            f"Source distribution differs from source: {name}",
        )
    licenses = [name for name in wheel if name.endswith(".dist-info/licenses/LICENSE")]
    require(
        len(licenses) == 1, "Wheel must contain its LICENSE in distribution metadata"
    )
    require(wheel[licenses[0]] == sdist["LICENSE"], "Wheel and sdist licenses differ")


def check_modules(
    wheel: Mapping[str, bytes], sdist: Mapping[str, bytes], source_root: Path
) -> int:
    source_dir = source_root / "src"
    expected = {
        path.relative_to(source_dir).as_posix(): path.read_bytes()
        for path in (source_dir / "flex_shard").rglob("*.py")
    }
    require(
        "flex_shard/__init__.py" in expected,
        "Source checkout has no flex_shard package",
    )
    for kind, files, prefix in (("wheel", wheel, ""), ("sdist", sdist, "src/")):
        expected_files = {prefix + name: data for name, data in expected.items()}
        modules = {name: data for name, data in files.items() if name.endswith(".py")}
        missing = sorted(expected_files.keys() - modules.keys())
        extra = sorted(modules.keys() - expected_files.keys())
        require(
            not missing and not extra,
            f"{kind} module mismatch: missing={missing}, extra={extra}",
        )
        for name, contents in modules.items():
            require(
                contents == expected_files[name], f"{kind} differs from source: {name}"
            )
            require(
                b"pytorch.flex_shard" not in contents,
                f"{kind} contains a non-public package import: {name}",
            )
    return len(expected)


def check_dist(dist_dir: Path, source_root: Path) -> None:
    wheel, sdist = read_archives(dist_dir)
    check_documentation(wheel, sdist, source_root)
    count = check_modules(wheel, sdist, source_root)
    print(f"Validated {count} Python modules, licenses, and source documentation")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dist_dir", type=Path, nargs="?", default=Path("dist"))
    parser.add_argument(
        "--source-root", type=Path, default=Path(__file__).resolve().parents[2]
    )
    args = parser.parse_args()
    try:
        check_dist(args.dist_dir, args.source_root)
    except (OSError, ValueError, tarfile.TarError, zipfile.BadZipFile) as error:
        parser.exit(1, f"Distribution validation failed: {error}\n")


if __name__ == "__main__":
    main()
