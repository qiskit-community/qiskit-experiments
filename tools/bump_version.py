#!/usr/bin/env python3

# This code is part of Qiskit.
#
# (C) Copyright IBM 2025.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Bump package minor version"""

# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "packaging",
# ]
# ///

from pathlib import Path
from subprocess import run

from packaging.version import Version


def replace_text(path: Path, old: str, new: str, count: int = 1):
    """Replace old string with new in a file

    Args:
        path: Path to the file to edit
        old: Old string to replace
        new: New string to insert
        count: How many replacements to expect

    Raises:
        ValueError: the number of replacements does not match ``count``
    """
    lines = path.read_text().splitlines()

    match_num = 0
    for idx, line in enumerate(lines):
        if old in line:
            match_num += 1
            start = line.index(old)
            lines[idx] = line[:start] + new + line[start + len(old) :]

    if match_num != count:
        raise ValueError(f"Expected {count} matches for '{old}' in {path} but found {match_num}")

    path.write_text("\n".join(lines) + "\n")


def main():
    """Bump minor version in package files"""
    proc = run(["git", "rev-parse", "--show-toplevel"], check=True, capture_output=True, text=True)
    git_root = Path(proc.stdout.strip())

    version_file = git_root / "qiskit_experiments/VERSION.txt"
    version_str = version_file.read_text().strip()
    old_version = Version(version_str)
    old_version_short = f"{old_version.major}.{old_version.minor}"
    prev_old_version = f"{old_version.major}.{old_version.minor - 1}.0"
    prev_old_version_short = f"{old_version.major}.{old_version.minor - 1}"

    new_version = f"{old_version.major}.{old_version.minor + 1}.0"
    new_version_short = f"{old_version.major}.{old_version.minor + 1}"

    version_file.write_text(f"{new_version}\n")

    replace_text(
        git_root / "docs/release_notes.rst",
        f":earliest-version: {prev_old_version}",
        f":earliest-version: {old_version}",
    )
    replace_text(
        git_root / "docs/conf.py",
        f'release = os.getenv("RELEASE_STRING", "{old_version}")',
        f'release = os.getenv("RELEASE_STRING", "{new_version}")',
    )
    replace_text(
        git_root / "docs/conf.py",
        f'version = os.getenv("VERSION_STRING", "{old_version_short}")',
        f'version = os.getenv("VERSION_STRING", "{new_version_short}")',
    )
    replace_text(
        git_root / ".mergify.yml",
        f"- stable/{prev_old_version_short}",
        f"- stable/{old_version_short}",
    )


if __name__ == "__main__":
    main()
