#!/usr/bin/env python3

# This code is part of Qiskit Experiments.
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

"""Check that the highest git version tag on the repo matches the current version"""

import re
import subprocess
import sys
from pathlib import Path


version_pattern = re.compile(r"^\d+(\.\d+)*$")


def _main():
    """Return True if VERSION.txt matches a tag higher than any other version tag

    NOTE: this function is primarily intended for the docs publishing automated
    workflow. It retruns true even if the current commit is not tagged with the
    latest version tag. It works this way so that a follow up commit can be
    pushed to fix a possible error in the docs publishing workflow without
    needing to tag again with a higher verison number just for the docs.
    """
    proc = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"], check=True, text=True, capture_output=True
    )
    repo_base = Path(proc.stdout.strip())

    proc = subprocess.run(["git", "tag", "--list"], check=True, text=True, capture_output=True)
    all_tags = [t.strip() for t in proc.stdout.splitlines()]
    all_tags = [t for t in all_tags if version_pattern.match(t)]
    highest_version = max(all_tags, key=lambda t: tuple(int(p) for p in t.split(".")))

    version = (repo_base / "qiskit_experiments/VERSION.txt").read_text().strip()

    if version != highest_version:
        return False

    return True


if __name__ == "__main__":
    if not _main():
        sys.exit(1)
