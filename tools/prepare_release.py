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

"""
This script does the following:

1. Run `reno report` on most recent version tag *before* the version in
   `qiskit_experiments/VERSION.txt`.
2. Copy the `reno` output into the top of `docs/release_notes.rst` (under
   the header).
3. Delete any release notes in `releasenotes/notes` that were included in a
   previous release.
4. Create a new `prep-<version>` git branch.
5. Create a new release note file with a name starting with `prep-<version>`.

If it detects a state that it does not expect (like a non-clean working
directory or a previously existing `prep-<version>` branch), it exits early.

It is expected that this script is run with a clean working directory when the
repository is ready (other than writing the release notes prelude and making
any other release notes edits) to tag the next qiskit-experiments release with
the version in `qiskit_experiments/VERSION.txt`.
"""
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "packaging",
# ]
# ///

# Note that the script requires git and reno as commands on the system path

from pathlib import Path
from shutil import rmtree
from subprocess import run
from textwrap import fill

from packaging.version import InvalidVersion, Version


def main():
    """Prepare minor release"""
    proc = run(["git", "diff", "--quiet"], check=False, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"Working directory not clean\n{proc.stdout}\n{proc.stderr}")

    proc = run(["git", "rev-parse", "--show-toplevel"], check=True, capture_output=True, text=True)
    git_root = Path(proc.stdout.strip())

    version_str = (git_root / "qiskit_experiments/VERSION.txt").read_text().strip()
    version = Version(version_str)

    # Create branch now so we can error early if it already exists
    prep_branch = f"prep-{version}"
    proc = run(["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True)
    orig_commit = proc.stdout.strip()
    proc = run(["git", "rev-parse", prep_branch], check=False, capture_output=True, text=True)
    if proc.returncode == 0:
        # prep branch already exists, maybe from a previous run that failed
        prep_branch_hash = proc.stdout.strip()
        if prep_branch_hash != orig_commit:
            raise RuntimeError(f"Branch {prep_branch} already exists and is not current commit!")
    else:
        run(["git", "branch", prep_branch], check=True)

    proc = run(["git", "tag", "--list"], check=True, capture_output=True, text=True)
    previous_version = None
    for tag in proc.stdout.splitlines():
        try:
            tag_version = Version(tag)
        except InvalidVersion:
            continue

        if tag_version < version and (previous_version is None or tag_version > previous_version):
            previous_version = tag_version

    if previous_version is None:
        raise RuntimeError(
            f"Could not find previous version tag for current version {version}:\n{proc.stdout}"
        )

    print(f"Starting from commit {orig_commit[:8]} (branch {prep_branch})...\n")

    # Switch to previous release to render notes file and get list of
    # individual notes to remove
    print("Temporarily switching git to previous release commit...")
    run(["git", "-c", "advice.detachedHead=False", "checkout", str(previous_version)], check=True)
    print("")

    proc = run(["reno", "report"], check=True, capture_output=True, text=True)
    notes_last_release = proc.stdout

    old_note_files = set((git_root / "releasenotes/notes").iterdir())

    # Back to prep branch to apply notes updates
    print(f"Switching git to {prep_branch} branch...")
    run(["git", "checkout", prep_branch], check=True)
    print("")

    # Do some surgery to cut note content from current notes file and the file
    # generated on the last release tag and join the two together with one
    # header and an updated release-notes directive.
    release_notes_file = git_root / "docs/release_notes.rst"
    notes_older = release_notes_file.read_text()

    insert_idx = 0
    for idx, line in enumerate(notes_older.splitlines()):
        if line == ".. release-notes::":
            # +3 = (release-notes line) + (earliest-version option) + (newline)
            insert_idx = idx + 3
            break
    else:
        raise RuntimeError(f"Could not find release-notes:: directive in {release_notes_file}")

    header_len = 4  # 4 = "===" + "Release Notes" + "===" + (empty line)
    updated_notes = "\n".join(
        [
            *notes_older.splitlines()[:insert_idx],
            *notes_last_release.splitlines()[header_len:],
            *notes_older.splitlines()[insert_idx:],
        ]
    )
    print(
        f"Updating {release_notes_file.relative_to(git_root)} to include notes "
        f"from {previous_version}...\n"
    )
    release_notes_file.write_text(updated_notes)

    for path in (git_root / "releasenotes/notes").iterdir():
        if path in old_note_files:
            if path.is_file():
                path.unlink()
            elif path.is_dir():
                rmtree(path)
            else:
                print(f"What is this file? {path}")

    print(f"Creating a new release note with name starting with prepare-{version}...")
    run(["reno", "new", f"prepare-{version}"], check=True)

    print("\n*********\n")
    print(
        fill(
            f"Release notes prepared. Edit the prepare-{version} note to include "
            "a prelude for the release and then commit changes if they look okay!"
        )
    )


if __name__ == "__main__":
    main()
