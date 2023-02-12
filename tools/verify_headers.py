#!/usr/bin/env python3
# This code is part of Qiskit.
#
# (C) Copyright IBM 2020
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
Verify that headers of Python files comply with the rules
"""

import argparse
import multiprocessing
import os
import sys
import re

# regex for character encoding from PEP 263
pep263 = re.compile(r"^[ \t\f]*#.*?coding[:=][ \t]*([-_.a-zA-Z0-9]+)")


def discover_files(code_paths):
    """
    Find all Python files in the given paths
    """
    out_paths = []
    for path in code_paths:
        if os.path.isfile(path):
            out_paths.append(path)
        else:
            for directory in os.walk(path):
                dir_path = directory[0]
                for subfile in directory[2]:
                    if subfile.endswith(".py") or subfile.endswith(".pyx"):
                        out_paths.append(os.path.join(dir_path, subfile))
    return out_paths


def validate_header(file_path):
    """
    Check if the file header complies with the rules
    """
    header = """# This code is part of Qiskit.
#
"""
    apache_text = """#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
"""
    count = 0
    with open(file_path, encoding="utf8") as fd:
        lines = fd.readlines()
    start = 0
    for index, line in enumerate(lines):
        count += 1
        if count > 5:
            return file_path, False, "Header not found in first 5 lines"
        if count <= 2 and pep263.match(line):
            return file_path, False, "Unnecessary encoding specification (PEP 263, 3120)"
        if line == "# This code is part of Qiskit.\n":
            start = index
            break
    if "".join(lines[start : start + 2]) != header:
        return (file_path, False, f"Header up to copyright line does not match: {header}")
    if not lines[start + 2].startswith("# (C) Copyright IBM 20"):
        return (file_path, False, "Header copyright line not found")
    if "".join(lines[start + 3 : start + 11]) != apache_text:
        return (file_path, False, f"Header apache text string doesn't match:\n {apache_text}")
    return (file_path, True, None)


def main():
    """
    Run the verifier
    """
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    default_paths = [os.path.join(root_dir, "qiskit_experiments"), os.path.join(root_dir, "test")]
    parser = argparse.ArgumentParser(description="Check file headers.")
    parser.add_argument("paths", type=str, nargs="*", default=default_paths)
    args = parser.parse_args()
    files = discover_files(args.paths)
    with multiprocessing.Pool() as pool:
        res = pool.map(validate_header, files)
    failed_files = [x for x in res if x[1] is False]
    if len(failed_files) > 0:
        for failed_file in failed_files:
            sys.stderr.write(f"{failed_file[0]} failed header check because:\n")
            sys.stderr.write(f"{failed_file[2]}\n\n")
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()
