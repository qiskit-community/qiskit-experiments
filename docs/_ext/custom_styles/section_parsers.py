# This code is part of Qiskit.
#
# (C) Copyright IBM 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
Documentation section parsers.
"""

import re
from typing import List

from .utils import _trim_empty_lines


def load_standard_section(docstring_lines: List[str]) -> List[str]:
    """Load standard docstring section."""
    return _trim_empty_lines(docstring_lines)


def load_fit_parameters(docstring_lines: List[str]) -> List[str]:
    """Load fit parameter section."""
    regex_paramdef = re.compile(r"defpar (?P<param>.+):")

    # item finder
    description_kind = {
        "desc": re.compile(r"desc: (?P<s>.+)"),
        "init_guess": re.compile(r"init_guess: (?P<s>.+)"),
        "bounds": re.compile(r"bounds: (?P<s>.+)"),
    }

    # parse lines
    parameter_desc = dict()
    current_param = None
    current_item = None
    for line in docstring_lines:
        if not list:
            # remove line feed
            continue

        # check if line is new parameter definition
        match = re.match(regex_paramdef, line)
        if match:
            current_param = match["param"]
            parameter_desc[current_param] = {
                "desc": "",
                "init_guess": "",
                "bounds": "",
            }
            continue

        # check description
        for kind, regex in description_kind.items():
            match = re.search(regex, line)
            if match:
                current_item = kind
                line = match["s"].rstrip()

        # add line if parameter and item are already set
        if current_param and current_item:
            if parameter_desc[current_param][current_item]:
                parameter_desc[current_param][current_item] += " " + line.lstrip()
            else:
                parameter_desc[current_param][current_item] = line.lstrip()

    section_lines = list()

    def write_description(header: str, kind: str):
        section_lines.append(header)
        for param, desc in parameter_desc.items():
            if not desc:
                section_lines.append(
                    f"    - :math:`{param}`: No description is provided. See source for details."
                )
            else:
                section_lines.append(f"    - :math:`{param}`: {desc[kind]}")
        section_lines.append("")

    write_description("Descriptions", "desc")
    write_description("Initial Guess", "init_guess")
    write_description("Boundaries", "bounds")

    return _trim_empty_lines(section_lines)
