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
A collection of utilities to generate documentation.
"""

import inspect
import re
from typing import List, Tuple, Dict, Any, Callable

from sphinx.config import Config as SphinxConfig
from sphinx.ext.napoleon.docstring import GoogleDocstring
from sphinx.util.docstrings import prepare_docstring


def _trim_empty_lines(docstring_lines: List[str]) -> List[str]:
    """A helper function to remove redundant line feeds."""
    i_start = 0
    lines_iter = iter(docstring_lines)
    while not next(lines_iter):
        i_start += 1

    i_end = len(docstring_lines)
    lines_iter = iter(docstring_lines[::-1])
    while not next(lines_iter):
        i_end -= 1

    return docstring_lines[i_start:i_end]


def _parse_option_field(
    docstring: str,
    config: SphinxConfig,
    target_args: List[str],
    indent: str = "",
) -> Tuple[List[str], List[str]]:
    """A helper function to extract descriptions of target arguments."""

    # use GoogleDocstring parameter parser
    experiment_option_parser = GoogleDocstring(
        docstring=prepare_docstring(docstring, tabsize=len(indent)), config=config
    )
    parsed_lines = experiment_option_parser.lines()

    # remove redundant descriptions
    param_regex = re.compile(r":(param|type) (?P<pname>\S+):")
    target_params_description = []
    described_params = set()
    valid_line = False
    for line in parsed_lines:
        is_item = re.match(param_regex, line)
        if is_item:
            if is_item["pname"] in target_args:
                valid_line = True
                described_params.add(is_item["pname"])
            else:
                valid_line = False
        if valid_line:
            target_params_description.append(line)

    # find missing parameters
    missing = set(target_args) - described_params

    return target_params_description, list(missing)


def _generate_options_documentation(
    current_class: object,
    method_name: str,
    target_args: List[str] = None,
    config: SphinxConfig = None,
    indent: str = "",
) -> List[str]:
    """Automatically generate documentation from the default options method."""

    if current_class == object:
        # check if no more base class
        raise Exception(f"Option docstring for {', '.join(target_args)} is missing.")

    options_docstring_lines = []

    default_opts = getattr(current_class, method_name, None)
    if not default_opts:
        # getter option is not defined
        return []

    if not target_args:
        target_args = list(default_opts().__dict__.keys())

    # parse default options method
    parsed_lines, target_args = _parse_option_field(
        docstring=default_opts.__doc__ or "",
        config=config,
        target_args=target_args,
        indent=indent,
    )

    if target_args:
        # parse parent class method docstring if some arg documentation is missing
        parent_parsed_lines = _generate_options_documentation(
            current_class=inspect.getmro(current_class)[1],
            method_name=method_name,
            target_args=target_args,
            config=config,
            indent=indent,
        )
        options_docstring_lines.extend(parent_parsed_lines)

    options_docstring_lines.extend(parsed_lines)

    if options_docstring_lines:
        return _trim_empty_lines(options_docstring_lines)

    return options_docstring_lines


def _format_default_options(defaults: Dict[str, Any], indent: str = "") -> List[str]:
    """Format default options to docstring lines."""
    docstring_lines = [
        ".. dropdown:: Default values",
        indent + ":animate: fade-in-slide-down",
        "",
    ]

    if not defaults:
        docstring_lines.append(indent + "No default options are set.")
    else:
        docstring_lines.append(indent + "Following values are set by default.")
        docstring_lines.append("")
        docstring_lines.append(indent + ".. parsed-literal::")
        docstring_lines.append("")
        for par, value in defaults.items():
            if callable(value):
                value_repr = f"Callable {value.__name__}"
            else:
                value_repr = repr(value)
            docstring_lines.append(indent * 2 + f"{par:<25} := {value_repr}")

    return docstring_lines


def _check_no_indent(method: Callable) -> Callable:
    """Check indent of lines and return if this block is correctly indented."""
    def wraps(self, lines: List[str], *args, **kwargs):
        if all(l.startswith(" ") for l in lines):
            text_block = "\n".join(lines)
            raise ValueError(
                "Following documentation may have invalid indentation. "
                f"Please carefully check all indent levels are aligned. \n\n{text_block}"
            )
        return method(self, lines, *args, **kwargs)

    return wraps
