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
from typing import List, Set, Tuple, Dict, Any, Callable, Type

from sphinx.config import Config as SphinxConfig
from sphinx.ext.napoleon.docstring import GoogleDocstring
from sphinx.util.docstrings import prepare_docstring

from qiskit_experiments.framework import BaseExperiment


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
    target_args: Set[str],
    indent: str = "",
) -> Tuple[List[str], Set[str]]:
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
    missing = target_args - described_params

    return target_params_description, missing


def _generate_options_documentation(
    current_class: Type,
    method_name: str,
    target_args: List[str] = None,
    config: SphinxConfig = None,
    indent: str = "",
) -> List[str]:
    """Automatically generate documentation from the default options method."""
    options_docstring_lines = []

    if not target_args:
        default_opts_clsmethod = getattr(current_class, method_name, None)
        if not default_opts_clsmethod:
            # getter option is not defined
            return []
        target_args = set(default_opts_clsmethod().__dict__.keys())

    mro_classes = inspect.getmro(current_class)
    for i, mro_cls in enumerate(mro_classes):
        default_opts_clsmethod = getattr(mro_cls, method_name, None)
        if not default_opts_clsmethod:
            continue
        parsed_lines, target_args = _parse_option_field(
            docstring=default_opts_clsmethod.__doc__ or "",
            config=config,
            target_args=target_args,
            indent=indent,
        )
        if parsed_lines:
            if i == 0:
                description = "defined in the current class"
            else:
                description = "inherited from the parent class"
            options_docstring_lines.extend(
                [
                    f"(Options {description} :class:`.{mro_cls.__name__}`)",
                    "",
                ]
            )
            options_docstring_lines.extend(parsed_lines)
        if not target_args:
            break
    else:
        # Investigated all parent classes but all args are not described.
        raise Exception(
            f"Option documentation for {', '.join(target_args)} is missing "
            f"for the class {mro_classes[0].__name__}."
        )

    if options_docstring_lines:
        return _trim_empty_lines(options_docstring_lines)

    return options_docstring_lines


def _generate_analysis_ref(
    current_class: object,
    config: SphinxConfig = None,
    indent: str = "",
) -> List[str]:
    """Automatically generate analysis class reference with recursive ref to superclass."""

    if not issubclass(current_class, BaseExperiment):
        # check if no more base class
        raise TypeError("This is not valid experiment class.")

    experiment_option_parser = GoogleDocstring(
        docstring=prepare_docstring(current_class.__doc__, tabsize=len(indent)),
        config=config,
    )
    lines = list(map(lambda l: l.strip(), experiment_option_parser.lines()))

    analysis_ref_start = None
    try:
        analysis_ref_start = lines.index("# section: analysis_ref")
    except ValueError:
        super_classes = getattr(current_class, "__bases__")
        for super_cls in super_classes:
            try:
                return _generate_analysis_ref(
                    current_class=super_cls,
                    config=config,
                    indent=indent,
                )
            except Exception:
                pass

    if analysis_ref_start is None:
        raise Exception(f"Option docstring for analysis_ref is missing.")

    analysis_ref_lines = []
    for line in lines[analysis_ref_start + 1 :]:
        # add lines until hitting to next section
        if line.startswith("# section:"):
            break
        analysis_ref_lines.append(line)

    return analysis_ref_lines


def _format_default_options(defaults: Dict[str, Any], indent: str = "") -> List[str]:
    """Format default options to docstring lines."""
    docstring_lines = [
        ".. dropdown:: Default values",
        indent + ":animate: fade-in-slide-down",
        "",
    ]

    if not defaults:
        docstring_lines.append(indent + "No default  options are set.")
    else:
        docstring_lines.append(indent + "The following values are set by default.")
        docstring_lines.append("")
        docstring_lines.append(indent + ".. parsed-literal::")
        docstring_lines.append("")
        for par, value in defaults.items():
            if callable(value):
                if value.__class__.__name__ == "function":
                    # callback function
                    value_repr = f"Callable {value.__name__}"
                else:
                    # class instance with call method
                    value_repr = repr(value)
            else:
                value_repr = repr(value)
            docstring_lines.append(indent * 2 + f"{par:<25} := {value_repr}")
    docstring_lines.insert(0, "")
    docstring_lines.append("")

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
