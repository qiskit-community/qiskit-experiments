# This code is part of Qiskit.
#
# (C) Copyright IBM 2021, 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
A class to recursively collect option documentation from current class.
"""

import copy
import inspect
import re
from itertools import chain
from typing import Any

import numpy as np
from sphinx.application import Sphinx
from sphinx.ext.autodoc import Options as SphinxOptions
from sphinx.ext.napoleon import Config as NapoleonConfig
from sphinx.ext.napoleon import GoogleDocstring
from qiskit.providers import Options


_parameter_doc_regex = re.compile(r"(.+?)\(\s*(.*[^\s]+)\s*\):(.*[^\s]+)")


def _inject_roles(
    lines: list[str],
    default_opt_values: Options = None,
    desc_sources: dict[str, str] | None = None,
) -> list[str]:
    """Post-process GoogleDocstring output to inject custom roles.
    
    This function loops through a Sphinx reStructuredText function signature docstring
    and adds extra :default_val: and :source: roles at the end of each parameter's group.
    
    Example:
        Input lines::
        
            :param delay: Delay times
            :type delay: float
            :param shots: Number of shots
            :type shots: int
        
        Output lines::
        
            :param delay: Delay time
            :type delay: float
            :default_val delay: 1.0
            :source delay: :class:`~.MyClass`
            :param shots: Number of shots
            :type shots: int
            :default_val shots: 1024
            :source shots: :class:`~.MyClass`
    
    Args:
        lines: Sphinx reStructuredText function argument docstring lines from
            (such as output by GoogleDocstring.lines()).
        default_opt_values: Options object containing default values for parameters.
        desc_sources: Dictionary mapping parameter names to their fully qualified source class
            names.
        
    Returns:
        Modified list of lines with custom roles injected.
    """
    # Pattern to match :param lines
    param_pattern = re.compile(r'^:param\s+(\w+):')
    
    result = []
    current_param = None
    
    # Use chain to append None as sentinel value to trigger last parameter processing
    for line in chain(lines, [None]):
        end_of_docstring = line is None
        
        # Check if this is a :param line (None won't match)
        param_match = param_pattern.match(line) if not end_of_docstring else None
        
        if param_match or end_of_docstring:
            # If we already have a current_param, inject its custom roles first
            if current_param is not None:
                if default_opt_values is not None:
                    value = _value_repr(default_opt_values.get(current_param, None))
                    result.append(f":default_val {current_param}: {value}")
                
                if desc_sources is not None and current_param in desc_sources:
                    source = desc_sources[current_param]
                    result.append(f":source {current_param}: :class:`~.{source}`")
                else:
                    result.append(f":source {current_param}: Unknown class")
            
            # Update current_param to the new parameter (or None if sentinel)
            current_param = param_match.group(1) if param_match else None
        
        # Add the current line to result (skip the sentinel None)
        if not end_of_docstring:
            result.append(line)
    
    return result


def process_default_options(
    current_class: type,
    default_option_method: str,
    section_repr: str,
    app: Sphinx,
    options: SphinxOptions,
    config: NapoleonConfig,
    indent: str | None = "",
):
    """A helper function to generate docstring for default options."""
    default_clsmethod = getattr(current_class, default_option_method, None)
    if not default_clsmethod:
        return []
    default_options = default_clsmethod()
    target_args = set(default_options.__dict__.keys())

    descriptions = ["Parameters:"]
    desc_sources = {}
    for mro_class in inspect.getmro(current_class):
        if default_option_method not in mro_class.__dict__:
            # Do not directly get method docs from parent class.
            continue
        default_opts_clsmethod = getattr(mro_class, default_option_method)
        parsed_lines, added_args = _flatten_option_docs(
            docstring=default_opts_clsmethod.__doc__,
            section_repr=section_repr,
            target_args=target_args,
        )
        for line in parsed_lines:
            descriptions.append(indent + line)
        for added_arg in added_args:
            desc_sources[added_arg] = ".".join([mro_class.__module__, mro_class.__name__])
            target_args.remove(added_arg)
        if not target_args:
            break
    else:
        raise Exception(
            f"Option documentation for {', '.join(target_args)} is missing or incomplete "
            f"for the class {current_class.__name__}. "
            "Use Google style docstring. PEP484 type annotations is not supported for options."
        )

    # Use GoogleDocstring to parse and format the docstring
    _config = copy.copy(config)
    _config.napoleon_use_param = True
    docstring = GoogleDocstring(
        docstring=descriptions,
        config=_config,
        app=app,
        obj=current_class,
        options=options,
    )
    
    # Post-process to inject custom roles
    return _inject_roles(
        lines=docstring.lines(),
        default_opt_values=default_options,
        desc_sources=desc_sources,
    )


def _flatten_option_docs(
    docstring: str,
    section_repr: str,
    target_args: set[str] | None = None,
) -> tuple[list[str], set[str]]:
    """A helper function to convert multi-line description into single line."""
    if not docstring:
        return [], set()

    docstring_lines = docstring.splitlines()

    line_ind = 0
    while line_ind < len(docstring_lines):
        if section_repr in docstring_lines[line_ind]:
            line_ind += 1
            break
        line_ind += 1
    else:
        return [], set()

    indent = len(docstring_lines[line_ind]) - len(docstring_lines[line_ind].lstrip())
    parsed_lines = []
    added_args = set()
    idx = line_ind
    while idx < len(docstring_lines):
        arg_entry = docstring_lines[idx]
        idx = idx + 1
        while idx < len(docstring_lines):
            # Remove linefeed and turn multi-line description into single-line
            if docstring_lines[idx][indent:].startswith(" "):
                arg_entry += " " + docstring_lines[idx].strip()
                idx += 1
            else:
                break
        if not arg_entry.strip():
            # Skip blank lines
            continue
        matched = _parameter_doc_regex.match(arg_entry)
        if not matched:
            raise ValueError(
                f"Option documentation '{arg_entry}' doesn't conform to the "
                "expected documentation style. "
                "Use '<name> (<type>): <description>' format."
            )
        opt_name = matched.group(1).strip()
        if target_args and opt_name in target_args:
            parsed_lines.append(arg_entry.lstrip())
            added_args.add(opt_name)

    return parsed_lines, added_args


def _value_repr(value: Any) -> str:
    """Get option value representation."""
    max_elems = 5

    if isinstance(value, str):
        return f'``"{value}"``'
    if isinstance(value, list):
        if len(value) > max_elems:
            elm_repr = ", ".join(map(_value_repr, value[:max_elems])) + ", ..."
        else:
            elm_repr = ", ".join(map(_value_repr, value))
        return f"[{elm_repr}]"
    if isinstance(value, tuple):
        if len(value) > max_elems:
            elm_repr = ", ".join(map(_value_repr, value[:max_elems])) + ", ..."
        else:
            elm_repr = ", ".join(map(_value_repr, value))
        return f"({elm_repr})"
    if isinstance(value, dict):
        keys_repr = map(_value_repr, value.keys())
        vals_repr = map(_value_repr, value.items())
        dict_repr = ", ".join([f"{kr}: {vr}" for kr, vr in zip(keys_repr, vals_repr)])
        return f"{{{dict_repr}}}"
    if value.__class__.__module__ == "builtins":
        return f":obj:`{value}`"
    if value.__class__.__module__ and value.__class__.__module__.startswith("qiskit"):
        return f"Instance of :class:`.{value.__class__.__name__}`"
    # for singleton gates that don't have directly accessible module names
    if hasattr(value, "base_class") and value.base_class.__module__.startswith("qiskit"):
        return f"Instance of :class:`.{value.base_class.__name__}`"
    if callable(value):
        return f"Callable :func:`{value.__name__}`"
    if isinstance(value, np.ndarray):
        if len(value) > max_elems:
            num_repr = ", ".join(map(str, value[:max_elems])) + ", ..."
        else:
            num_repr = ", ".join(map(str, value))
        return f"``array({num_repr}, size={len(value)})``"

    repr_generic = repr(value).replace("\n", "")
    return f"``{repr_generic}``"
