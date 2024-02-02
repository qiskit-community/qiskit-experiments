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
from typing import Any, Set, List, Tuple
from typing import Type, Optional

import numpy as np
from sphinx.application import Sphinx
from sphinx.ext.autodoc import Options as SphinxOptions
from sphinx.ext.napoleon import Config as NapoleonConfig
from sphinx.ext.napoleon import GoogleDocstring


_parameter_doc_regex = re.compile(r"(.+?)\(\s*(.*[^\s]+)\s*\):(.*[^\s]+)")


class QiskitExperimentsOptionsDocstring(GoogleDocstring):
    """GoogleDocstring with updated parameter field formatter.

    This docstring parser may take options mapping in the Sphinx option
    and inject :default_val: role to the restructured text to manage default value.

    Since this class overrides a protected member, it might be sensitive to napoleon version.
    """

    def _format_docutils_params(
        self,
        fields: List[Tuple[str, str, List[str]]],
        field_role: str = "param",
        type_role: str = "type",
        default_role: str = "default_val",
        source_role: str = "source",
    ) -> List[str]:
        lines = []
        for _name, _type, _desc in fields:
            _desc = self._strip_empty(_desc)
            if any(_desc):
                _desc = self._fix_field_desc(_desc)[0]
                lines.append(f":{field_role} {_name}: {_desc}")
            else:
                lines.append(f":{field_role} {_name}: ")
            if _type:
                lines.append(f":{type_role} {_name}: {_type}")
            if "default_opt_values" in self._opt:
                value = _value_repr(self._opt["default_opt_values"].get(_name, None))
                lines.append(f":{default_role} {_name}: {value}")
            if "desc_sources" in self._opt and _name in self._opt["desc_sources"]:
                source = self._opt["desc_sources"][_name]
                lines.append(f":{source_role} {_name}: :class:`~.{source}`")
            else:
                lines.append(f":{source_role} {_name}: Unknown class")
        return lines + [""]


def process_default_options(
    current_class: Type,
    default_option_method: str,
    section_repr: str,
    app: Sphinx,
    options: SphinxOptions,
    config: NapoleonConfig,
    indent: Optional[str] = "",
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

    extra_info = {
        "default_opt_values": default_options,
        "desc_sources": desc_sources,
    }

    # Relying on GoogleDocstring to apply typehint automation
    _options = options.copy()
    _options.update(extra_info)
    _config = copy.copy(config)
    _config.napoleon_use_param = True
    docstring = QiskitExperimentsOptionsDocstring(
        docstring=descriptions,
        config=_config,
        app=app,
        obj=current_class,
        options=_options,
    )
    return docstring.lines()


def _flatten_option_docs(
    docstring: str,
    section_repr: str,
    target_args: Optional[Set[str]] = None,
) -> Tuple[List[str], Set[str]]:
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
    tmp = ""
    parsed_lines = []
    added_args = set()
    for line in docstring_lines[line_ind:]:
        if line[indent:].startswith(" "):
            # Remove linefeed and turn multi-line description into single-line
            tmp += " " + line.lstrip()
        else:
            if tmp:
                matched = _parameter_doc_regex.match(tmp)
                if not matched:
                    raise ValueError(
                        f"Option documentation '{tmp}' doesn't conform to the "
                        "expected documentation style. "
                        "Use '<name> (<type>): <description>' format."
                    )
                opt_name = matched.group(1).strip()
                if target_args and opt_name in target_args:
                    parsed_lines.append(tmp.lstrip())
                    added_args.add(opt_name)
            # Start new line
            tmp = line

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
