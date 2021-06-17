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
Description of field of experiment options.
"""

import dataclasses
import functools
from typing import Any, Type, Dict, Callable

from qiskit.providers.options import Options


@dataclasses.dataclass
class OptionsField:
    default: Any
    annotation: Type
    description: str


def _compile_docstring(
        header: str,
        fields: Dict[str, OptionsField],
) -> str:
    """Dynamically generates docstring based on information provided by ``OptionsField``s.

    Args:
        header: Header string of the method docstring.
        fields: List of ``OptionsField`` object.

    Returns:
        Method docstring tailored for subclasses.
    """
    _indent = "    "

    options = ""
    for field_name, field in fields.items():
        if isinstance(field.default, Callable):
            default_obj = f":py:func:`{field.default.__name__}`"
        else:
            default_obj = f":py:obj:`{field.default}`"

        options += _indent + f"{field_name}: {field.description} (Default: {default_obj}).\n"

    docstring = f"""{header}

This method is always called with at least one of following keyword arguments.

Args:
{options}

.. note::
    
    You can define arbitrary field with this method.
    If you specify a field name not defined in above list, 
    the name-value pair is passed as ``**kwargs``.
    If the target API does not support the keyword, you may fail in execution.

"""

    return docstring


def _compile_annotations(fields: Dict[str, OptionsField]) -> Dict[str, Any]:
    """Dynamically generate method annotation based on information provided by ``OptionsField``s.

    Args:
        fields: List of ``OptionsField`` object.

    Returns:
        Dictionary of field name and type annotation.
    """
    annotations = dict()

    for field_name, field in fields.items():
        if not isinstance(field.annotation, str):
            annotations[field_name] = field.annotation

    return annotations


def to_options(fields: Dict[str, OptionsField]) -> Options:
    """Converts a list of ``OptionsField`` into ``Options`` object.

    Args:
        fields: List of ``OptionsField`` object to convert.

    Returns:
        ``Options`` that filled with ``.default`` value of ``OptionsField``.
    """
    if isinstance(fields, Options):
        return fields

    default_options = dict()
    for field_name, field in fields.items():
        default_options[field_name] = field.default

    return Options(**default_options)


def create_options_docs(experiment):
    """A class decorator that overrides the docstring and annotation of option setters."""

    # experiment.set_analysis_options directly calls base class method.
    # Thus we cannot directly override __doc__ attribute.

    @functools.wraps(experiment.set_analysis_options)
    def set_analysis_options(self, **fields):
        self._analysis_options.update_options(**fields)

    set_analysis_options.__doc__ = _compile_docstring(
        header="Set the analysis options for :meth:`run` method.",
        fields=experiment.__analysis_class__._default_options(),
    )
    set_analysis_options.__annotations__ = _compile_annotations(
        fields=experiment.__analysis_class__._default_options()
    )

    setattr(experiment, set_analysis_options.__name__, set_analysis_options)

    return experiment
