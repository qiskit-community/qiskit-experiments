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
Documentation for set option methods.
"""
import functools
from types import FunctionType
from typing import Optional, Dict, Any, Type

from qiskit.exceptions import QiskitError

from .descriptions import OptionsField
from .writer import _DocstringWriter, _DocstringMaker


class StandardSetOptionsDocstring(_DocstringMaker):
    """A facade class to write standard set options docstring."""

    @classmethod
    def make_docstring(
        cls,
        description: str,
        options: Dict[str, OptionsField],
        note: Optional[str] = None,
        raises: Optional[Dict[str, str]] = None,
    ) -> str:
        try:
            writer = _DocstringWriter()
            writer.write_lines(description)
            writer.write_options_as_args(options)
            if note:
                writer.write_note(note)
            if raises:
                writer.write_raises(*list(zip(*raises.items())))
        except Exception as ex:
            raise QiskitError(f"Auto docstring generation failed with the error: {ex}")
        return writer.docstring


def _copy_method(experiment: "BaseExperiment", method_name: str) -> FunctionType:
    """A helper function to duplicate base class method.

    Note that calling set options method will access to the base class method.
    If we override attribute, the change will propagate through the all subclass attributes.
    This function prevent this by copying the base class method.

    Args:
        experiment: Base class to get a method.
        method_name: Name of method to copy.

    Returns:
        Duplicated function object.
    """
    base_method = getattr(experiment, method_name)

    new_method = FunctionType(
        code=base_method.__code__,
        globals=base_method.__globals__,
        name=base_method.__name__,
        argdefs=base_method.__defaults__,
        closure=base_method.__closure__,
    )
    return functools.update_wrapper(wrapper=new_method, wrapped=base_method)


def _compile_annotations(fields: Dict[str, OptionsField]) -> Dict[str, Any]:
    """Dynamically generate method annotation based on information provided by ``OptionsField``s.

    Args:
        fields: A dictionary of ``OptionsField`` object.

    Returns:
        Dictionary of field name and type annotation.
    """
    annotations = dict()

    for field_name, field in fields.items():
        if not isinstance(field.annotation, str):
            annotations[field_name] = field.annotation

    return annotations


def base_options_method_documentation(style: Type[_DocstringMaker]):
    """A class decorator that overrides set options method docstring."""

    def decorator(experiment: "BaseExperiment"):
        analysis_options = experiment.__analysis_class__._default_options()
        experiment_options = experiment._default_experiment_options()

        # update analysis options setter
        analysis_setter = _copy_method(experiment, "set_analysis_options")
        analysis_setter.__annotations__ = _compile_annotations(analysis_options)
        analysis_setter.__doc__ = style.make_docstring(
            description="Set the analysis options for :py:meth:`run_analysis` method.",
            options=analysis_options,
            note="Here you can set arbitrary parameter, even if it is not listed. "
            "Such option is passed as a keyword argument to the analysis fitter functions "
            "(if exist). The execution may fail if the function API doesn't support "
            "extra keyword arguments.",
        )
        setattr(experiment, "set_analysis_options", analysis_setter)

        # update experiment options setter
        experiment_setter = _copy_method(experiment, "set_experiment_options")
        experiment_setter.__annotations__ = _compile_annotations(experiment_options)
        experiment_setter.__doc__ = style.make_docstring(
            description="Set the analysis options for :py:meth:`run` method.",
            options=experiment_options,
            raises={"AttributeError": "If the field passed in is not a supported options."},
        )
        setattr(experiment, "set_experiment_options", experiment_setter)

        return experiment

    return decorator
