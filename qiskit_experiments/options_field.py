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
from types import FunctionType
from typing import Any, Type, Dict, Optional, List

from qiskit.exceptions import QiskitError
from qiskit.providers.options import Options


@dataclasses.dataclass
class OptionsField:
    """A data container to describe a single entry in options."""

    # Default value
    default: Any

    # Type annotation
    annotation: Type

    # Docstring description of the entry
    description: str

    # Set True if this is not a default option
    extra_option: bool = False


class _OptionMethodDocstringMaker:
    """Facade of method docstring writer."""
    @classmethod
    def make_docstring(
            cls,
            header: str,
            fields: Dict[str, OptionsField],
            notes: Optional[str] = None,
            raises: Optional[Dict[str, str]] = None
    ) -> str:
        """Create method docstring.

        This mutably update method docstring.

        Args:
            header: Short string for method docstring header to write Args section.
            fields: Dictionary of argument name and ``OptionsField``.
            notes: Additional text block for notation.
            raises: Dictionary of error name and descriptions to write Raises section.

        Returns:
            Automatically generated docstring.
        """
        try:
            writer = _OptionMethodDocstringWriter()
            writer.header(header)
            arg_elems = [
                (key, field.annotation, field.description, field.default)
                for key, field in fields.items()
            ]
            writer.args(*list(zip(*arg_elems)))
            if raises:
                writer.raises(*list(zip(*raises.items())))
            if notes:
                writer.note(notes)
        except Exception as ex:
            raise QiskitError(
                f"Auto docstring failed due to following error: {ex}"
            ) from ex
        return writer.docstring


class _OptionMethodDocstringWriter:
    """Actual docstring writer."""
    __indent__ = "    "

    def __init__(self):
        self.docstring = ""

    def header(self, header: str):
        """Output header."""
        self.docstring += f"{header}\n\n"

    def args(
            self,
            argnames: List[str],
            annotations: List[Any],
            descriptions: List[str],
            defaults: List[str]
    ):
        """Output argument section."""
        self.docstring += "Args:\n"
        for argname, annotation, description, default in \
                zip(argnames, annotations, descriptions, defaults):
            self.docstring += self.__indent__

            # write argument name
            self.docstring += f"{argname} (:py:obj:`{self._parse_type(annotation)}`): "

            # write multi line description of argument
            for line in description.split("\n"):
                self.docstring += f"{line}\n"
                self.docstring += self.__indent__ * 2

            # write default value
            if isinstance(default, FunctionType):
                default_obj = f":py:func:`~{default.__module__}.{default.__name__}`"
            else:
                default_obj = f":py:obj:`{default}`"
            self.docstring += f"(Default: {default_obj})\n"
        self.docstring += "\n"

    def note(self, note: str):
        """Output note section."""
        self.docstring += ".. note::\n\n"
        for line in note.split("\n"):
            self.docstring += self.__indent__
            self.docstring += f"{line}\n"
        self.docstring += "\n"

    def raises(self, error_kinds: List[str], descriptions: List[str]):
        """Output raises section."""
        self.docstring += "Raises:\n"
        for error_kind, description in zip(error_kinds, descriptions):
            self.docstring += self.__indent__
            self.docstring += f"{error_kind}: {description}\n"
        self.docstring += "\n"

    def _parse_type(self, type_obj: Any):
        """Convert type alias to string."""
        if isinstance(type_obj, str):
            # forward reference
            return type_obj

        module = type_obj.__module__

        if module == "builtins":
            return type_obj.__name__
        elif module == "typing":
            # type name
            if hasattr(type_obj, "_name") and type_obj._name:
                # _SpecialForm or special=True
                name = type_obj._name
            else:
                # _GenericAlias and special=False
                if type_obj.__origin__:
                    name = self._parse_type(type_obj.__origin__)
                else:
                    name = type_obj.__name__
            # arguments
            if hasattr(type_obj, "__args__") and type_obj.__args__:
                args = [self._parse_type(arg) for arg in type_obj.__args__]
                return f"{name}[{', '.join(args)}]"
            else:
                return name
        else:
            return f":py:class`{module}.{type_obj.__name__}`"


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


def _copy_method(experiment, method_name: str) -> FunctionType:
    """A helper function to duplicate base calss method.

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
    fields = experiment.__analysis_class__._default_options()

    method = _copy_method(experiment, "set_analysis_options")
    method.__annotations__ = _compile_annotations(fields=fields)
    method.__doc__ = _OptionMethodDocstringMaker.make_docstring(
        header=f"Set the analysis options for :meth:`run_analysis` method.",
        fields=fields,
        notes="""You can define arbitrary field with this method.
If you specify a field name not defined in above list, 
the name-value pair is passed as ``**kwargs``.
If your ``curve_fitter`` API does not support the keyword, you may fail in analysis.
""",
    )

    setattr(experiment, "set_analysis_options", method)

    # experiment.set_experiment_options directly calls base class method.
    # Thus we cannot directly override __doc__ attribute.







    return experiment
