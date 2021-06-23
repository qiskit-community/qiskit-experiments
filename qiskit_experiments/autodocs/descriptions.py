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
Description of options.
"""

import dataclasses
import typing
from qiskit.providers import Options


@dataclasses.dataclass
class OptionsField:
    """A data container to describe a single entry in options."""

    # Default value
    default: typing.Any

    # Type annotation
    annotation: typing.Type

    # Docstring description of the entry
    description: str

    # Set True if this is not a default option
    is_extra: bool = False


@dataclasses.dataclass
class Reference:
    """A data class to describe reference."""

    # Article title
    title: typing.Optional[str] = None

    # Author
    authors: typing.Optional[str] = None

    # Journal info
    journal_info: typing.Optional[str] = None

    # Open Access
    open_access_link: typing.Optional[str] = None


@dataclasses.dataclass
class CurveFitParameter:
    """A data class to describe fit parameter."""

    # Name of fit parameter
    name: str

    # Description about the parameter
    description: str

    # How initial guess is calculated
    initial_guess: str

    # How bounds are calculated
    bounds: str


def _parse_annotation(_type: typing.Any) -> str:
    """A helper function to convert type object into string."""

    if isinstance(_type, str):
        # forward reference
        return _type

    module = _type.__module__

    if module == "builtins":
        return _type.__name__
    elif module == "typing":
        # type representation
        name = getattr(_type, "_name", None)
        if name is None:
            # _GenericAlias and special=False
            type_repr = repr(_type).replace("typing.", "")
            if type_repr in typing.__all__:
                name = type_repr
            else:
                name = _parse_annotation(_type.__origin__)
        # arguments
        if hasattr(_type, "__args__") and _type.__args__:
            args = [_parse_annotation(arg) for arg in _type.__args__]
            return f"{name}[{', '.join(args)}]"
        else:
            return name
    else:
        return f":py:class:`~{module}.{_type.__name__}`"


def to_options(fields: typing.Union[Options, typing.Dict[str, OptionsField]]) -> Options:
    """Converts a dictionary of ``OptionsField`` into ``Options`` object.

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
