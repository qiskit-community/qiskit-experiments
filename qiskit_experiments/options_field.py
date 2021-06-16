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
from typing import Any, Type, Dict
from qiskit.providers.options import Options


@dataclasses.dataclass
class OptionsField:
    default: Any
    annotation: Type
    description: str


def _compile_docstring(header: str, fields: Dict[str, OptionsField]):

    __indent__ = "    "

    docstring = f"{header}\n\n"
    docstring += "Args:\n"

    for field_name, field in fields.items():
        docstring += __indent__
        docstring += f"{field_name}: {field.description}. Defaults to {repr(field.default)}.\n"

    return docstring


def _compile_annotations(fields: Dict[str, OptionsField]):

    annotations = dict()

    for field_name, field in fields.items():
        annotations[field_name] = field.annotation

    return annotations


def _to_options(fields: Dict[str, OptionsField]) -> Options:

    default_options = dict()
    for field_name, field in fields.items():
        default_options[field_name] = field.default

    return Options(**default_options)
