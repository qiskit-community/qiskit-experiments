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
Docstring writer. This module takes facade pattern to implement the functionality.
"""
import typing
from types import FunctionType

from .descriptions import OptionsField, Reference, CurveFitParameter, _parse_annotation
from abc import abstractclassmethod
from qiskit.exceptions import QiskitError


class _DocstringMaker:
    """A base facade class to write docstring."""

    @abstractclassmethod
    def make_docstring(cls, *args, **kwargs) -> str:
        """Write a docstring."""
        pass


class _DocstringWriter:
    """A docstring writer."""
    __indent__ = "    "

    def __init__(self):
        self.docstring = ""

    def write_header(self, header: str):
        """Write header."""
        self.docstring += f"{header}\n\n"

    def write_options_as_args(self, fields: typing.Dict[str, OptionsField]):
        """Write option descriptions as an argument section.

        This can write multi line description for each option.
        This style is used for detailed summary for the set method docstring.
        Extra fields (non-default options) are also shown.
        """
        def _write_field(_arg_name, _field):

            # parse type
            arg_str_type = f":py:obj:`{_parse_annotation(_field.annotation)}`"

            # write multi line description
            arg_description = self._write_multi_line(_field.description, self.__indent__ * 2)

            # write default value
            default = _field.default
            if default is not None:
                # format representation
                if isinstance(_field.default, FunctionType):
                    default_str = f":py:func:`~{default.__module__}.{default.__name__}`"
                else:
                    default_str = f":py:obj:`{default}`"
                arg_description += self.__indent__ * 2
                arg_description += f"(Default: {default_str})"
            self.docstring += self.__indent__
            self.docstring += f"{_arg_name} ({arg_str_type}):\n{arg_description}\n"

        self.docstring += "Parameters:\n"
        extra_fields = dict()
        for arg_name, field in fields.items():
            if field.is_extra:
                extra_fields[arg_name] = field
                continue
            _write_field(arg_name, field)
        self.docstring += "\n"

        if extra_fields:
            self.docstring += "Other Parameters:\n"
            for arg_name, field in extra_fields.items():
                _write_field(arg_name, field)
            self.docstring += "\n"

    def write_options_as_sections(self, fields: typing.Dict[str, OptionsField], section: str):
        """Write option descriptions as a custom section.

        This writes only the first line of description, if multiple lines exist.
        This style is mainly used for the short summary (options are itemized).
        """
        self.docstring += f"{section}\n"

        for arg_name, field in fields.items():
            if field.is_extra:
                continue
            arg_str_type = f":py:obj:`{_parse_annotation(field.annotation)}`"
            arg_description = field.description.split('\n')[0]
            # write multi line description
            self.docstring += self.__indent__
            self.docstring += f"- **{arg_name}** ({arg_str_type}): {arg_description}\n"
        self.docstring += "\n"

    def write_lines(self, text_block: str):
        """Write text without section."""
        self.docstring += self._write_multi_line(text_block)
        self.docstring += "\n"

    def write_example(self, text_block: str):
        """Write error descriptions."""
        self.docstring += "Example:\n"
        self.docstring += self._write_multi_line(text_block, self.__indent__)
        self.docstring += "\n"

    def write_raises(self, error_kinds: typing.List[str], descriptions: typing.List[str]):
        """Write error descriptions."""
        self.docstring += "Raises:\n"
        for error_kind, description in zip(error_kinds, descriptions):
            self.docstring += self.__indent__
            self.docstring += f"{error_kind}: {description}\n"
        self.docstring += "\n"

    def write_section(self, text_block: str, section: str):
        """Write new user defined section."""
        self.docstring += f"{section}\n"
        self.docstring += self._write_multi_line(text_block, self.__indent__)
        self.docstring += "\n\n"

    def write_note(self, text_block: str):
        """Write note."""
        self.docstring += "Note:\n"
        self.docstring += self._write_multi_line(text_block, self.__indent__)
        self.docstring += "\n"

    def write_warning(self, text_block: str):
        """Write warning."""
        self.docstring += "Warning:\n"
        self.docstring += self._write_multi_line(text_block, self.__indent__)
        self.docstring += "\n"

    def write_references(self, refs: typing.List[Reference]):
        """Write references."""
        self.docstring += "References:\n"
        for idx, ref in enumerate(refs):
            ref_repr = []
            if ref.authors:
                ref_repr.append(f"{ref.authors}")
            if ref.title:
                ref_repr.append(f"`{ref.title}`")
            if ref.journal_info:
                ref_repr.append(f"{ref.journal_info}")
            if ref.open_access_link:
                ref_repr.append(f"`open access <{ref.open_access_link}>`_")
            self.docstring += self.__indent__
            self.docstring += f"- [{idx + 1}] {', '.join(ref_repr)}\n"
        self.docstring += "\n"

    def write_tutorial_link(self, link: str):
        self.docstring += "See Also:\n"
        self.docstring += self.__indent__
        self.docstring += f"- `Qiskit Experiment Tutorial <{link}>`_\n"
        self.docstring += "\n"

    @staticmethod
    def _write_multi_line(text_block: str, indent: typing.Optional[str] = None) -> str:
        """A util method to write multi line text with indentation."""
        indented_text = ""
        for line in text_block.split("\n"):
            if indent is not None:
                indented_text += indent
            indented_text += f"{line}\n"
        return indented_text


class _CurveFitDocstringWriter(_DocstringWriter):

    def write_fit_parameter(self, fit_params: typing.List[CurveFitParameter]):
        """Write fit parameters."""
        self.docstring += "Fit Parameters\n"

        for fit_param in fit_params:
            self.docstring += self.__indent__
            self.docstring += f":math:`{fit_param.name}`: {fit_param.description}\n"
        self.docstring += "\n"

    def write_initial_guess(self, fit_params: typing.List[CurveFitParameter]):
        """Write initial guess estimation method."""
        self.docstring += "Initial Guess\n"

        for fit_param in fit_params:
            self.docstring += self.__indent__
            self.docstring += f":math`{fit_param.name}`: {fit_param.initial_guess}\n"
        self.docstring += "\n"

    def write_bounds(self, fit_params: typing.List[CurveFitParameter]):
        """Write fit parameter bound."""
        self.docstring += "Parameter Boundaries\n"

        for fit_param in fit_params:
            self.docstring += self.__indent__
            self.docstring += f":math`{fit_param.name}`: {fit_param.bounds}\n"
        self.docstring += "\n"

    def write_fit_models(self, equations: typing.List[str]):
        """Write fitting models."""
        self.docstring += "Fit Model\n\n"
        self.docstring += ".. math::\n\n"

        if len(equations) > 1:
            for equation in equations:
                self.docstring += self.__indent__ * 2
                try:
                    lh, rh = equation.split("=")
                except ValueError:
                    raise QiskitError(f"Equation {equation} is not a valid form.")
                self.docstring += f"{lh} &= {rh}\n"
        else:
            self.docstring += self.__indent__ * 2
            self.docstring += f"{equations[0]}\n"
        self.docstring += "\n"
