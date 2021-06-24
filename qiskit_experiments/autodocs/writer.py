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
import os
import typing
from abc import ABC, abstractmethod
from types import FunctionType

from qiskit.exceptions import QiskitError

from .descriptions import OptionsField, Reference, CurveFitParameter, _parse_annotation


class _DocstringMaker(ABC):
    """A base facade class to write docstring."""

    @classmethod
    @abstractmethod
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
        self._write_line(header)
        self.docstring += os.linesep

    def write_options_as_args(self, fields: typing.Dict[str, OptionsField]):
        """Write option descriptions as an argument section.

        This can write multi line description for each option.
        This style is used for detailed summary for the set method docstring.
        Extra fields (non-default options) are also shown.
        """

        def _write_field(_arg_name, _field):

            arg_str_type = f":py:obj:`{_parse_annotation(_field.annotation)}`"
            self._write_line(f"{_arg_name} ({arg_str_type}):")
            self._write_multi_line(_field.description, self.__indent__ * 2)
            default = _field.default
            if default is not None:
                # format representation
                if isinstance(_field.default, FunctionType):
                    default_str = f":py:func:`~{default.__module__}.{default.__name__}`"
                else:
                    default_str = f":py:obj:`{default}`"
                self.docstring += self.__indent__ * 2
                self._write_line(f"(Default: {default_str})")

        self._write_line("Parameters:")
        extra_fields = dict()
        for arg_name, field in fields.items():
            if field.is_extra:
                extra_fields[arg_name] = field
                continue
            _write_field(arg_name, field)
        self.docstring += os.linesep

        if extra_fields:
            self._write_line("Other Parameters:")
            for arg_name, field in extra_fields.items():
                _write_field(arg_name, field)
            self.docstring += os.linesep

    def write_options_as_sections(
        self,
        fields: typing.Dict[str, OptionsField],
        section: str,
        text_block: typing.Optional[str] = None,
    ):
        """Write option descriptions as a custom section.

        This writes only the first line of description, if multiple lines exist.
        This style is mainly used for the short summary (options are itemized).
        This section will be shown as a drop down box.
        """
        self._write_line(f".. dropdown:: {section}")
        self.docstring += self.__indent__
        self._write_line(":animate: fade-in-slide-down")
        self.docstring += os.linesep

        if text_block:
            self._write_multi_line(text_block, self.__indent__)

        for arg_name, field in fields.items():
            if field.is_extra:
                continue
            arg_str_type = f":py:obj:`{_parse_annotation(field.annotation)}`"
            arg_description = field.description.split(os.linesep)[0]
            # write multi line description
            self.docstring += self.__indent__
            self._write_line(f"- **{arg_name}** ({arg_str_type}): {arg_description}")
        self.docstring += os.linesep

    def write_lines(self, text_block: str):
        """Write text without section."""
        self._write_multi_line(text_block)
        self.docstring += os.linesep

    def write_example(self, text_block: str):
        """Write error descriptions."""
        self._write_line("Example:")
        self._write_multi_line(text_block, self.__indent__)
        self.docstring += os.linesep

    def write_raises(self, error_kinds: typing.List[str], descriptions: typing.List[str]):
        """Write error descriptions."""
        self._write_line("Raises:")
        for error_kind, description in zip(error_kinds, descriptions):
            self.docstring += self.__indent__
            self._write_line(f"{error_kind}: {description}")
        self.docstring += os.linesep

    def write_section(self, text_block: str, section: str):
        """Write new user defined section."""
        self._write_line(f"{section}")
        self._write_multi_line(text_block, self.__indent__)
        self.docstring += os.linesep

    def write_note(self, text_block: str):
        """Write note."""
        self._write_line("Note:")
        self._write_multi_line(text_block, self.__indent__)
        self.docstring += os.linesep

    def write_warning(self, text_block: str):
        """Write warning."""
        self._write_line("Warning:")
        self._write_multi_line(text_block, self.__indent__)
        self.docstring += os.linesep

    def write_returns(self, text_block: str):
        """Write returns."""
        self._write_line("Returns:")
        self._write_multi_line(text_block, self.__indent__)
        self.docstring += os.linesep

    def write_references(self, refs: typing.List[Reference]):
        """Write references."""
        self._write_line("References:")
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
            self._write_line(f"- [{idx + 1}] {', '.join(ref_repr)}")
        self.docstring += os.linesep

    def write_tutorial_link(self, link: str):
        """Write link to tutorial website."""
        self._write_line("See Also:")
        self.docstring += self.__indent__
        self._write_line(f"- `Qiskit Experiment Tutorial <{link}>`_")
        self.docstring += os.linesep

    def _write_multi_line(self, text_block: str, indent: typing.Optional[str] = None):
        """A util method to write multi line text with indentation."""
        indented_text = ""
        for line in text_block.split(os.linesep):
            if len(line) > 0 and indent is not None:
                indented_text += indent
            self._write_line(line.rstrip())

    def _write_line(self, text: str):
        """A helper function to write single line."""
        self.docstring += text.rstrip() + os.linesep


class _CurveFitDocstringWriter(_DocstringWriter):
    """A docstring writer supporting fit model descriptions."""

    def write_fit_parameter(self, fit_params: typing.List[CurveFitParameter]):
        """Write fit parameters."""
        self._write_line("Fit Parameters")
        for fit_param in fit_params:
            self.docstring += self.__indent__
            self._write_line(f"- :math:`{fit_param.name}`: {fit_param.description}")
        self.docstring += os.linesep

    def write_initial_guess(self, fit_params: typing.List[CurveFitParameter]):
        """Write initial guess estimation method."""
        self._write_line("Initial Guess")
        for fit_param in fit_params:
            self.docstring += self.__indent__
            self._write_line(f"- :math:`{fit_param.name}`: {fit_param.initial_guess}")
        self.docstring += os.linesep

    def write_bounds(self, fit_params: typing.List[CurveFitParameter]):
        """Write fit parameter bound."""
        self._write_line("Parameter Boundaries")

        for fit_param in fit_params:
            self.docstring += self.__indent__
            self._write_line(f"- :math:`{fit_param.name}`: {fit_param.bounds}")
        self.docstring += os.linesep

    def write_fit_models(self, equations: typing.List[str]):
        """Write fitting models."""
        self._write_line("Fit Model")
        self.docstring += self.__indent__
        self._write_line(".. math::")
        self.docstring += os.linesep

        if len(equations) > 1:
            eqs = []
            for equation in equations:
                try:
                    lh, rh = equation.split("=")
                except ValueError as ex:
                    raise QiskitError(f"Equation {equation} is not a valid form.") from ex
                eqs.append(f"{self.__indent__ * 2}{lh} &= {rh}")
            self.docstring += f" \\\\{os.linesep}".join(eqs)
            self.docstring += os.linesep
        else:
            self.docstring += self.__indent__ * 2
            self._write_line(equations[0])
        self.docstring += os.linesep
