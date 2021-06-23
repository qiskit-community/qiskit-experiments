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
Documentation for analysis class.
"""
import re
from typing import Optional, Dict, List, Type

from qiskit.exceptions import QiskitError

from .descriptions import OptionsField, Reference, CurveFitParameter
from .writer import _DocstringWriter, _CurveFitDocstringWriter, _DocstringMaker


class StandardAnalysisDocstring(_DocstringMaker):
    """A facade class to write standard analysis docstring."""

    @classmethod
    def make_docstring(
            cls,
            default_options: Dict[str, OptionsField],
            overview: Optional[str] = None,
            example: Optional[str] = None,
            references: Optional[List[Reference]] = None,
            note: Optional[str] = None,
            warning: Optional[str] = None,
            tutorial: Optional[str] = None,
    ) -> str:
        try:
            writer = _DocstringWriter()
            if warning:
                writer.write_warning(warning)
            if overview:
                writer.write_section(overview, "Overview")
            if example:
                writer.write_example(example)
            writer.write_lines(
                "Default class options. These options are automatically set "
                "when :py:meth:`run` method is called."
            )
            writer.write_options_as_sections(default_options, "Default Options")
            if references:
                writer.write_references(references)
            if note:
                writer.write_note(note)
            if tutorial:
                writer.write_tutorial_link(tutorial)
        except Exception as ex:
            raise QiskitError(f"Auto docstring generation failed with the error: {ex}")
        return writer.docstring


class CurveAnalysisDocstring(_DocstringMaker):
    """A facade class to write curve analysis docstring."""

    @classmethod
    def make_docstring(
            cls,
            default_options: Dict[str, OptionsField],
            overview: Optional[str] = None,
            equations: Optional[List[str]] = None,
            fit_params: Optional[List[CurveFitParameter]] = None,
            example: Optional[str] = None,
            references: Optional[List[Reference]] = None,
            note: Optional[str] = None,
            warning: Optional[str] = None,
            tutorial: Optional[str] = None,
    ) -> str:
        try:
            writer = _CurveFitDocstringWriter()
            if warning:
                writer.write_warning(warning)
            if overview:
                writer.write_section(overview, "Overview")
            if equations:
                if isinstance(equations, str):
                    equations = [equations]
                writer.write_lines("This analysis assumes following fit function(s).")
                writer.write_fit_models(equations)
            if fit_params:
                writer.write_lines(
                    "The fit model takes following fit parameters."
                    "These parameters are fit by the ``curve_fitter`` function specified in "
                    "the analysis options."
                )
                writer.write_fit_parameter(fit_params)
                writer.write_lines(
                    "The parameter initial guess are generated as follows. "
                    "If you want to override, you can provide ``p0`` of the analysis options."
                )
                writer.write_initial_guess(fit_params)
                writer.write_lines(
                    "The parameter boundaries are generated as follows. "
                    "If you want to override, you can provide ``bounds`` of the analysis options."
                )
                writer.write_bounds(fit_params)
            if example:
                writer.write_example(example)
            writer.write_lines(
                "Default class options. These options are automatically set "
                "when :py:meth:`run` method is called."
            )
            writer.write_options_as_sections(default_options, "Default Options")
            if references:
                writer.write_references(references)
            if note:
                writer.write_note(note)
            if tutorial:
                writer.write_tutorial_link(tutorial)
        except Exception as ex:
            raise QiskitError(f"Auto docstring generation failed with the error: {ex}")
        return writer.docstring


def base_analysis_documentation(style: Type[_DocstringMaker]):
    """A class decorator that overrides analysis class docstring."""
    def decorator(analysis: "BaseAnalysis"):
        regex = r"__doc_(?P<kwarg>\S+)__"

        kwargs = {}
        for attribute in dir(analysis):
            match = re.match(regex, attribute)
            if match:
                arg = match["kwarg"]
                kwargs[arg] = getattr(analysis, attribute)

        exp_docs = style.make_docstring(
            default_options=analysis._default_options(),
            **kwargs
        )
        analysis.__doc__ += f"\n\n{exp_docs}"

        return analysis
    return decorator
