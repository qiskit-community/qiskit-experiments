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
A class that formats documentation sections.
"""
from typing import List
from .utils import _check_no_indent


class DocstringSectionFormatter:
    """A class that formats parsed docstring lines.

    This formatter formats sections with Google Style Python Docstrings with
    several reStructuredText directives.
    """

    def __init__(self, indent: str):
        self.indent = indent

    def format_header(self, lines: List[str]) -> List[str]:
        """Format header section."""
        format_lines = lines
        format_lines.append("")

        return format_lines

    @_check_no_indent
    def format_overview(self, lines: List[str]) -> List[str]:
        """Format overview section."""
        format_lines = [".. rubric:: Overview", ""]
        format_lines.extend(lines)
        format_lines.append("")

        return format_lines

    @_check_no_indent
    def format_reference(self, lines: List[str]) -> List[str]:
        """Format reference section."""
        format_lines = [".. rubric:: References", ""]
        format_lines.extend(lines)
        format_lines.append("")

        return format_lines

    def format_warning(self, lines: List[str]) -> List[str]:
        """Format warning section."""
        format_lines = [".. warning::", ""]
        for line in lines:
            format_lines.append(self.indent + line)
        format_lines.append("")

        return format_lines

    @_check_no_indent
    def format_example(self, lines: List[str]) -> List[str]:
        """Format example section."""
        format_lines = [".. rubric:: Example", ""]
        format_lines.extend(lines)
        format_lines.append("")

        return format_lines

    def format_note(self, lines: List[str]) -> List[str]:
        """Format notification section."""
        format_lines = [".. note::", ""]
        for line in lines:
            format_lines.append(self.indent + line)
        format_lines.append("")

        return format_lines

    @_check_no_indent
    def format_tutorial(self, lines: List[str]) -> List[str]:
        """Format tutorial section."""
        format_lines = [".. rubric:: Tutorials", ""]
        format_lines.extend(lines)
        format_lines.append("")

        return format_lines


class ExperimentSectionFormatter(DocstringSectionFormatter):
    """Formatter for experiment class."""

    @_check_no_indent
    def format_analysis_ref(self, lines: List[str]) -> List[str]:
        """Format analysis class reference section."""
        format_lines = [".. rubric:: Analysis Class Reference", ""]
        format_lines.extend(lines)
        format_lines.append("")

        return format_lines

    @_check_no_indent
    def format_experiment_opts(self, lines: List[str]) -> List[str]:
        """Format experiment options section."""
        format_lines = [
            ".. rubric:: Experiment Options",
            "",
            "These options can be set by :py:meth:`set_experiment_options` method.",
            "",
        ]
        format_lines.extend(lines)
        format_lines.append("")

        return format_lines

    @_check_no_indent
    def format_analysis_opts(self, lines: List[str]) -> List[str]:
        """Format analysis options section."""
        format_lines = [
            ".. rubric:: Analysis Options",
            "",
            "These options can be set by :py:meth:`set_analysis_options` method.",
            "",
        ]
        format_lines.extend(lines)
        format_lines.append("")

        return format_lines

    @_check_no_indent
    def format_transpiler_opts(self, lines: List[str]) -> List[str]:
        """Format transpiler options section."""
        format_lines = [
            ".. rubric:: Transpiler Options",
            "",
            "This option can be set by :py:meth:`set_transpile_options` method.",
            "",
        ]
        format_lines.extend(lines)
        format_lines.append("")

        return format_lines

    @_check_no_indent
    def format_run_opts(self, lines: List[str]) -> List[str]:
        """Format run options section."""
        format_lines = [
            ".. rubric:: Backend Run Options",
            "",
            "This option can be set by :py:meth:`set_run_options` method.",
            "",
        ]
        format_lines.extend(lines)
        format_lines.append("")

        return format_lines


class AnalysisSectionFormatter(DocstringSectionFormatter):
    """Formatter for analysis class."""

    @_check_no_indent
    def format_analysis_opts(self, lines: List[str]) -> List[str]:
        """Format analysis options section."""
        format_lines = [
            ".. rubric:: Run Options",
            "",
            "These are the keyword arguments of :py:meth:`run` method.",
            "",
        ]
        format_lines.extend(lines)
        format_lines.append("")

        return format_lines

    @_check_no_indent
    def format_fit_model(self, lines: List[str]) -> List[str]:
        """Format fit model section."""
        format_lines = [
            ".. rubric:: Fit Model",
            "",
            "This is the curve fitting analysis. ",
            "Following equation(s) are used to represent curve(s).",
            "",
        ]
        format_lines.extend(lines)
        format_lines.append("")

        return format_lines

    @_check_no_indent
    def format_fit_parameters(self, lines: List[str]) -> List[str]:
        """Format fit parameter section."""
        format_lines = [
            ".. rubric:: Fit Parameters",
            "",
            "Following fit parameters are estimated during the analysis.",
            "",
        ]
        format_lines.extend(lines)
        format_lines.append("")

        return format_lines
