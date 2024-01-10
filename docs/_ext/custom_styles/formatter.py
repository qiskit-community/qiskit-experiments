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
A class that formats documentation sections.
"""
from typing import List
from .utils import _check_no_indent, _write_options


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
        format_lines = [
            "",
            ".. rubric:: Overview",
            "",
        ]
        format_lines.extend(lines)
        format_lines.append("")

        return format_lines

    @_check_no_indent
    def format_reference(self, lines: List[str]) -> List[str]:
        """Format reference section."""
        format_lines = [
            ".. rubric:: References",
            "",
        ]
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
        format_lines = [
            ".. rubric:: Example",
            "",
        ]
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
    def format_see_also(self, lines: List[str]) -> List[str]:
        """Format see also section."""
        format_lines = [
            ".. rubric:: See also",
            "",
        ]

        format_lines.extend(lines)
        format_lines.append("")

        return format_lines

    @_check_no_indent
    def format_manual(self, lines: List[str]) -> List[str]:
        """Format user manual section."""
        format_lines = [
            ".. rubric:: User manual",
            "",
        ]
        format_lines.extend(lines)
        format_lines.append("")

        return format_lines

    @_check_no_indent
    def format_init(self, lines: List[str]) -> List[str]:
        """Format user manual section."""
        format_lines = [
            ".. rubric:: Initialization",
            "",
        ]
        format_lines.extend(lines)
        format_lines.append("")

        return format_lines


class ExperimentSectionFormatter(DocstringSectionFormatter):
    """Formatter for experiment class."""

    @_check_no_indent
    def format_analysis_ref(self, lines: List[str]) -> List[str]:
        """Format analysis class reference section."""
        format_lines = [
            ".. rubric:: Analysis class reference",
            "",
        ]
        format_lines.extend(lines)
        format_lines.append("")

        return format_lines

    @_check_no_indent
    def format_experiment_opts(self, lines: List[str]) -> List[str]:
        """Format experiment options section."""
        format_lines = [
            ".. rubric:: Experiment options",
            "",
            "These options can be set by the :meth:`set_experiment_options` method.",
            "",
        ]
        for line in _write_options(lines, self.indent):
            format_lines.append(line)
        format_lines.append("")

        return format_lines


class AnalysisSectionFormatter(DocstringSectionFormatter):
    """Formatter for analysis class."""

    @_check_no_indent
    def format_analysis_opts(self, lines: List[str]) -> List[str]:
        """Format analysis options section."""
        format_lines = [
            ".. rubric:: Analysis options",
            "",
            "These are the keyword arguments of the :meth:`run` method.",
            "",
        ]
        for line in _write_options(lines, self.indent):
            format_lines.append(line)
        format_lines.append("")

        return format_lines

    @_check_no_indent
    def format_fit_model(self, lines: List[str]) -> List[str]:
        """Format fit model section."""
        format_lines = [
            ".. rubric:: Fit model",
            "",
            "This is the curve fitting analysis. ",
            "The following equation(s) are used to represent curve(s).",
            "",
        ]
        format_lines.extend(lines)
        format_lines.append("")

        return format_lines

    @_check_no_indent
    def format_fit_parameters(self, lines: List[str]) -> List[str]:
        """Format fit parameter section."""
        format_lines = [
            ".. rubric:: Fit parameters",
            "",
            "The following fit parameters are estimated during the analysis.",
            "",
        ]
        format_lines.extend(lines)
        format_lines.append("")

        return format_lines


class VisualizationSectionFormatter(DocstringSectionFormatter):
    """Formatter for visualization classes."""

    @_check_no_indent
    def format_opts(self, lines: List[str]) -> List[str]:
        """Format options section."""

        format_lines = [
            ".. rubric:: Options",
            "",
            "The following can be set using :meth:`set_options`.",
            "",
        ]
        for line in _write_options(lines, self.indent):
            format_lines.append(line)
        format_lines.append("")

        return format_lines

    @_check_no_indent
    def format_figure_opts(self, lines: List[str]) -> List[str]:
        """Format figure options section."""
        format_lines = [
            ".. rubric:: Figure options",
            "",
            "The following can be set using :meth:`set_figure_options`.",
            "",
        ]
        for line in _write_options(lines, self.indent):
            format_lines.append(line)
        format_lines.append("")

        return format_lines
