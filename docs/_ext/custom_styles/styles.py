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
Documentation extension for experiment class.
"""
import copy
import re
import sys
from abc import ABC
from typing import Union, List, Dict

from qiskit_experiments.framework.base_analysis import BaseAnalysis
from qiskit_experiments.framework.base_experiment import BaseExperiment
from qiskit_experiments.visualization import BaseDrawer, BasePlotter
from sphinx.config import Config as SphinxConfig

from .formatter import (
    ExperimentSectionFormatter,
    AnalysisSectionFormatter,
    DocstringSectionFormatter,
    VisualizationSectionFormatter,
)
from .section_parsers import load_standard_section, load_fit_parameters
from .utils import (
    _generate_options_documentation,
    _generate_analysis_ref,
    _format_default_options,
)

section_regex = re.compile(r"# section: (?P<section_name>\S+)")


class QiskitExperimentDocstring(ABC):
    """Qiskit Experiment style docstring parser base class."""

    # mapping of sections supported by this style to parsing method or function
    __sections__ = {}

    # section formatter
    __formatter__ = DocstringSectionFormatter

    def __init__(
        self,
        target_cls: object,
        docstring_lines: Union[str, List[str]],
        config: SphinxConfig,
        indent: str = "",
    ):
        """Create new parser and parse formatted docstring."""

        if isinstance(docstring_lines, str):
            lines = docstring_lines.splitlines()
        else:
            lines = docstring_lines

        self._target_cls = target_cls
        self._indent = indent
        self._config = config

        self._parsed_lines = self._classify(lines)

    def _classify(self, docstrings: List[str]) -> Dict[str, List[str]]:
        """Classify formatted docstring into sections."""
        sectioned_docstrings = dict()

        def add_new_section(section: str, lines: List[str]):
            if lines:
                parser = self.__sections__[section]
                if not parser:
                    raise KeyError(
                        f"Section {section} is automatically generated section. "
                        "This section cannot be overridden by class docstring."
                    )
                sectioned_docstrings[section] = parser(temp_lines)

        current_section = list(self.__sections__.keys())[0]
        temp_lines = list()
        margin = sys.maxsize
        for docstring_line in docstrings:
            match = re.match(section_regex, docstring_line.strip())
            if match:
                section_name = match["section_name"]
                if section_name in self.__sections__:
                    # parse previous section
                    if margin < sys.maxsize:
                        temp_lines = [l[margin:] for l in temp_lines]
                    add_new_section(current_section, temp_lines)
                    # set new section
                    current_section = section_name
                    temp_lines.clear()
                    margin = sys.maxsize
                else:
                    raise KeyError(f"Section name {section_name} is invalid.")
                continue

            # calculate section indent
            if len(docstring_line) > 0 and not docstring_line.isspace():
                # ignore empty line
                indent = len(docstring_line) - len(docstring_line.lstrip())
                margin = min(indent, margin)

            temp_lines.append(docstring_line)

        # parse final section
        if margin < sys.maxsize:
            temp_lines = [l[margin:] for l in temp_lines]
        add_new_section(current_section, temp_lines)

        # add extra section
        self._extra_sections(sectioned_docstrings)

        return sectioned_docstrings

    def _extra_sections(self, sectioned_docstring: Dict[str, List[str]]):
        """Generate extra sections."""
        pass

    def _format(self) -> Dict[str, List[str]]:
        """Format each section with predefined formatter."""
        formatter = self.__formatter__(self._indent)

        formatted_sections = {section: None for section in self.__sections__}
        for section, lines in self._parsed_lines.items():
            if not lines:
                continue
            section_formatter = getattr(formatter, f"format_{section}", None)
            if section_formatter:
                formatted_sections[section] = section_formatter(lines)
            else:
                formatted_sections[section] = lines + [""]

        return formatted_sections

    def generate_class_docs(self) -> List[List[str]]:
        """Output formatted experiment class documentation."""
        formatted_sections = self._format()

        classdoc_lines = []
        for section_lines in formatted_sections.values():
            if section_lines:
                classdoc_lines.extend(section_lines)

        return [classdoc_lines]


class ExperimentDocstring(QiskitExperimentDocstring):
    """Documentation parser for the experiment class introduction."""

    __sections__ = {
        "header": load_standard_section,
        "warning": load_standard_section,
        "overview": load_standard_section,
        "reference": load_standard_section,
        "tutorial": load_standard_section,
        "analysis_ref": load_standard_section,
        "experiment_opts": None,
        "transpiler_opts": None,
        "run_opts": None,
        "example": load_standard_section,
        "note": load_standard_section,
        "see_also": load_standard_section,
    }

    __formatter__ = ExperimentSectionFormatter

    def __init__(
        self,
        target_cls: BaseExperiment,
        docstring_lines: Union[str, List[str]],
        config: SphinxConfig,
        indent: str = "",
    ):
        """Create new parser and parse formatted docstring."""
        super().__init__(target_cls, docstring_lines, config, indent)

    def _extra_sections(self, sectioned_docstring: Dict[str, List[str]]):
        """Generate extra sections."""

        # add experiment option
        exp_option_desc = []

        exp_docs_config = copy.copy(self._config)
        exp_docs_config.napoleon_custom_sections = [("experiment options", "args")]
        exp_option = _generate_options_documentation(
            current_class=self._target_cls,
            method_name="_default_experiment_options",
            config=exp_docs_config,
            indent=self._indent,
        )
        if exp_option:
            exp_option_desc.extend(exp_option)
            exp_option_desc.append("")
            exp_option_desc.extend(
                _format_default_options(
                    defaults=self._target_cls._default_experiment_options().__dict__,
                    indent=self._indent,
                )
            )
        else:
            exp_option_desc.append("No experiment option available for this experiment.")

        sectioned_docstring["experiment_opts"] = exp_option_desc

        # add transpiler option
        transpiler_option_desc = [
            "This option is used for circuit optimization. ",
            "See the documentation of :func:`qiskit.transpile <qiskit.compiler.transpile>` "
            "for available options.",
            "",
        ]
        transpiler_option_desc.extend(
            _format_default_options(
                defaults=self._target_cls._default_transpile_options().__dict__,
                indent=self._indent,
            )
        )

        sectioned_docstring["transpiler_opts"] = transpiler_option_desc

        # add run option
        run_option_desc = [
            "This option is used for controlling job execution condition. "
            "Note that this option is provider dependent. "
            "See provider's backend runner API for available options. "
            "See the documentation of "
            ":meth:`IBMQBackend.run <qiskit.providers.ibmq.IBMQBackend.run>` "
            "for the IBM Quantum Service.",
            "",
        ]
        run_option_desc.extend(
            _format_default_options(
                defaults=self._target_cls._default_run_options().__dict__,
                indent=self._indent,
            )
        )
        sectioned_docstring["run_opts"] = run_option_desc

        # add analysis reference, if nothing described, it copies from parent
        if not sectioned_docstring.get("analysis_ref", None):
            analysis_desc = _generate_analysis_ref(
                current_class=self._target_cls,
                config=exp_docs_config,
                indent=self._indent,
            )

            sectioned_docstring["analysis_ref"] = analysis_desc


class AnalysisDocstring(QiskitExperimentDocstring):
    """Documentation parser for the analysis class introduction."""

    __sections__ = {
        "header": load_standard_section,
        "warning": load_standard_section,
        "overview": load_standard_section,
        "fit_model": load_standard_section,
        "fit_parameters": load_fit_parameters,
        "reference": load_standard_section,
        "tutorial": load_standard_section,
        "analysis_opts": None,
        "example": load_standard_section,
        "note": load_standard_section,
        "see_also": load_standard_section,
    }

    __formatter__ = AnalysisSectionFormatter

    def __init__(
        self,
        target_cls: BaseAnalysis,
        docstring_lines: Union[str, List[str]],
        config: SphinxConfig,
        indent: str = "",
    ):
        """Create new parser and parse formatted docstring."""
        super().__init__(target_cls, docstring_lines, config, indent)

    def _extra_sections(self, sectioned_docstring: Dict[str, List[str]]):
        """Generate extra sections."""

        # add analysis option
        option_desc = []

        analysis_docs_config = copy.copy(self._config)
        analysis_docs_config.napoleon_custom_sections = [("analysis options", "args")]
        analysis_option = _generate_options_documentation(
            current_class=self._target_cls,
            method_name="_default_options",
            config=analysis_docs_config,
            indent=self._indent,
        )
        if analysis_option:
            option_desc.extend(analysis_option)
            option_desc.append("")
            option_desc.extend(
                _format_default_options(
                    defaults=self._target_cls._default_options().__dict__,
                    indent=self._indent,
                )
            )
        else:
            option_desc.append("No option available for this analysis.")

        sectioned_docstring["analysis_opts"] = option_desc


class VisualizationDocstring(QiskitExperimentDocstring):
    """Documentation parser for visualization classes' introductions."""

    __sections__ = {
        "header": load_standard_section,
        "warning": load_standard_section,
        "overview": load_standard_section,
        "reference": load_standard_section,
        "tutorial": load_standard_section,
        "opts": None,           # For standard options
        "figure_opts": None,    # For figure options
        "example": load_standard_section,
        "note": load_standard_section,
        "see_also": load_standard_section,
    }

    __formatter__ = VisualizationSectionFormatter

    def __init__(
        self,
        target_cls: Union[BaseDrawer, BasePlotter],
        docstring_lines: Union[str, List[str]],
        config: SphinxConfig,
        indent: str = "",
    ):
        """Create new parser and parse formatted docstring."""
        super().__init__(target_cls, docstring_lines, config, indent)

    def _extra_sections(self, sectioned_docstring: Dict[str, List[str]]):
        """Generate extra sections."""
        # add options
        option_desc = []
        figure_option_desc = []

        docs_config = copy.copy(self._config)
        docs_config.napoleon_custom_sections = [
            ("options", "args"),
            ("figure options", "args"),
        ]

        # Generate options docs
        option = _generate_options_documentation(
            current_class=self._target_cls,
            method_name="_default_options",
            config=docs_config,
            indent=self._indent,
        )
        if option:
            option_desc.extend(option)
            option_desc.append("")
            option_desc.extend(
                _format_default_options(
                    defaults=self._target_cls._default_options().__dict__,
                    indent=self._indent,
                )
            )
        else:
            option_desc.append("No options available.")

        # Generate figure options docs
        figure_option = _generate_options_documentation(
            current_class=self._target_cls,
            method_name="_default_figure_options",
            config=docs_config,
            indent=self._indent,
        )
        if figure_option:
            figure_option_desc.extend(figure_option)
            figure_option_desc.append("")
            figure_option_desc.extend(
                _format_default_options(
                    defaults=self._target_cls._default_figure_options().__dict__,
                    indent=self._indent,
                )
            )
        else:
            figure_option_desc.append("No figure options available.")

        sectioned_docstring["opts"] = option_desc
        sectioned_docstring["figure_opts"] = figure_option_desc
