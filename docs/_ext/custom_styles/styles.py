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
Documentation extension for experiment class.
"""
import copy
import re
import sys
from abc import ABC
from typing import Union, List, Dict

from qiskit_experiments.framework.base_analysis import BaseAnalysis
from qiskit_experiments.framework.base_experiment import BaseExperiment
from sphinx.config import Config as SphinxConfig

from .formatter import (
    ExperimentSectionFormatter,
    AnalysisSectionFormatter,
    DocstringSectionFormatter,
)
from .section_parsers import load_standard_section, load_fit_parameters
from .utils import _generate_options_documentation, _format_default_options

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

        for section, lines in self._extra_sections().items():
            sectioned_docstrings[section] = lines

        return sectioned_docstrings

    def _extra_sections(self) -> Dict[str, List[str]]:
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
        "analysis_ref": None,
        "experiment_opts": None,
        "analysis_opts": None,
        "transpiler_opts": None,
        "run_opts": None,
        "example": load_standard_section,
        "note": load_standard_section,
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

    def _extra_sections(self) -> Dict[str, List[str]]:
        """Generate extra sections."""
        parsed_sections = {}

        # add analysis class reference
        analysis_class = getattr(self._target_cls, "__analysis_class__", None)
        if analysis_class:
            analysis_ref = f":py:class:`~{analysis_class.__module__}.{analysis_class.__name__}`"
            parsed_sections["analysis_ref"] = [analysis_ref]

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

        parsed_sections["experiment_opts"] = exp_option_desc

        # add analysis option
        analysis_option_desc = []

        if analysis_class:
            analysis_docs_config = copy.copy(self._config)
            analysis_docs_config.napoleon_custom_sections = [("analysis options", "args")]
            analysis_option = _generate_options_documentation(
                current_class=analysis_class,
                method_name="_default_options",
                config=analysis_docs_config,
                indent=self._indent,
            )

            if analysis_option:
                analysis_option_desc.extend(analysis_option)
                analysis_option_desc.append("")
                analysis_option_desc.extend(
                    _format_default_options(
                        defaults=analysis_class._default_options().__dict__,
                        indent=self._indent,
                    )
                )
            else:
                analysis_option_desc.append("No analysis option available for this experiment.")

        parsed_sections["analysis_opts"] = analysis_option_desc

        # add transpiler option
        transpiler_option_desc = [
            "This option is used for circuit optimization. ",
            "See `Qiskit Transpiler <https://qiskit.org/documentation/stubs/",
            "qiskit.compiler.transpile.html>`_ documentation for available options.",
            "",
        ]
        transpiler_option_desc.extend(
            _format_default_options(
                defaults=self._target_cls._default_transpile_options().__dict__,
                indent=self._indent,
            )
        )

        parsed_sections["transpiler_opts"] = transpiler_option_desc

        # add run option
        run_option_desc = [
            "This option is used for controlling job execution condition. "
            "Note that this option is provider dependent. "
            "See provider's backend runner API for available options. "
            "See `here <https://qiskit.org/documentation/stubs/qiskit.providers.ibmq.",
            "IBMQBackend.html#qiskit.providers.ibmq.IBMQBackend.run>`_ for IBM Quantum Service.",
            "",
        ]
        run_option_desc.extend(
            _format_default_options(
                defaults=self._target_cls._default_run_options().__dict__,
                indent=self._indent,
            )
        )

        parsed_sections["run_opts"] = run_option_desc

        return parsed_sections


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

    def _extra_sections(self) -> Dict[str, List[str]]:
        """Generate extra sections."""
        parsed_sections = {}

        # add analysis option
        analysis_docs_config = copy.copy(self._config)
        analysis_docs_config.napoleon_custom_sections = [("analysis options", "args")]
        analysis_option = _generate_options_documentation(
            current_class=self._target_cls,
            method_name="_default_options",
            config=analysis_docs_config,
            indent=self._indent,
        )
        if analysis_option:
            parsed_sections["analysis_opts"] = analysis_option

        return parsed_sections
