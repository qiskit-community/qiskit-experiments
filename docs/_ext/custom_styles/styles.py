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
from sphinx.config import Config as SphinxConfig

from .formatter import (
    ExperimentSectionFormatter,
    AnalysisSectionFormatter,
    DocstringSectionFormatter,
    VisualizationSectionFormatter,
)
from .section_parsers import load_standard_section, load_fit_parameters
from .utils import (
    _generate_analysis_ref,
    _get_superclass,
)


_section_regex = re.compile(r"\s*# section:\s*(?P<section_key>\S+)")


class QiskitExperimentDocstring(ABC):
    """Qiskit Experiment style docstring parser base class."""

    # mapping of sections supported by this style to parsing method or function
    __sections__ = {
        "header": load_standard_section,
    }

    # section formatter
    __formatter__ = DocstringSectionFormatter

    def __init__(
        self,
        target_cls: object,
        docstring_lines: Union[str, List[str]],
        config: SphinxConfig,
        indent: str = "",
        **extra_sections: List[str],
    ):
        """Create new parser and parse formatted docstring."""

        if isinstance(docstring_lines, str):
            lines = docstring_lines.splitlines()
        else:
            lines = docstring_lines

        self._target_cls = target_cls
        self._indent = indent
        self._config = config

        self._parsed_lines = self._classify(lines, **extra_sections)

    def _classify(
        self,
        docstring_lines: List[str],
        **extra_sections: List[str],
    ) -> Dict[str, List[str]]:
        """Classify formatted docstring into sections."""
        sectioned_docstrings = dict()

        for sec_key, parsed_lines in extra_sections.items():
            if sec_key not in self.__sections__:
                raise KeyError(
                    f"Section key {sec_key} is not a valid Qiskit Experiments extension "
                    f"section keys. Use one of {','.join(self.__sections__.keys())}."
                )
            sectioned_docstrings[sec_key] = parsed_lines

        current_section = "header"
        min_indent = sys.maxsize
        tmp_lines = []
        for line in docstring_lines:
            matched = _section_regex.match(line)
            if matched:
                # Process previous section
                if min_indent < sys.maxsize:
                    tmp_lines = [_line[min_indent:] for _line in tmp_lines]
                parser = self.__sections__[current_section]
                sectioned_docstrings[current_section] = parser(tmp_lines)
                # Start new line
                sec_key = matched["section_key"]
                if sec_key not in self.__sections__:
                    raise KeyError(
                        f"Section key {sec_key} is not a valid Qiskit Experiments extension "
                        f"section keys. Use one of {','.join(self.__sections__.keys())}."
                    )
                current_section = sec_key
                tmp_lines.clear()
                min_indent = sys.maxsize
                continue
            # calculate section indent
            if len(line) > 0 and not line.isspace():
                # ignore empty line
                indent = len(line) - len(line.lstrip())
                min_indent = min(indent, min_indent)
            tmp_lines.append(line)
        # Process final section
        if tmp_lines:
            if min_indent < sys.maxsize:
                tmp_lines = [_line[min_indent:] for _line in tmp_lines]
            parser = self.__sections__[current_section]
            sectioned_docstrings[current_section] = parser(tmp_lines)

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
        "manual": load_standard_section,
        "analysis_ref": load_standard_section,
        "experiment_opts": load_standard_section,
        "example": load_standard_section,
        "note": load_standard_section,
        "see_also": load_standard_section,
        "init": load_standard_section,
    }

    __formatter__ = ExperimentSectionFormatter

    def _extra_sections(self, sectioned_docstring: Dict[str, List[str]]):
        """Generate extra sections."""
        current_class = self._target_cls

        # add see also for super classes
        if "see_also" not in sectioned_docstring:
            class_refs = _get_superclass(current_class, BaseExperiment)
            if class_refs:
                sectioned_docstring["see_also"] = class_refs

        # add analysis reference, if nothing described, it copies from parent
        exp_docs_config = copy.copy(self._config)
        exp_docs_config.napoleon_custom_sections = [("experiment options", "args")]

        if not sectioned_docstring.get("analysis_ref", None):
            analysis_desc = _generate_analysis_ref(
                current_class=current_class,
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
        "manual": load_standard_section,
        "analysis_opts": load_standard_section,
        "example": load_standard_section,
        "note": load_standard_section,
        "see_also": load_standard_section,
        "init": load_standard_section,
    }

    __formatter__ = AnalysisSectionFormatter

    def _extra_sections(self, sectioned_docstring: Dict[str, List[str]]):
        """Generate extra sections."""
        current_class = self._target_cls

        # add see also for super classes
        if "see_also" not in sectioned_docstring:
            class_refs = _get_superclass(current_class, BaseAnalysis)
            if class_refs:
                sectioned_docstring["see_also"] = class_refs


class VisualizationDocstring(QiskitExperimentDocstring):
    """Documentation parser for visualization classes' introductions."""

    __sections__ = {
        "header": load_standard_section,
        "warning": load_standard_section,
        "overview": load_standard_section,
        "reference": load_standard_section,
        "manual": load_standard_section,
        "opts": load_standard_section,  # For standard options
        "figure_opts": load_standard_section,  # For figure options
        "example": load_standard_section,
        "note": load_standard_section,
        "see_also": load_standard_section,
        "init": load_standard_section,
    }

    __formatter__ = VisualizationSectionFormatter
