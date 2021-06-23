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
Documentation for experiment.
"""
from typing import Optional, Dict, List

from qiskit.exceptions import QiskitError

from qiskit_experiments.base_experiment import BaseExperiment
from .descriptions import OptionsField, Reference
from .writer import _DocstringWriter, _DocstringMaker


class StandardExperimentDocstring(_DocstringMaker):
    """A facade class to write standard experiment docstring."""

    @classmethod
    def make_docstring(
            cls,
            analysis_options: Dict[str, OptionsField],
            experiment_options: Dict[str, OptionsField],
            analysis: str,
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
            writer.write_lines("This experiment uses following analysis class.")
            writer.write_section(f":py:class:`~{analysis}`", "Analysis Class Reference")
            writer.write_lines("Experiment options to generate circuits. \
Options can be updated with :py:meth:`set_experiment_options`. \
See method documentation for details.")
            writer.write_options_as_sections(experiment_options, "Experiment Options")
            writer.write_lines("Analysis options to run the analysis class. \
Options can be updated with :py:meth:`set_analysis_options`. \
See method documentation for details.")
            writer.write_options_as_sections(analysis_options, "Analysis Options")
            if references:
                writer.write_references(references)
            if note:
                writer.write_note(note)
            if tutorial:
                writer.write_tutorial_link(tutorial)
        except Exception as ex:
            raise QiskitError(f"Auto docstring generation failed with the error: {ex}")
        return writer.docstring


def auto_experiment_documentation(style: _DocstringMaker = StandardExperimentDocstring):
    """A class decorator that overrides experiment class docstring."""
    def decorator(experiment: BaseExperiment):
        analysis = experiment.__analysis_class__

        exp_docs = style.make_docstring(
            analysis_options=experiment.__analysis_class__._default_options(),
            experiment_options=experiment._default_experiment_options(),
            analysis=f"{analysis.__module__}.{analysis.__name__}",
            overview=getattr(experiment, "__doc_overview__", None),
            example=getattr(experiment, "__doc_example__", None),
            references=getattr(experiment, "__doc_references__", None),
            note=getattr(experiment, "__doc_note__", None),
            warning=getattr(experiment, "__doc_warning__", None),
            tutorial=getattr(experiment, "__doc_tutorial__", None),
        )
        experiment.__doc__ += f"\n\n{exp_docs}"

        return experiment
    return decorator
