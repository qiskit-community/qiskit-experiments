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

"""Fake experiment using resonator instead of qubits for testing."""

from typing import Iterable, Tuple, List
from qiskit_experiments.framework import BaseExperiment, Options, BaseAnalysis, AnalysisResultData


class FakeResonatorAnalysis(BaseAnalysis):
    """
    Simple analysis to test experiment using resonators instead of qubits
    """

    # pylint: disable=arguments-differ
    def _run_analysis(
        self,
        experiment_data,
    ) -> Tuple[List[AnalysisResultData], List["matplotlib.figure.Figure"]]:
        return [AnalysisResultData("ResonatorTest", 0)], None


class FakeResonatorExperiment(BaseExperiment):
    """
    An experiment to show how to use resonators instead of qubits
    """

    __analysis_class__ = FakeResonatorAnalysis

    @classmethod
    def _default_experiment_options(cls) -> Options:
        """
        Add resonators to the experiment options
        """
        options = super()._default_experiment_options()
        options.resonators = []
        return options

    def __init__(self, resonators: Iterable[int]):
        super().__init__([])

        # Set experiment options
        self.set_experiment_options(resonators=resonators)

    def _additional_metadata(self):
        return {"resonators": self.experiment_options.resonators}

    def circuits(self):
        """return empty circuits for test"""
        return []
