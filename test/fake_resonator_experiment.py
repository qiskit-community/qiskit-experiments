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

from typing import Iterable, Optional, Tuple, List, Dict
from qiskit_experiments.framework import BaseExperiment, Options, BaseAnalysis, AnalysisResultData

from qiskit.providers import Backend
from qiskit.circuit import QuantumCircuit

class FakeResonatorAnalysis(BaseAnalysis):
    # pylint: disable=arguments-differ
    def _run_analysis(
            self,
            experiment_data,
    ) -> Tuple[List[AnalysisResultData], List["matplotlib.figure.Figure"]]:
        return [], None


class FakeResonatorExperiment(BaseExperiment):
    """
    An experiment to show how to use resonators instead of qubits
    """

    @classmethod
    def _default_experiment_options(cls) -> Options:
        """
        Add resonators to the experiment options
        """
        options = super()._default_experiment_options()
        options.resonators = []
        return options

    def __init__(
            self,
            resonators: Iterable[int]):
        super().__init__([])

        # Set experiment options
        self.set_experiment_options(resonators=resonators)

    def circuits(self, backend: Optional[Backend] = None):
        circ = QuantumCircuit(1, 1)
        circ.measure(0, 0)

        return [circ]