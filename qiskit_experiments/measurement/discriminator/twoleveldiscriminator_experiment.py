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
Standard discriminator experiment class.
"""

from typing import List, Optional

from qiskit_experiments.base_experiment import BaseExperiment

from qiskit.circuit import QuantumCircuit
from qiskit.qobj.utils import MeasLevel
from qiskit.providers.options import Options

from .twoleveldiscriminator_analysis import TwoLevelDiscriminatorAnalysis


class TwoLevelDiscriminator(BaseExperiment):
    """0 and 1 Discriminator Experiment class"""

    # Analysis class for experiment
    __analysis_class__ = TwoLevelDiscriminatorAnalysis

    @classmethod
    def _default_run_options(cls) -> Options:
        """Default options values for the experiment :meth:`run` method."""
        return Options(
            meas_level=MeasLevel.KERNELED,
            meas_return="single",
        )

    def __init__(
        self,
        qubit: int,
    ):
        """Standard discriminator experiment

        Args:
            qubit: The qubit to discriminate on.
        """

        super().__init__([qubit])

    def circuits(self, backend: Optional["Backend"] = None) -> List[QuantumCircuit]:
        """Return a list of discriminator circuits.
        Args:
            backend (Backend): Optional, a backend object.
        Returns:
            List[QuantumCircuit]: A list of :class:`QuantumCircuit`s.
        """
        circuits = []
        for label in (0, 1):
            circ = QuantumCircuit(1)
            if label == 1:
                circ.x(0)
            circ.measure_all()

            circ.metadata = {
                "experiment_type": self._type,
                "ylabel": str(label),
                "qubit": self.physical_qubits[0],
                "meas_level": self.run_options.meas_level,
                "meas_return": self.run_options.meas_return,
            }
            circuits.append(circ)

        return circuits
