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
Readout Angle Experiment class.
"""

from typing import List, Optional, Union
import numpy as np

from qiskit.circuit import QuantumCircuit
from qiskit.qobj.utils import MeasLevel
from qiskit.providers.backend import Backend
from qiskit.test.mock import FakeBackend

from qiskit_experiments.framework import BaseExperiment, Options
from qiskit_experiments.library.characterization.readout_angle_analysis import ReadoutAngleAnalysis


class ReadoutAngle(BaseExperiment):
    r"""
    Readout angle experiment class

    # section: overview

        Design and analyze experiments for estimating readout angle of the qubit.

        Each experiment consists of the following steps:

        1. Circuits generation: two circuits, the first circuit measures the qubit
         in the ground state, the second circuit sets the qubit in the excited state
         and measures it. Measurements are in level 1 (kerneled).

        2. Backend execution: actually running the circuits on the device
        (or a simulator that supports level 1 measurements).

        3. Analysis of results: return the average of the centers of the two angles.

    """

    __analysis_class__ = ReadoutAngleAnalysis

    @classmethod
    def _default_run_options(cls) -> Options:
        """Default run options."""
        options = super()._default_run_options()
        
        options.meas_level = MeasLevel.KERNELED
        options.meas_return = "avg"
        
        return options

    def __init__(
        self,
        qubit: int,
        backend: Optional[Backend] = None,
    ):
        """
        Initialize the readout angle experiment class

        Args:
            qubit: the qubit whose readout angle is to be estimated
            backend: Optional, the backend to run the experiment on.
        """
        # Initialize base experiment
        super().__init__([qubit], backend=backend)

    def circuits(self) -> List[QuantumCircuit]:
        """
        Return a list of experiment circuits

        Returns:
            The experiment circuits
        """
        circ0 = QuantumCircuit(1, 1)
        circ0.measure(0, 0)

        circ1 = QuantumCircuit(1, 1)
        circ1.x(0)
        circ1.measure(0, 0)

        for i, circ in enumerate([circ0, circ1]):
            circ.metadata = {
                "experiment_type": self._type,
                "qubit": self.physical_qubits[0],
                "xval": i,
            }

        return [circ0, circ1]
