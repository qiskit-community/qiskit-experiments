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
T1 Experiment class.
"""

from typing import List, Optional, Union
import numpy as np

from qiskit.providers import Backend
from qiskit.circuit import QuantumCircuit
from qiskit.providers.options import Options

from qiskit_experiments.framework import BaseExperiment
from qiskit_experiments.library.characterization.t1_analysis import T1Analysis


class T1(BaseExperiment):
    r"""
    T1 experiment class

    Experiment Options:
        * delays: delay times of the experiments
        * unit: Optional, unit of the delay times. Supported units are
                's', 'ms', 'us', 'ns', 'ps', 'dt'.

    Design and analyze experiments for estimating T\ :sub:`1` of the device.

    Each experiment consists of the following steps:
    1. Circuits generation: the circuits set the qubit in the excited state,
    wait different time intervals, then measure the qubit.
    2. Backend execution: actually running the circuits on the device
    (or simulator).
    3. Analysis of results: deduction of T\ :sub:`1`\ , based on the outcomes,
    by fitting to an exponential curve.

    """

    __analysis_class__ = T1Analysis

    @classmethod
    def _default_experiment_options(cls) -> Options:
        return Options(delays=None, unit="s")

    def __init__(
        self,
        qubit: int,
        delays: Union[List[float], np.array],
        unit: Optional[str] = "s",
    ):
        """
        Initialize the T1 experiment class

        Args:
            qubit: the qubit whose T1 is to be estimated
            delays: delay times of the experiments
            unit: Optional, unit of the delay times. Supported units:
                  's', 'ms', 'us', 'ns', 'ps', 'dt'.

        Raises:
            ValueError: if the number of delays is smaller than 3
        """
        if len(delays) < 3:
            raise ValueError("T1 experiment: number of delays must be at least 3")

        # Initialize base experiment
        super().__init__([qubit])

        # Set experiment options
        self.set_experiment_options(delays=delays, unit=unit)

    def circuits(self, backend: Optional[Backend] = None) -> List[QuantumCircuit]:
        """
        Return a list of experiment circuits

        Args:
            backend: Optional, a backend object

        Returns:
            The experiment circuits

        Raises:
            AttributeError: if unit is dt but dt parameter is missing in the backend configuration
        """
        if self.experiment_options.unit == "dt":
            try:
                dt_factor = getattr(backend.configuration(), "dt")
            except AttributeError as no_dt:
                raise AttributeError("Dt parameter is missing in backend configuration") from no_dt

        circuits = []

        for delay in self.experiment_options.delays:
            circ = QuantumCircuit(1, 1)
            circ.x(0)
            circ.barrier(0)
            circ.delay(delay, 0, self.experiment_options.unit)
            circ.barrier(0)
            circ.measure(0, 0)

            circ.metadata = {
                "experiment_type": self._type,
                "qubit": self.physical_qubits[0],
                "xval": delay,
                "unit": self.experiment_options.unit,
            }

            if self.experiment_options.unit == "dt":
                circ.metadata["dt_factor"] = dt_factor

            circuits.append(circ)

        return circuits
