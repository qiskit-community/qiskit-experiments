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
T2Ramsey Experiment class.

"""

from typing import List, Optional, Union
import numpy as np

import qiskit
from qiskit.providers import Backend
from qiskit.circuit import QuantumCircuit
from qiskit_experiments.base_experiment import BaseExperiment
from .t2ramsey_analysis import T2RamseyAnalysis

class T2Ramsey(BaseExperiment):
    
    r"""
    This experiment is used to estimate two properties for a single qubit:
    $T_2$* and Ramsey frequency.
    The basic circuit used consists of:
    
    #. Hadamard gate
    
    #. Delay
    
    #. Phase gate
    
    #. Hadamard gate
    
    #. Measurement
    
    A series of such circuits is created, with increasing delays, as specified
    by the user.
    The probabilities of measuring 0 is assumed to be of the form
    
    .. math::
        f(t) = A\mathrm{e}^{-t / T_2^*}\cos(2\pi f t + \phi) + B
        
    for unknown parameters :math:`A, B, f, \phi, T_2^*`.
    
    """

    __analysis_class__ = T2RamseyAnalysis

    def __init__(
        self,
        qubit: int,
        delays: Union[List[float], np.array],
        unit: str = "s",
        osc_freq: float = 0.0,
        experiment_type: Optional[str] = None,
    ):
        """
        **T2Ramsey class**
        
        Initialize the T2Ramsey class.

        Args:
            qubit: the qubit under test
            delays: delay times of the experiments
            unit: Optional, time unit of `delays`.
            Supported units: 's', 'ms', 'us', 'ns', 'ps', 'dt'.
            The unit is used for both T2Ramsey and the frequency
            osc_freq: the oscillation frequency induced using by the user
            experiment_type: String indicating the experiment type.
        """

        self._qubit = qubit
        self._delays = delays
        self._unit = unit
        self._osc_freq = osc_freq
        super().__init__([qubit], experiment_type)

    def circuits(self, backend: Optional[Backend] = None) -> List[QuantumCircuit]:
        """Return a list of experiment circuits.

        Each circuit consists of a Hadamard gate, followed by a fixed delay,
        a phase gate (with a linear phase), and an additional Hadamard gate.

        Args:
            backend: Optional, a backend object

        Returns:
            The experiment circuits

        Raises:
            AttributeError: if unit is dt but dt parameter is missing in the backend configuration
        """
        if self._unit == "dt":
            try:
                dt_factor = getattr(backend._configuration, "dt")
            except AttributeError as no_dt:
                raise AttributeError("Dt parameter is missing in backend configuration") from no_dt

        circuits = []
        for delay in self._delays:
            circ = qiskit.QuantumCircuit(1, 1)
            circ.h(0)
            circ.delay(delay, 0, self._unit)
            circ.p(2 * np.pi * self._osc_freq, 0)
            circ.barrier(0)
            circ.h(0)
            circ.barrier(0)
            circ.measure(0, 0)

            circ.metadata = {
                "experiment_type": self._type,
                "qubit": self._qubit,
                "osc_freq": self._osc_freq,
                "xval": delay,
                "unit": self._unit,
            }
            if self._unit == "dt":
                circ.metadata["dt_factor"] = dt_factor

            circuits.append(circ)

        return circuits
