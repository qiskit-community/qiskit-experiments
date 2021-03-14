# -*- coding: utf-8 -*-

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
T2Star Experiment class.
"""
from typing import List, Optional, Union, Tuple
import numpy as np
from qiskit.circuit import QuantumCircuit

from qiskit_experiments.base_experiment import BaseExperiment
from qiskit_experiments.base_analysis import BaseAnalysis

class T2StarExperiment(BaseExperiment):
    """T2Star experiment class"""

    __analysis_class__ = T2StarAnalysis

    def __init__(self, qubit, delays, unit, nosc):
    #             qubit: int,
    #             delays: List[float],
    #             unit: str = 'dt',
   #              nosc: int):

        """Initialize the T2Star experiment object.

        Args:
            qubit (int): the qubit under test
            delays: delay times of the experiments
            unit: time unit of `delays`
            nosc (int): number of oscillations to induce using the phase gate

        Raises:
            QiskitError: ?
        """

        self._qubit = qubit
        self._delays = delays
        self._unit = unit
        self._nosc = nosc
        experiment_type: str = "T2StarExperiment"
        super().__init__([qubit], experiment_type="T2StarExperiment")

    def circuits(self, backend: Optional["Backend"] = None) -> List[QuantumCircuit]:
        """
        Return a list of experiment circuits
        Each circuit consists of a Hadamard gate, followed by a fixed delay, a phase gate (with a linear phase), and an additional Hadamard gate.
        Args:
            backend: Optional, a backend object
        Returns:
            The experiment circuits
        """

        osc_freq = self._nosc

        circuits = []
        for delay in self._delays:
            circ = qiskit.QuantumCircuit(1, 1)
            circ.name = 't2starcircuit_' + str(delay)
            circ.h(0)
            circ.append(U1Gate(2 * np.pi * osc_freq), 0)
            circ.barrier(0)
            circ.h(0)
            circ.barrier(qr)
            circ.measure(0, 0)
        circuits.append(circ)

        return circuits