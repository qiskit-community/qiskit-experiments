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
Ramsey Experiment class.
"""

from typing import List, Optional, Tuple
import numpy as np

import qiskit
from qiskit.circuit import QuantumCircuit

from qiskit_experiments.base_experiment import BaseExperiment
from qiskit_experiments.base_analysis import BaseAnalysis
#from qiskit_experiments.experiment_data import Analysis
from .analysis_functions import exp_fit_fun, curve_fit_wrapper

class RamseyAnalysis(BaseAnalysis):
    """Ramsey Experiment result analysis class."""

    def _run_analysis(
            self, experiment_data, **kwargs
    ):
        """
        Calculate Ramsey experiment
        Args:
            experiment_data (ExperimentData): the experiment data to analyze
            kwargs: Trailing unused function parameters
        Returns:
            The analysis result with the estimated Ramsey
        """
        size = len(experiment_data._data)
        delays = np.zeros(size, dtype=float)
        means = np.zeros(size, dtype=float)
        stddevs = np.zeros(size, dtype=float)

        for i, circ in enumerate(experiment_data._data):
            delays[i] = circ["metadata"]["delay"]
            count0 = circ["counts"].get("0", 0)
            count1 = circ["counts"].get("1", 0)
            shots = count0 + count1
            means[i] = count1 / shots
            stddevs[i] = np.sqrt(means[i] * (1 - means[i]) / shots)
            # problem for the fitter if one of the std points is
            # exactly zero
            if stddevs[i] == 0:
                stddevs[i] = 1e-4

        fit_out, fit_err, fit_cov, chisq = curve_fit_wrapper(
            cos_fit_function,
            delays,
            means,
            stddevs,
            p0=[amplitude_guess, t1_guess, offset_guess],
        )

        analysis_result = RamseyAnalysis()
        return analysis_result, None

class RamseyExperiment(BaseExperiment):
    """Ramsey experiment class"""

    __analysis_class__ = RamseyAnalysis

    def __init__(self, qubit, delays, unit, nosc):
    #             qubit: int,
    #             delays: List[float],
    #             unit: str = 'dt',
    #             nosc: int):

        """Initialize the Ramsey experiment object.

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
        experiment_type: str = "RamseyExperiment"
        super().__init__([qubit], experiment_type="RamseyExperiment")

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
            circ.name = 'Ramseycircuit_' + str(delay)
            circ.h(0)
            circ.delay(delay, 0, self._unit)
            circ.p(2 * np.pi * osc_freq, 0)
            circ.barrier(0)
            circ.h(0)
            circ.barrier(0)
            circ.measure(0, 0)
            circuits.append(circ)

        return circuits

