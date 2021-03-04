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
T1 Experiment class.
"""

from typing import List, Optional, Union, Tuple
import numpy as np
from scipy.optimize import curve_fit

from qiskit.circuit import QuantumCircuit

from qiskit_experiments.base_experiment import BaseExperiment
from qiskit_experiments.base_analysis import BaseAnalysis
from qiskit_experiments import AnalysisResult


class T1Analysis(BaseAnalysis):
    """T1 Experiment result analysis class."""

    def _run_analysis(self, experiment_data, **params) -> Tuple[AnalysisResult, None]:
        """
        Calculate T1

        Args:
            experiment_data: the experiment data to analyze
            params: expected parameters therein are:
                   `fit_p0` and `fit_bounds` - to be passed to scipy.optimize.curve_fit
                                               as the `p0` and `bounds` parameters

        Returns:
            The analysis result with the estimated T1
        """

        prob1 = {}
        for circ in experiment_data._data:
            delay = circ["metadata"]["delay"]
            count0 = circ["counts"].get("0", 0)
            count1 = circ["counts"].get("1", 0)
            shots = count0 + count1
            mean = count1 / shots
            std = np.sqrt(mean * (1 - mean) / shots)
            # problem for the fitter if one of the std points is
            # exactly zero
            if std == 0:
                std = 1e-4
            prob1[delay] = (mean, std)

        delays = []
        means = []
        stds = []
        for delay in sorted(prob1):
            delays.append(delay)
            means.append(prob1[delay][0])
            stds.append(prob1[delay][1])

        def exp_fit_fun(x, a, tau, c):
            return a * np.exp(-x / tau) + c

        fit_out, _ = curve_fit(
            exp_fit_fun, delays, means, sigma=stds, p0=params["fit_p0"], bounds=params["fit_bounds"]
        )

        analysis_result = AnalysisResult({"value": fit_out[1]})
        return analysis_result, None


class T1Experiment(BaseExperiment):
    """T1 experiment class"""

    __analysis_class__ = T1Analysis

    def __init__(self, qubit: int, delays: Union[List[float], np.array], unit: str = "dt"):
        """
        Initialize the T1 experiment class

        Args:
            qubit: the qubit whose T1 is to be estimated
            delays: delay times of the experiments
            unit: time unit of `delays`
        """
        self._delays = delays
        self._unit = unit
        super().__init__([qubit], type(self).__name__)

    def circuits(
        self, backend: Optional["Backend"] = None, **circuit_options
    ) -> List[QuantumCircuit]:
        """
        Return a list of experiment circuits

        Args:
            backend: a backend object
            circuit_options: kwarg options for the function

        Returns:
            The experiment circuits
        """

        circuits = []

        for circ_index, delay in enumerate(self._delays):
            circ = QuantumCircuit(1, 1)
            circ.x(0)
            circ.delay(delay, 0, self._unit)
            circ.measure(0, 0)

            # pylint: disable = eval-used
            circ.metadata = {
                "experiment_type": self._type,
                "qubit": self.physical_qubits[0],
                "delay": delay,
            }

            circuits.append(circ)

        return circuits
