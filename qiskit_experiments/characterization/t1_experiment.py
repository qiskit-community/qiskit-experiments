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

from qiskit.circuit import QuantumCircuit

from qiskit_experiments.base_experiment import BaseExperiment
from qiskit_experiments.base_analysis import BaseAnalysis
from qiskit_experiments import AnalysisResult


class T1Analysis(BaseAnalysis):
    """T1 Experiment result analysis class."""

    def _run_analysis(
        self, experiment_data, t1_guess=None, amplitude_guess=None, offset_guess=None, **kwargs
    ) -> Tuple[AnalysisResult, None]:
        """
        Calculate T1

        Args:
            experiment_data (ExperimentData): the experiment data to analyze
            t1_guess: Optional, an initial guess of T1
            amplitude_guess: Optional, an initial guess of the coefficient of the exponent
            offset_guess: Optional, an initial guess of the offset

        Returns:
            The analysis result with the estimated T1
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

        if t1_guess is None:
            t1_guess = np.mean(delays)
        if offset_guess is None:
            offset_guess = means[-1]
        if amplitude_guess is None:
            amplitude_guess = means[0] - offset_guess

        fit_out, fit_err, chisq = BaseAnalysis.curve_fit_wrapper(
            BaseAnalysis.exp_fit_fun,
            delays,
            means,
            stddevs,
            p0=[amplitude_guess, t1_guess, offset_guess],
        )

        analysis_result = AnalysisResult(
            {
                "amplitude": fit_out[0],
                "t1": fit_out[1],
                "offset": fit_out[2],
                "amplitude_err": fit_err[0],
                "t1_err": fit_err[1],
                "offset_err": fit_err[2],
                "chisq": chisq,
            }
        )

        analysis_result["is_good_fit"] = (
            abs(analysis_result["amplitude"] - 1.0) < 0.1
            and abs(analysis_result["offset"]) < 0.1
            and analysis_result["chisq"] < 3
            and (analysis_result["amplitude_err"] is None or analysis_result["amplitude_err"] < 0.1)
            and (analysis_result["offset_err"] is None or analysis_result["offset_err"] < 0.1)
            and (
                analysis_result["t1_err"] is None
                or analysis_result["t1_err"] < analysis_result["t1"]
            )
        )

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
            unit: unit of the duration. Supported units: 's', 'ms', 'us', 'ns', 'ps', 'dt'.
                Default is ``dt``, i.e. integer time unit depending on the target backend.

        Raises:
            ValueError: if the number of delays is smaller than 3
        """
        if len(delays) < 3:
            raise ValueError("T1 experiment: number of delays must be at least 3")

        self._delays = delays
        self._unit = unit
        super().__init__([qubit], type(self).__name__)

    def circuits(self, backend: Optional["Backend"] = None) -> List[QuantumCircuit]:
        """
        Return a list of experiment circuits

        Args:
            backend: Optional, a backend object

        Returns:
            The experiment circuits
        """

        circuits = []

        for delay in self._delays:
            circ = QuantumCircuit(1, 1)
            circ.x(0)
            circ.barrier(0)
            circ.delay(delay, 0, self._unit)
            circ.barrier(0)
            circ.measure(0, 0)

            # pylint: disable = eval-used
            circ.metadata = {
                "experiment_type": self._type,
                "qubit": self.physical_qubits[0],
                "delay": delay,
            }

            circuits.append(circ)

        return circuits
