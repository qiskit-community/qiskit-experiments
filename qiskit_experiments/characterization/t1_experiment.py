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
from .analysis_functions import exp_fit_fun, curve_fit_wrapper


class T1Analysis(BaseAnalysis):
    """T1 Experiment result analysis class."""

    # pylint: disable=arguments-differ, unused-argument
    def _run_analysis(
        self, experiment_data, t1_guess=None, amplitude_guess=None, offset_guess=None, **kwargs
    ) -> Tuple[AnalysisResult, None]:
        """
        Calculate T1

        Args:
            experiment_data (ExperimentData): the experiment data to analyze
            t1_guess (float): Optional, an initial guess of T1
            amplitude_guess (float): Optional, an initial guess of the coefficient of the exponent
            offset_guess (float): Optional, an initial guess of the offset
            kwargs: Trailing unused function parameters

        Returns:
            The analysis result with the estimated T1
        """

        circuit_unit = experiment_data._data[0]["metadata"]["unit"]
        dt_factor_in_sec = experiment_data._data[0]["metadata"]["dt_factor_in_sec"]
        if dt_factor_in_sec is None:
            dt_factor_in_microsec = 1
            result_unit = circuit_unit
        else:
            dt_factor_in_microsec = dt_factor_in_sec * 1000000
            result_unit = "us"

        size = len(experiment_data._data)
        delays = np.zeros(size, dtype=float)
        means = np.zeros(size, dtype=float)
        stddevs = np.zeros(size, dtype=float)

        for i, circ in enumerate(experiment_data._data):
            delays[i] = circ["metadata"]["delay"] * dt_factor_in_microsec
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
        else:
            t1_guess = t1_guess * dt_factor_in_microsec
        if offset_guess is None:
            offset_guess = means[-1]
        if amplitude_guess is None:
            amplitude_guess = means[0] - offset_guess

        fit_out, fit_err, fit_cov, chisq = curve_fit_wrapper(
            exp_fit_fun,
            delays,
            means,
            stddevs,
            p0=[amplitude_guess, t1_guess, offset_guess],
        )

        analysis_result = AnalysisResult(
            {
                "value": fit_out[1],
                "stderr": fit_err[1],
                "unit": result_unit,
                "label": "T1",
                "fit": {
                    "params": fit_out,
                    "stderr": fit_err,
                    "labels": ["amplitude", "T1", "offset"],
                    "chisq": chisq,
                    "cov": fit_cov,
                },
                "quality": self._fit_quality(fit_out, fit_err, chisq),
            }
        )

        return analysis_result, None

    @staticmethod
    def _fit_quality(fit_out, fit_err, chisq):
        # pylint: disable = too-many-boolean-expressions
        if (
            abs(fit_out[0] - 1.0) < 0.1
            and abs(fit_out[2]) < 0.1
            and chisq < 3
            and (fit_err[0] is None or fit_err[0] < 0.1)
            and (fit_err[1] is None or fit_err[1] < fit_out[1])
            and (fit_err[2] is None or fit_err[2] < 0.1)
        ):
            return "computer_good"
        else:
            return "computer_bad"


class T1Experiment(BaseExperiment):
    """T1 experiment class"""

    __analysis_class__ = T1Analysis

    def __init__(
        self,
        qubit: int,
        delays: Union[List[float], np.array],
        unit: Optional[str] = "us",
        experiment_type: Optional[str] = None,
    ):
        """
        Initialize the T1 experiment class

        Args:
            qubit: the qubit whose T1 is to be estimated
            delays: delay times of the experiments
            unit:Optional, unit of the duration. Supported units: 's', 'ms', 'us', 'ns', 'ps', 'dt'.
            experiment_type: Optional, the experiment type string.

        Raises:
            ValueError: if the number of delays is smaller than 3
        """
        if len(delays) < 3:
            raise ValueError("T1 experiment: number of delays must be at least 3")

        self._delays = delays
        self._unit = unit
        super().__init__([qubit], experiment_type)

    # pylint: disable=arguments-differ
    def circuits(self, backend: Optional["Backend"] = None) -> List[QuantumCircuit]:
        """
        Return a list of experiment circuits

        Args:
            backend: Optional, a backend object

        Returns:
            The experiment circuits

        Raises:
            AttributeError: if unit is dt but dt parameter is missing in the backend configuration
        """

        if self._unit == "dt":
            try:
                dt_factor_in_sec = getattr(backend.configuration(), "dt")
            except AttributeError as no_dt:
                raise AttributeError("Dt parameter is missing in backend configuration") from no_dt
        else:
            dt_factor_in_sec = None

        circuits = []

        for delay in self._delays:
            circ = QuantumCircuit(1, 1)
            circ.x(0)
            circ.barrier(0)
            circ.delay(delay, 0, self._unit)
            circ.barrier(0)
            circ.measure(0, 0)

            circ.metadata = {
                "experiment_type": self._type,
                "qubit": self.physical_qubits[0],
                "delay": delay,
                "unit": self._unit,
                "dt_factor_in_sec": dt_factor_in_sec,
            }

            circuits.append(circ)

        return circuits
