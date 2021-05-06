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

from typing import List, Optional, Union, Tuple, Dict
import numpy as np
import copy

import qiskit
from qiskit.circuit import QuantumCircuit
from qiskit.utils import apply_prefix
from qiskit_experiments.base_experiment import BaseExperiment
from qiskit_experiments.base_analysis import BaseAnalysis, AnalysisResult
from ..experiment_data import ExperimentData
from qiskit_experiments.analysis.curve_fitting import curve_fit, multi_curve_fit, process_curve_data
from qiskit_experiments.analysis.plotting import plot_curve_fit, plot_scatter, plot_errorbar
from qiskit_experiments.analysis.data_processing import level2_probability

try:
    from matplotlib import pyplot as plt

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


class T2StarAnalysis(BaseAnalysis):
    """T2Star Experiment result analysis class."""

    # pylint: disable=arguments-differ, unused-argument
    def _run_analysis(
        self,
        experiment_data: ExperimentData,
        p0: List[float],
        bounds: Tuple[List[float], Tuple[List[float]]],
        plot: bool = True,
        ax: Optional["AxesSubplot"] = None,
        **kwargs,
    ):
        r"""
            Calculate T2Star experiment
            The probabilities of measuring 0 is assumed to be of the form
        .. math::
            f(t) = A\mathrm{e}^{-t / T_2^*}\cos(2\pi ft + \phi) + B
        for unknown parameters :math:`A, B, f, \phi, T_2^*`.

            Args:
                experiment_data (ExperimentData): the experiment data to analyze

                p0: contains initial values for the fit parameters :math:`(A, T_2^*, f, \phi, B)`
                bounds: lower and upper bounds on the parameters in p0.
                        The first tuple is the lower bounds,
                        The second tuple is the upper bounds.
                        For both params, the order is :math:`A, T_2^*, f, \phi, B`.
            Returns:
                The analysis result with the estimated :math:`T_2^*` and 'f' (frequency)

        """

        def osc_fit_fun(x, a, t2star, f, phi, c):
            """
            Decay cosine fit function
            """
            return a * np.exp(-x / t2star) * np.cos(2 * np.pi * f * x + phi) + c

        def _format_plot(ax, unit):
            """Format curve fit plot"""
            # Formatting
            ax.tick_params(labelsize=10)
            ax.set_xlabel("Delay (" + str(unit) + ")", fontsize=12)
            ax.set_ylabel("Probability to measure |0>", fontsize=12)

        # implementation of  _run_analysis
        self._p0 = p0
        self._bounds = bounds
        unit = experiment_data._data[0]["metadata"]["unit"]
        self._conversion_factor = experiment_data._data[0]["metadata"].get("dt_factor", None)
        if self._conversion_factor is None:
            self._conversion_factor = 1 if unit == "s" else apply_prefix(1, unit)
        xdata, ydata, sigma = process_curve_data(
            experiment_data._data, lambda datum: level2_probability(datum, "1")
        )

        si_xdata = xdata * self._conversion_factor
        t2star_estimate = np.mean(si_xdata)

        p0, bounds = self._t2star_default_params(t2star=t2star_estimate)
        fit_result = curve_fit(
            osc_fit_fun, si_xdata, ydata, p0=list(p0.values()), sigma=sigma, bounds=bounds
        )

        if plot:
            ax = plot_curve_fit(osc_fit_fun, fit_result, ax=ax)
            ax = plot_scatter(si_xdata, ydata, ax=ax)
            ax = plot_errorbar(si_xdata, ydata, sigma, ax=ax)
            _format_plot(ax, unit)
            fit_result.plt = plt
            plt.show()

        analysis_result = AnalysisResult(
            {
                "T2star_value": fit_result["popt"][1],
                "Frequency_value": fit_result["popt"][2],
                "stderr": fit_result["popt_err"][1],
                "unit": "s",
                "label": "T2*",
                "fit": fit_result,
                "quality": self._fit_quality(
                    p0, fit_result["popt"], fit_result["popt_err"], fit_result["reduced_chisq"]
                ),
            }
        )

        analysis_result["fit"]["circuit_unit"] = unit
        if unit == "dt":
            analysis_result["fit"]["dt"] = self._conversion_factor
        return analysis_result, None

    def _t2star_default_params(
        self,
        t2star: float,
    ) -> Tuple[List[float], Tuple[List[float]]]:
        """
        Default fit parameters for oscillation data
        Args:
            t2star: default for t2star if p0==None
        Returns:
            Fit guessed parameters: either from the input (if given) or
            else assign default values.
        """
        if self._p0 == None:
            A = 0.5
            t2star = t2star
            f = 0.1
            phi = 0.0
            B = 0.5
        else:
            A = self._p0["A"]
            t2star = self._p0["t2star"]
            t2star *= self._conversion_factor
            f = self._p0["f"]
            phi = self._p0["phi"]
            B = self._p0["B"]
        f /= self._conversion_factor
        p0 = {"A_guess": A, "t2star": t2star, "f_guess": f, "phi_guess": phi, "B_guess": B}
        if self._bounds == None:
            A_bounds = [-0.5, 1.5]
            t2star_bounds = [0, np.inf]
            f_bounds = [0.5 * f, 1.5 * f]
            phi_bounds = [-np.pi, np.pi]
            B_bounds = [-0.5, 1.5]
            bounds = (
                [A_bounds[0], t2star_bounds[0], f_bounds[0], phi_bounds[0], B_bounds[0]],
                [A_bounds[1], t2star_bounds[1], f_bounds[1], phi_bounds[1], B_bounds[1]],
            )
        return p0, bounds

    @staticmethod
    def _fit_quality(p0, fit_out, fit_err, reduced_chisq):
        # pylint: disable = too-many-boolean-expressions
        if (
            np.allclose(fit_out, [0.5, p0["t2star"], p0["f_guess"], 0, 0.5], rtol=0.3, atol=0.1)
            and (reduced_chisq < 3)
            and (fit_err[0] is None or fit_err[0] < 0.1 * fit_out[0])
            and (fit_err[1] is None or fit_err[1] < 0.1 * fit_out[1])
            and (fit_err[2] is None or fit_err[2] < 0.1 * fit_out[2])
        ):
            return "computer_good"
        else:
            return "computer_bad"


class T2StarExperiment(BaseExperiment):
    """T2Star experiment class"""

    __analysis_class__ = T2StarAnalysis

    def __init__(
        self,
        qubit: int,
        delays: Union[List[float], np.array],
        unit: Optional[str] = "s",
        osc_freq: float = 0.0,
        experiment_type: Optional[str] = None,
    ):

        """Initialize the T2Star experiment class.

        Args:
            qubit: the qubit under test
            delays: delay times of the experiments
            unit: Optional, time unit of `delays`. Supported units: 's', 'ms', 'us', 'ns', 'ps', 'dt'.
            osc_freq: the oscillation frequency induced using by the user
            experiment_type: String indicating the experiment type. Can be 'RamseyExperiment' or 'T2StarExperiment'.
        """

        self._qubit = qubit
        self._delays = delays
        self._unit = unit
        self._osc_freq = osc_freq
        super().__init__([qubit], experiment_type)

    def circuits(self, backend: Optional["Backend"] = None) -> List[QuantumCircuit]:
        """
        Return a list of experiment circuits
        Each circuit consists of a Hadamard gate, followed by a fixed delay, a phase gate (with a linear phase),
        and an additional Hadamard gate.
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
            conversion_factor = dt_factor
        else:
            conversion_factor = 1 if self._unit == "s" else apply_prefix(1, self._unit)

        xdata = self._delays

        circuits = []
        for delay in self._delays:
            circ = qiskit.QuantumCircuit(1, 1)
            circ.name = "T2Starcircuit_" + str(delay)
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
