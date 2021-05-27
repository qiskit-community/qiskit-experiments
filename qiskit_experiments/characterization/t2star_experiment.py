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

import qiskit
from qiskit.providers import Backend
from qiskit.circuit import QuantumCircuit
from qiskit.utils import apply_prefix
from qiskit.providers.options import Options
from qiskit_experiments.base_experiment import BaseExperiment
from qiskit_experiments.base_analysis import BaseAnalysis, AnalysisResult
from qiskit_experiments.analysis.curve_fitting import curve_fit, process_curve_data
from qiskit_experiments.analysis.data_processing import level2_probability
from qiskit_experiments.analysis import plotting
from ..experiment_data import ExperimentData

# pylint: disable = invalid-name
class T2StarAnalysis(BaseAnalysis):
    """T2Star Experiment result analysis class."""

    @classmethod
    def _default_options(cls):
        return Options(user_p0=None, user_bounds=None)

    # pylint: disable=arguments-differ, unused-argument
    def _run_analysis(
        self,
        experiment_data: ExperimentData,
        user_p0: Optional[Dict[str, float]] = None,
        user_bounds: Optional[Tuple[List[float], List[float]]] = None,
        plot: bool = True,
        ax: Optional["AxesSubplot"] = None,
        **kwargs,
    ) -> Tuple[AnalysisResult, List["matplotlib.figure.Figure"]]:
        r"""Calculate T2Star experiment.

        The probability of measuring `+` is assumed to be of the form
        :math:`f(t) = a\mathrm{e}^{-t / T_2^*}\cos(2\pi freq t + \phi) + b`
        for unknown parameters :math:`a, b, freq, \phi, T_2^*`.

        Args:
            experiment_data (ExperimentData): the experiment data to analyze
            user_p0: contains initial values given by the user, for the
            fit parameters :math:`(a, T_2^*, freq, \phi, b)`
            User_bounds: lower and upper bounds on the parameters in p0,
                         given by the user.
                         The first tuple is the lower bounds,
                         The second tuple is the upper bounds.
                         For both params, the order is :math:`a, T_2^*, freq, \phi, b`.
            plot: if True, create the plot, otherwise, do not create the plot.
            ax: the plot object
            **kwargs: additional parameters for curve fit.

        Returns:
            The analysis result with the estimated :math:`T_2^*` and 'freq' (frequency)
            The graph of the function.
        """

        def osc_fit_fun(x, a, t2star, freq, phi, c):
            """Decay cosine fit function"""
            return a * np.exp(-x / t2star) * np.cos(2 * np.pi * freq * x + phi) + c

        def _format_plot(ax, unit):
            """Format curve fit plot"""
            # Formatting
            ax.tick_params(labelsize=10)
            ax.set_xlabel("Delay (" + str(unit) + ")", fontsize=12)
            ax.set_ylabel("Probability to measure |0>", fontsize=12)

        # implementation of  _run_analysis
        unit = experiment_data._data[0]["metadata"]["unit"]
        conversion_factor = experiment_data._data[0]["metadata"].get("dt_factor", None)
        if conversion_factor is None:
            conversion_factor = 1 if unit == "s" else apply_prefix(1, unit)
        xdata, ydata, sigma = process_curve_data(
            experiment_data._data, lambda datum: level2_probability(datum, "0")
        )

        si_xdata = xdata * conversion_factor
        t2star_estimate = np.mean(si_xdata)

        p0, bounds = self._t2star_default_params(
            conversion_factor, user_p0, user_bounds, t2star_estimate
        )
        fit_result = curve_fit(
            osc_fit_fun, si_xdata, ydata, p0=list(p0.values()), sigma=sigma, bounds=bounds
        )

        if plot and plotting.HAS_MATPLOTLIB:
            ax = plotting.plot_curve_fit(osc_fit_fun, fit_result, ax=ax)
            ax = plotting.plot_scatter(si_xdata, ydata, ax=ax)
            ax = plotting.plot_errorbar(si_xdata, ydata, sigma, ax=ax)
            _format_plot(ax, unit)
            figures = [ax.get_figure()]
        else:
            figures = None

        # Output unit is 'sec', regardless of the unit used in the input
        analysis_result = AnalysisResult(
            {
                "t2star_value": fit_result["popt"][1],
                "frequency_value": fit_result["popt"][2],
                "stderr": fit_result["popt_err"][1],
                "unit": "s",
                "label": "T2*",
                "fit": fit_result,
                "quality": self._fit_quality(
                    fit_result["popt"], fit_result["popt_err"], fit_result["reduced_chisq"]
                ),
            }
        )

        analysis_result["fit"]["circuit_unit"] = unit
        if unit == "dt":
            analysis_result["fit"]["dt"] = conversion_factor
        return analysis_result, figures

    def _t2star_default_params(
        self,
        conversion_factor,
        user_p0=None,
        user_bounds=None,
        t2star_input=None,
    ) -> Tuple[List[float], Tuple[List[float]]]:
        """Default fit parameters for oscillation data.

        Note that :math:`T_2^*` and 'freq' units are converted to 'sec' and
        will be output in 'sec'.
        """
        if user_p0 is None:
            a = 0.5
            t2star = t2star_input * conversion_factor
            freq = 0.1
            phi = 0.0
            b = 0.5
        else:
            a = user_p0["A"]
            t2star = user_p0["t2star"]
            t2star *= conversion_factor
            freq = user_p0["f"]
            phi = user_p0["phi"]
            b = user_p0["B"]
        freq /= conversion_factor
        p0 = {"a_guess": a, "t2star": t2star, "f_guess": freq, "phi_guess": phi, "b_guess": b}
        if user_bounds is None:
            a_bounds = [-0.5, 1.5]
            t2star_bounds = [0, np.inf]
            f_bounds = [0.5 * freq, 1.5 * freq]
            phi_bounds = [-np.pi, np.pi]
            b_bounds = [-0.5, 1.5]
            bounds = [
                [a_bounds[i], t2star_bounds[i], f_bounds[i], phi_bounds[i], b_bounds[i]]
                for i in range(2)
            ]
        else:
            bounds = user_bounds
        return p0, bounds

    @staticmethod
    def _fit_quality(fit_out, fit_err, reduced_chisq):
        # pylint: disable = too-many-boolean-expressions
        if (
            (reduced_chisq < 3)
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
        unit: str = "s",
        osc_freq: float = 0.0,
        experiment_type: Optional[str] = None,
    ):
        """Initialize the T2Star experiment class.

        Args:
            qubit: the qubit under test
            delays: delay times of the experiments
            unit: Optional, time unit of `delays`.
            Supported units: 's', 'ms', 'us', 'ns', 'ps', 'dt'.
            The unit is used for both T2* and the frequency
            osc_freq: the oscillation frequency induced using by the user
            experiment_type: String indicating the experiment type.
            Can be 'RamseyExperiment' or 'T2StarExperiment'.
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
