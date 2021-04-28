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
import qiskit
from qiskit.circuit import QuantumCircuit
from qiskit.utils import apply_prefix
from qiskit_experiments.base_experiment import BaseExperiment
from qiskit_experiments.base_analysis import BaseAnalysis
from qiskit_experiments.analysis.curve_fitting import curve_fit, multi_curve_fit, process_curve_data
from qiskit_experiments.analysis.plotting import plot_curve_fit, plot_scatter, plot_errorbar
from qiskit_experiments.analysis.data_processing import level2_probability
from matplotlib import pyplot as plt

# from qiskit_experiments.experiment_data import Analysis
#from .analysis_functions import exp_fit_fun, curve_fit_wrapper

def temp_plot(xdata, ydata):
    plt.figure(1)
    plt.plot(xdata, ydata,'o--', color='navy', label="my plot")

    plt.xlabel('delay')
    plt.ylabel('prob1')
    plt.title("my graph")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig('my graph')
    plt.show()

class T2StarAnalysis(BaseAnalysis):
    """T2Star Experiment result analysis class."""

    # pylint: disable=arguments-differ, unused-argument
    def _run_analysis(self,
                      experiment_data,
                      p0,
                      bounds,
                      plot: bool = True,
                      ax: Optional["AxesSubplot"] = None,
                      **kwargs):
        r"""
        Calculate T2Star experiment
        The probabilities of measuring 0 is assumed to be of the form
    .. math::
        f(t) = A\mathrm{e}^{-t / T_2^*}\cos(2\pi ft + \phi) + B
    for unknown parameters :math:`A, B, f, \phi, T_2^*`.

        Args:
            experiment_data (ExperimentData): the experiment data to analyze

            params: Includes fit_p0 and fit_bounds.

                    fit_p0 are initial values for the fit parameters :math:`(A, T_2^*, f, \phi, B)`

                    fit_bounds: lower and upper bounds on the parameters in fit_p0.

                    The first tuple is the lower bounds,

                    The second tuple is the upper bounds.

                    For both params, the order is :math:`A, T_2^*, f, \phi, B`.
        Returns:
            The analysis result with the estimated :math:`T_2^*`

        """

        def osc_fit_fun(x, a, t2star, f, phi, c):
            """
            Decay cosine fit function
            """
            return a * np.exp(-x / t2star) * np.cos(2 * np.pi * f * x + phi) + c

        xdata, ydata, sigma = process_curve_data(
            experiment_data._data, lambda datum: level2_probability(datum, "1")
            )
        result = curve_fit(
            osc_fit_fun, xdata, ydata, p0=list(p0.values()), sigma=sigma,
            bounds=bounds)

        if plot:
            ax = plot_curve_fit(osc_fit_fun, result, ax=ax)
            ax = plot_scatter(xdata, ydata, ax=ax)
            #ax = plot_errorbar(xdata, ydata, sigma, ax=ax)
            result.plt = plt
            plt.show()

        return result, None

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
            nosc: number of oscillations to induce using the phase gate
            experiment_type: String indicating the experiment type. Can be 'RamseyExperiment' or 'T2StarExperiment'.

        Raises:
            QiskitError: ?
        """

        self._qubit = qubit
        self._delays = delays
        self._unit = unit
        self._osc_freq = osc_freq
        #: str = "T2StarExperiment"
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
        """
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
            }

            circuits.append(circ)

        return circuits

    def T2Star_default_params(self, A=None, t2star=None, osc_freq=None, phi=None, B=None) -> Tuple[List[float], Tuple[List[float]]]:
        """
        Default fit parameters for oscillation data
        Args:
            qubit: the qubit index
        Returns:
            Fit guessed parameters
        """
        if A is None:
            A = 0.5
        if t2star is None:
            t2star = np.mean(self._delays)
        if osc_freq is None:
            f = self._osc_freq
        else:
            f = osc_freq
        print(">>> osc_freq = " + str(osc_freq))
        if phi is None:
            phi = 0.0
        if B is None:
            B = 0.5
        p0 = {'amplitude':A, 't2star':t2star, 'f_guess':f, 'phi_guess':phi, 'B_guess':B}
        A_bounds = [-0.5, 1.5]
        t2star_bounds = [0, np.inf]
        f_bounds = [0.5 * f, 1.5 * f]
        phi_bounds = [0, 2 * np.pi]
        B_bounds = [-0.5, 1.5]
        bounds=([A_bounds[0], t2star_bounds[0], f_bounds[0], phi_bounds[0], B_bounds[0]],
                [A_bounds[1], t2star_bounds[1], f_bounds[1], phi_bounds[1], B_bounds[1]])
        return p0, bounds



