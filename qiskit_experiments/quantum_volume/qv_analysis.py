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
Quantum Volume analysis class.
"""

import math
import warnings
import numpy as np
from typing import Optional

from qiskit_experiments.base_analysis import BaseAnalysis
from qiskit_experiments.base_analysis import AnalysisResult
from qiskit_experiments.analysis.plotting import plot_curve_fit, plot_scatter, plot_errorbar

try:
    from matplotlib import pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

class QVAnalysis(BaseAnalysis):
    """RB Analysis class."""

    # pylint: disable = arguments-differ
    def _run_analysis(self,
                      experiment_data,
                      plot: bool = True,
                      ax: Optional["AxesSubplot"] = None
                      ):
        """Run analysis on circuit data.
        Args:
            experiment_data (ExperimentData): the experiment data to analyze.
            plot: If True generate a plot of fitted data.
            ax: Optional, matplotlib axis to add plot to.
        Returns:
            tuple: A pair ``(analysis_result, figures)`` where
                   ``analysis_results`` may be a single or list of
                   AnalysisResult objects, and ``figures`` may be
                   None, a single figure, or a list of figures.
        """
        depth = experiment_data.experiment().num_qubits
        num_trials = experiment_data.experiment().trials

        heavy_outputs = np.zeros(num_trials, dtype=list)
        heavy_output_prob_exp = np.zeros(num_trials, dtype=list)

        # analyse ideal data to calculate all heavy outputs
        # must calculate first the ideal data, because the non-ideal calculation uses it
        for ideal_data in experiment_data.data:
            if not ideal_data["metadata"].get("is_simulation", None):
                continue
            trial = ideal_data["metadata"]["trial"]
            trial_index = trial - 1 # trials starts from 1, so as index use trials - 1

            heavy_outputs[trial_index] = self._calc_ideal_heavy_output(ideal_data)

        # analyse non-ideal data
        for data in experiment_data.data:
            if data["metadata"].get("is_simulation", None):
                continue
            trial = data["metadata"]["trial"]
            trial_index = trial - 1  # trials starts from 1, so as index use trials - 1

            heavy_output_prob_exp[trial_index] = \
                self._calc_exp_heavy_output_probability(data, heavy_outputs[trial_index])

        analysis_result = AnalysisResult(
            self._calc_quantum_volume(heavy_output_prob_exp, depth, num_trials))

        if plot:
            ax = self._format_plot(ax, analysis_result)
            if HAS_MATPLOTLIB:
                analysis_result.plt = plt
        return analysis_result, None

    @staticmethod
    def _calc_ideal_heavy_output(ideal_data):
        """
        calculate the bit strings of the heavy output for the ideal simulation
        Args:
            ideal_data (dict): the simulation result of the ideal circuit
        Returns:
             list: the bit strings of the heavy output
        """
        depth = ideal_data["metadata"]["depth"]
        probabilities_vector = ideal_data.get('probabilities')

        format_spec = "{0:0%db}" % depth
        # keys are bit strings and values are probabilities of observing those strings
        all_output_prob_ideal = \
            {format_spec.format(b):
                 float(np.real(probabilities_vector[b])) for b in range(2 ** depth)}

        median_probabilities = float(np.real(np.median(probabilities_vector)))
        heavy_strings = list(filter(lambda x: all_output_prob_ideal[x] > median_probabilities,
                                    list(all_output_prob_ideal.keys())))
        return heavy_strings

    @staticmethod
    def _calc_exp_heavy_output_probability(data, heavy_outputs):
        """
        calculate the probability of measuring heavy output string in the data
        Args:
            data (dict): the result of the circuit exectution
            heavy_outputs (list): the bit strings of the heavy output from the ideal simulation
        Returns:
            int: heavy output probability
        """
        circ_shots = sum(data["counts"].values())

        # calculate the number of heavy output counts in the experiment
        heavy_output_counts = sum([data["counts"].get(value, 0) for value in heavy_outputs])

        # calculate the experimental heavy output probability
        return heavy_output_counts / circ_shots

    @staticmethod
    def _calc_z_value(mean, sigma):
        """Calculate z value using mean and sigma.

        Args:
            mean (float): mean
            sigma (float): standard deviation

        Returns:
            float: z_value in standard normal distibution.
        """

        if sigma == 0:
            # assign a small value for sigma if sigma = 0
            sigma = 1e-10
            warnings.warn('Standard deviation sigma should not be zero.')

        z_value = (mean - 2 / 3) / sigma

        return z_value

    @staticmethod
    def _calc_confidence_level(z_value):
        """Calculate confidence level using z value.

        Accumulative probability for standard normal distribution
        in [-z, +infinity] is 1/2 (1 + erf(z/sqrt(2))),
        where z = (X - mu)/sigma = (hmean - 2/3)/sigma

        Args:
            z_value (float): z value in in standard normal distibution.

        Returns:
            float: confidence level in decimal (not percentage).
        """

        confidence_level = 0.5 * (1 + math.erf(z_value / 2 ** 0.5))

        return confidence_level

    def _calc_quantum_volume(self, heavy_output_prob_exp, depth, trials):
        """
        calc the quantum volume of the analysed system.
        quantum volume is determined by the largest successful depth.
        A depth is successful if it has 'mean heavy-output probability' > 2/3 with confidence
        level > 0.977 (corresponding to z_value = 2).
        we assume the error (standard deviation) of the heavy output probability is due to a
        binomial distribution. standard deviation for binomial distribution is sqrt(np(1-p)),
        where n is the number of trials and p is the success probability.

        Returns:
            dict: quantum volume calculations -
            the quantum volume,
            whether the results passed the threshold,
            the confidence fof the result,
            the heavy output probability for each trial,
            the mean heavy output probability,
            the error of the heavy output probability,
            the depth of the circuit,
            the number of trials ran
        """
        quantum_volume = 0
        success = False
        confidence_level_threshold = self._calc_confidence_level(z_value=2)
        mean_hop = np.mean(heavy_output_prob_exp)
        sigma_hop = mean_hop * ((1.0 - mean_hop) / trials) ** 0.5

        z_value = self._calc_z_value(mean_hop, sigma_hop)
        confidence_level = self._calc_confidence_level(z_value)
        if mean_hop > 2 / 3 and confidence_level > confidence_level_threshold:
            quantum_volume = 2 ** depth
            success = True

        result = {
            "quantum volume": quantum_volume,
            "success": success,
            "confidence list": confidence_level,
            "heavy output probability": heavy_output_prob_exp,
            "mean hop": mean_hop,
            "sigma": sigma_hop,
            "depth": depth,
            "trials": trials
        }
        return result

    @staticmethod
    def _format_plot(ax, analysis_result):
        """
        format the QV plot
        Args:
            ax: matplotlib axis to add plot to.
            analysis_result: the results of the experimnt
        Returns:
            AxesSubPlot: the matplotlib axes containing the plot.
        """
        trial_list = np.arange(1, analysis_result["trials"] + 1)  # x data

        hop_accumulative = np.cumsum(analysis_result["heavy output probability"]) / trial_list
        two_sigma = 2 * (hop_accumulative * (1 - hop_accumulative) / trial_list) ** 0.5

        # plot inidivual HOP as scatter
        ax = plot_scatter(trial_list, analysis_result["heavy output probability"], ax=ax,
                          s=3, zorder=3, label='Individual HOP')
        # plot accumulative HOP
        ax.plot(trial_list, hop_accumulative, color='r', label='Cumulative HOP')
        # plot two-sigma shaded area
        ax = plot_errorbar(trial_list, hop_accumulative, two_sigma, ax=ax,
                           fmt="none", ecolor='lightgray', elinewidth=20, capsize=0,
                           alpha=0.5, label='2$\\sigma$')
        # plot 2/3 success threshold
        ax.axhline(2 / 3, color='k', linestyle='dashed', linewidth=1, label='Threshold')

        ax.set_xlim(1, analysis_result["trials"] + 1)
        ax.set_ylim(hop_accumulative[-1] - 4 * two_sigma[-1],
                    hop_accumulative[-1] + 4 * two_sigma[-1])

        ax.set_xlabel('Number of Trials', fontsize=14)
        ax.set_ylabel('Heavy Output Probability', fontsize=14)

        ax.set_title('Quantum Volume ' + str(2 ** analysis_result["depth"]) +
                     ' - accumulative hop', fontsize = 14)

        # re-arrange legend order
        handles, labels = ax.get_legend_handles_labels()
        handles = [handles[1], handles[2], handles[0], handles[3]]
        labels = [labels[1], labels[2], labels[0], labels[3]]
        ax.legend(handles, labels)
        return ax
