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
from typing import Optional

import numpy as np
import uncertainties
from qiskit_experiments.exceptions import AnalysisError
from qiskit_experiments.curve_analysis import plot_scatter, plot_errorbar
from qiskit_experiments.framework import (
    BaseAnalysis,
    AnalysisResultData,
    Options,
)


class QuantumVolumeAnalysis(BaseAnalysis):
    r"""A class to analyze quantum volume experiments.

    # section: overview
        Calculate the quantum volume of the analysed system.
        The quantum volume is determined by the largest successful circuit depth.
        A depth is successful if it has 'mean heavy-output probability' > 2/3 with confidence
        level > 0.977 (corresponding to z_value = 2), and at least 100 trials have been ran.
        we assume the error (standard deviation) of the heavy output probability is due to a
        binomial distribution. The standard deviation for binomial distribution is
        :math:`\sqrt{(np(1-p))}`, where :math:`n` is the number of trials and :math:`p`
        is the success probability.
    """

    @classmethod
    def _default_options(cls) -> Options:
        """Return default analysis options.

        Analysis Options:
            plot (bool): Set ``True`` to create figure for fit result.
            ax(AxesSubplot): Optional. A matplotlib axis object to draw.
        """
        options = super()._default_options()
        options.plot = True
        options.ax = None
        return options

    def _run_analysis(self, experiment_data):
        data = experiment_data.data()
        num_trials = len(data)
        depth = None
        heavy_output_prob_exp = []

        for data_trial in data:
            trial_depth = data_trial["metadata"]["depth"]
            if depth is None:
                depth = trial_depth
            elif trial_depth != depth:
                raise AnalysisError("QuantumVolume circuits do not all have the same depth.")
            heavy_output = self._calc_ideal_heavy_output(
                data_trial["metadata"]["ideal_probabilities"], trial_depth
            )
            heavy_output_prob_exp.append(
                self._calc_exp_heavy_output_probability(data_trial, heavy_output)
            )

        hop_result, qv_result = self._calc_quantum_volume(heavy_output_prob_exp, depth, num_trials)

        if self.options.plot:
            ax = self._format_plot(hop_result, ax=self.options.ax)
            figures = [ax.get_figure()]
        else:
            figures = None
        return [hop_result, qv_result], figures

    @staticmethod
    def _calc_ideal_heavy_output(probabilities_vector, depth):
        """
        Calculate the bit strings of the heavy output for the ideal simulation

        Args:
            ideal_data (dict): the simulation result of the ideal circuit

        Returns:
             list: the bit strings of the heavy output
        """

        format_spec = "{0:0%db}" % depth
        # Keys are bit strings and values are probabilities of observing those strings
        all_output_prob_ideal = {
            format_spec.format(b): float(np.real(probabilities_vector[b]))
            for b in range(2**depth)
        }

        median_probabilities = float(np.real(np.median(probabilities_vector)))
        heavy_strings = list(
            filter(
                lambda x: all_output_prob_ideal[x] > median_probabilities,
                list(all_output_prob_ideal.keys()),
            )
        )
        return heavy_strings

    @staticmethod
    def _calc_exp_heavy_output_probability(data, heavy_outputs):
        """
        Calculate the probability of measuring heavy output string in the data

        Args:
            data (dict): the result of the circuit execution
            heavy_outputs (list): the bit strings of the heavy output from the ideal simulation

        Returns:
            int: heavy output probability
        """
        circ_shots = sum(data["counts"].values())

        # Calculate the number of heavy output counts in the experiment
        heavy_output_counts = sum([data["counts"].get(value, 0) for value in heavy_outputs])

        # Calculate the experimental heavy output probability
        return heavy_output_counts / circ_shots

    @staticmethod
    def _calc_z_value(mean, sigma):
        """Calculate z value using mean and sigma.

        Args:
            mean (float): mean
            sigma (float): standard deviation

        Returns:
            float: z_value in standard normal distribution.
        """

        if sigma == 0:
            # Assign a small value for sigma if sigma = 0
            sigma = 1e-10
            warnings.warn("Standard deviation sigma should not be zero.")

        z_value = (mean - 2 / 3) / sigma

        return z_value

    @staticmethod
    def _calc_confidence_level(z_value):
        """Calculate confidence level using z value.

        Accumulative probability for standard normal distribution
        in [-z, +infinity] is 1/2 (1 + erf(z/sqrt(2))),
        where z = (X - mu)/sigma = (hmean - 2/3)/sigma

        Args:
            z_value (float): z value in in standard normal distribution.

        Returns:
            float: confidence level in decimal (not percentage).
        """

        confidence_level = 0.5 * (1 + math.erf(z_value / 2**0.5))

        return confidence_level

    def _calc_quantum_volume(self, heavy_output_prob_exp, depth, trials):
        """
        Calc the quantum volume of the analysed system.
        quantum volume is determined by the largest successful depth.
        A depth is successful if it has 'mean heavy-output probability' > 2/3 with confidence
        level > 0.977 (corresponding to z_value = 2), and at least 100 trials have been ran.
        we assume the error (standard deviation) of the heavy output probability is due to a
        binomial distribution. standard deviation for binomial distribution is sqrt(np(1-p)),
        where n is the number of trials and p is the success probability.

        Returns:
            dict: quantum volume calculations -
            the quantum volume,
            whether the results passed the threshold,
            the confidence of the result,
            the heavy output probability for each trial,
            the mean heavy output probability,
            the error of the heavy output probability,
            the depth of the circuit,
            the number of trials ran
        """
        quantum_volume = 1
        success = False

        mean_hop = np.mean(heavy_output_prob_exp)
        sigma_hop = (mean_hop * ((1.0 - mean_hop) / trials)) ** 0.5
        z = 2
        threshold = 2 / 3 + z * sigma_hop
        z_value = self._calc_z_value(mean_hop, sigma_hop)
        confidence_level = self._calc_confidence_level(z_value)
        if confidence_level > 0.977:
            quality = "good"
        else:
            quality = "bad"

        # Must have at least 100 trials
        if trials < 100:
            warnings.warn("Must use at least 100 trials to consider Quantum Volume as successful.")

        if mean_hop > threshold and trials >= 100:
            quantum_volume = 2**depth
            success = True

        hop_result = AnalysisResultData(
            "mean_HOP",
            value=uncertainties.ufloat(nominal_value=mean_hop, std_dev=sigma_hop),
            quality=quality,
            extra={
                "HOPs": heavy_output_prob_exp,
                "two_sigma": 2 * sigma_hop,
                "depth": depth,
                "trials": trials,
            },
        )

        qv_result = AnalysisResultData(
            "quantum_volume",
            value=quantum_volume,
            quality=quality,
            extra={
                "success": success,
                "confidence": confidence_level,
                "depth": depth,
                "trials": trials,
            },
        )
        return hop_result, qv_result

    @staticmethod
    def _format_plot(
        hop_result: AnalysisResultData, ax: Optional["matplotlib.pyplot.AxesSubplot"] = None
    ):
        """Format the QV plot

        Args:
            hop_result: the heavy output probability analysis result.
            ax: matplotlib axis to add plot to.

        Returns:
            AxesSubPlot: the matplotlib axes containing the plot.
        """
        trials = hop_result.extra["trials"]
        heavy_probs = hop_result.extra["HOPs"]
        trial_list = np.arange(1, trials + 1)  # x data

        hop_accumulative = np.cumsum(heavy_probs) / trial_list
        two_sigma = 2 * (hop_accumulative * (1 - hop_accumulative) / trial_list) ** 0.5

        # Plot individual HOP as scatter
        ax = plot_scatter(
            trial_list,
            heavy_probs,
            ax=ax,
            s=3,
            zorder=3,
            label="Individual HOP",
        )
        # Plot accumulative HOP
        ax.plot(trial_list, hop_accumulative, color="r", label="Cumulative HOP")

        # Plot two-sigma shaded area
        ax = plot_errorbar(
            trial_list,
            hop_accumulative,
            two_sigma,
            ax=ax,
            fmt="none",
            ecolor="lightgray",
            elinewidth=20,
            capsize=0,
            alpha=0.5,
            label="2$\\sigma$",
        )
        # Plot 2/3 success threshold
        ax.axhline(2 / 3, color="k", linestyle="dashed", linewidth=1, label="Threshold")

        ax.set_ylim(
            max(hop_accumulative[-1] - 4 * two_sigma[-1], 0),
            min(hop_accumulative[-1] + 4 * two_sigma[-1], 1),
        )

        ax.set_xlabel("Number of Trials", fontsize=14)
        ax.set_ylabel("Heavy Output Probability", fontsize=14)

        ax.set_title(
            "Quantum Volume experiment for depth "
            + str(hop_result.extra["depth"])
            + " - accumulative hop",
            fontsize=14,
        )

        # Re-arrange legend order
        handles, labels = ax.get_legend_handles_labels()
        handles = [handles[1], handles[2], handles[0], handles[3]]
        labels = [labels[1], labels[2], labels[0], labels[3]]
        ax.legend(handles, labels)
        return ax
