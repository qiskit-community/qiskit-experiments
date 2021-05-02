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

from qiskit_experiments.base_analysis import BaseAnalysis

class QVAnalysis(BaseAnalysis):
    """RB Analysis class."""

    # pylint: disable = arguments-differ
    def _run_analysis(self, experiment_data):
        """Run analysis on circuit data.
        Args:
            experiment_data (ExperimentData): the experiment data to analyze.
        Returns:
            tuple: A pair ``(analysis_result, figures)`` where
                   ``analysis_results`` may be a single or list of
                   AnalysisResult objects, and ``figures`` may be
                   None, a single figure, or a list of figures.
        """
        depth = experiment_data.experiment.num_qubits
        num_trials = experiment_data.experiment.trials

        heavy_outputs = np.zeros((depth, num_trials), dtype=float)
        heavy_output_prob_exp = np.zeros((depth, num_trials), dtype=float)

        # analyse ideal data to calculate all heavy outputs
        for ideal_data in experiment_data.data:
            if not ideal_data["metadata"].get("is_simulation", None):
                continue
            depth = ideal_data["metadata"]["depth"]
            depthidx = depth - 1 # depth is starting from 1
            trial = ideal_data["metadata"]["trial"]
            trialidx = trial - 1  # trial is starting from 1

            heavy_outputs[depthidx, trialidx] = self._calc_ideal_heavy_output(ideal_data)

        # analyse non-ideal data
        for data in experiment_data.data:
            if data["metadata"].get("is_simulation", None):
                continue
            depth = data["metadata"]["depth"]
            depthidx = depth - 1  # depth is starting from 1
            trial = data["metadata"]["trial"]
            trialidx = trial - 1  # trial is starting from 1

            heavy_output_prob_exp[depthidx, trialidx] = \
                self._calc_exp_heavy_output_probability(data, heavy_outputs[depthidx, trialidx])

        return self._calc_quantum_volume(heavy_output_prob_exp, depth, num_trials)

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

    def _calc_quantum_volume(self, heavy_output_prob_exp, last_depth, trials):
        """
        calc the quantum volume of the analysed system.
        quantum volume is determined by the largest successful depth.
        A depth is successful if it has 'mean heavy-output probability' > 2/3 with confidence
        level > 0.977 (corresponding to z_value = 2).

        Returns:
            int: quantum volume
        """
        success_list = []
        confidence_level_threshold = self._calc_confidence_level(z_value=2)
        means, sigmas = self._calc_statistics(heavy_output_prob_exp, last_depth, trials)

        max_success_depth = 0
        for depthidx, depth in enumerate(range(1, last_depth + 1)):
            z_value = self._calc_z_value(means[depthidx], sigmas[depthidx])
            confidence_level = self._calc_confidence_level(z_value)

            if means[depthidx] > 2 / 3 and confidence_level > confidence_level_threshold:
                success_list.append(True)
                max_success_depth = depth
            else:
                success_list.append(False)
        return 2 ** max_success_depth

    @staticmethod
    def _calc_statistics(heavy_output_prob, depth, trials):
        """
        Convert the heavy outputs in the different trials into mean and error for plotting.

        Here we assume the error is due to a binomial distribution.
        Error (standard deviation) for binomial distribution is sqrt(np(1-p)),
        where n is the number of trials and p is the success probability.
        """
        mean_hop = np.zeros(depth, dtype=float)
        sigma_hop = np.zeros(depth, dtype=float)

        for depthidx, depth in enumerate(range(1, depth + 1)):
            mean_hop[depthidx] = np.mean(heavy_output_prob[depthidx, :])
            sigma_hop[depthidx] = mean_hop[depthidx] * ((1.0 - mean_hop[depthidx]) / trials) ** 0.5

        return mean_hop, sigma_hop
