# This code is part of Qiskit.
#
# (C) Copyright IBM 2019-2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
"""
Randomized benchmarking analysis classes
"""
# pylint: disable=no-name-in-module,import-error

from abc import abstractmethod
from typing import Dict, Optional, List, Callable, Tuple
import numpy as np
from scipy.optimize import curve_fit
from qiskit.result import Counts
from ..base_analysis import BaseAnalysis
from ..experiment_data import AnalysisResult

try:
    from matplotlib import pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


def build_counts_dict_from_list(count_list):
    """
    Add dictionary counts together.

    Parameters:
        count_list (list): List of counts.

    Returns:
        dict: Dict of counts.

    """
    if len(count_list) == 1:
        return count_list[0]

    new_count_dict = {}
    for countdict in count_list:
        for item in countdict:
            new_count_dict[item] = countdict[item]+new_count_dict.get(item, 0)
    return new_count_dict


class RBAnalysisResultBase(AnalysisResult):
    """Base class for results from RB experiments"""

    @abstractmethod
    def plot_all_data_series(self, ax):
        """Plots all data series of the RB; meant to be overridden by derived classes"""

    @abstractmethod
    def plot_label(self) -> str:
        """The label to be added as a top-right box in the plot"""

    def plot_y_axis_label(self) -> str:
        """Returns the string to be used as the plot's y label"""
        return "Ground State Population"

    def plot(self, ax=None, add_label=True, show_plt=True):
        """Plot randomized benchmarking data of a single pattern.

        Args:
            ax (Axes): plot axis (if passed in).
            add_label (bool): Add an EPC label.
            show_plt (bool): display the plot.

        Raises:
            ImportError: if matplotlib is not installed.
        """

        if not HAS_MATPLOTLIB:
            raise ImportError('The function plot_rb_data needs matplotlib. '
                              'Run "pip install matplotlib" before.')

        if ax is None:
            plt.figure()
            ax = plt.gca()

        self.plot_all_data_series(ax)

        ax.tick_params(labelsize=14)

        ax.set_xlabel('{} Length'.format(self.group_type_name()), fontsize=16)
        ax.set_ylabel(self.plot_y_axis_label(), fontsize=16)
        ax.grid(True)

        if add_label:
            bbox_props = dict(boxstyle="round,pad=0.3",
                              fc="white", ec="black", lw=2)

            ax.text(0.6, 0.9, self.plot_label(),
                    ha="center", va="center", size=14,
                    bbox=bbox_props, transform=ax.transAxes)

        if show_plt:
            plt.show()

    def group_type_name(self) -> str:
        """Returns "Clifford" or "CNOT_Dihedral" based on the underlying group"""
        names_dict = {
            'clifford': 'Clifford',
            'cnot_dihedral': 'CNOT-Dihedral'
        }
        return names_dict[self['group_type']]


class RBAnalysisBase(BaseAnalysis):
    """Base analysis class for randomized benchmarking experiments"""

    __analysis_result_class__ = RBAnalysisResultBase

    def _run_analysis(self, experiment_data, **options):
        """Run analysis on circuit data.

        Args:
            experiment_data (ExperimentData): the experiment data to analyze.
            options: kwarg options for analysis function.

        Returns:
            tuple: A pair ``(analysis_results, figures)`` where
                   ``analysis_results`` may be a single or list of
                   AnalysisResult objects, and ``figures`` may be
                   None, a single figure, or a list of figures.
        """
        fit_results = self.fit(experiment_data)
        results = self.__analysis_result_class__(fit_results)
        return (results, None)

    @abstractmethod
    def fit(self, experiment_data: List) -> Dict:
        """Fits the experimental data and returns a dictionary of fit result data"""

    def compute_prob(self, counts: Counts) -> float:
        """Computes the probability of getting the ground result
        Args:
            counts: The count dictionary
        Returns:
            The probability of the ground ("0") result from the counts dictionary
        """
        prob = 0
        if len(counts) > 0:
            n = len(list(counts)[0])
            ground_result = '0' * n
            if ground_result in counts:
                prob = counts[ground_result] / sum(counts.values())
        return prob

    @staticmethod
    def _rb_fit_fun(x: float, a: float, alpha: float, b: float) -> float:
        """The function used to fit RB: :math:`A\alpha^x + B`
            Args:
                x: The input to the function's variable
                a: The A parameter of the function
                alpha: The :math:`\alpha` parameter of the function
                b: The B parameter of the function
            Returns:
                The functions value on the specified parameters and input
        """
        # pylint: disable=invalid-name
        return a * alpha ** x + b

    def get_experiment_params(self, experiment_data):
        """Extracts relevant parameters of the experiment from the data object"""
        num_qubits = experiment_data._experiment.num_qubits()
        lengths = experiment_data._experiment.lengths()
        group_type = experiment_data._experiment.group_type()
        return (num_qubits, lengths, group_type)

    def collect_data(self,
                     data: List,
                     key_fn: Callable[[Dict[str, any]], any],
                     conversion_fn: Optional[Callable[[Counts], any]] = None
                     ) -> Dict:
        """
        Args:
            data: List of formatted data
            key_fn: Function acting on the metadata to produce a key used to identify counts
            that should be counted together.
            conversion_fn: A function to be applied to the counts after the combined count
            object is obtained (e.g. computing ground-state probability)
        Returns:
            The list of collected data elements, after merge and conversion.
        """
        result = {}
        for d in data:
            key = key_fn(d['metadata'])
            if key not in result:
                result[key] = []
            result[key].append(d['counts'])

        for key in result:
            result[key] = build_counts_dict_from_list(result[key])
            if conversion_fn is not None:
                result[key] = conversion_fn(result[key])

        return result

    def organize_data(self, data: List) -> np.array:
        """Converts the data to a list of probabilities for each seed
            Args:
                data: The counts data
            Returns:
                a list [seed_0_probs, seed_1_probs...] where seed_i_prob is
                a list of the probabilities for seed i for every length
        """
        seeds = sorted(list({d['metadata']['seed'] for d in data}))
        length_indices = sorted(list({d['metadata']['length_index'] for d in data}))
        prob_dict = self.collect_data(data,
                                      key_fn=lambda m: (m['seed'], m['length_index']),
                                      conversion_fn=self.compute_prob)
        return np.array([[prob_dict[(seed, length_index)]
                          for length_index in length_indices]
                         for seed in seeds])

    def calc_statistics(self, xdata: np.array) -> Dict[str, List[float]]:
        """Computes the mean and standard deviation of the probability data
        Args:
            xdata: List of lists of probabilities (for each seed and length)
        Returns:
            A dictionary {'mean': m, 'std': s} for the mean and standard deviation
            Standard deviation is computed only for more than 1 seed
        """
        ydata = {}
        ydata['mean'] = np.mean(xdata, 0)
        ydata['std'] = None
        if xdata.shape[0] != 1:  # more than 1 seed
            ydata['std'] = np.std(xdata, 0)
        return ydata

    def generate_fit_guess(self, mean: np.array,
                           num_qubits: int,
                           lengths: List[int]) -> Tuple[float]:
        """Generate initial guess for the fitter from the mean data
            Args:
                mean: A list of mean probabilities for each length
                num_qubits: The number of qubits in the RB experiment
                lengths: The lengths of the RB sequences
            Returns:
                The initial guess for the fit parameters (A, alpha, B)
        """
        # pylint: disable=invalid-name
        fit_guess = [0.95, 0.99, 1 / 2 ** num_qubits]
        # Use the first two points to guess the decay param
        y0 = mean[0]
        y1 = mean[1]
        dcliff = (lengths[1] - lengths[0])
        dy = ((y1 - fit_guess[2]) / (y0 - fit_guess[2]))
        alpha_guess = dy ** (1 / dcliff)
        if alpha_guess < 1.0:
            fit_guess[1] = alpha_guess

        if y0 > fit_guess[2]:
            fit_guess[0] = ((y0 - fit_guess[2]) /
                            fit_guess[1] ** lengths[0])

        return tuple(fit_guess)

    def run_curve_fit(self,
                      ydata: Dict[str, np.array],
                      fit_guess: Tuple[float],
                      lengths: List[int],
                      ) -> Tuple[Tuple[float], Tuple[float]]:
        """Runs the curve fir algorithm from the initial guess and based on the statistical data
            Args:
                ydata: The statistical data
                fit_guess: The initial guess
                lengths: The lengths of the RB sequences
            Returns:
                The resulting fit data
        """
        # if at least one of the std values is zero, then sigma is replaced
        # by None
        if ydata['std'] is None or 0 in ydata['std']:
            sigma = None
        else:
            sigma = ydata['std'].copy()

        params, pcov = curve_fit(self._rb_fit_fun, lengths,
                                 ydata['mean'],
                                 sigma=sigma,
                                 p0=fit_guess,
                                 bounds=([0, 0, 0], [1, 1, 1]))
        params_err = np.sqrt(np.diag(pcov))
        return (params, params_err)
