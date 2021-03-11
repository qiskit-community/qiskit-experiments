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

from typing import List, Union, Optional, Tuple
from numpy.random import RandomState
import numpy as np
from qiskit import QuantumCircuit
from .base_rb_generator import RBGeneratorBase
from .base_rb_analysis import RBAnalysisBase, RBAnalysisResultBase
from .base_rb_experiment import RBExperimentBase
from ..experiment_data import ExperimentData

class RBGenerator(RBGeneratorBase):
    """Generator class for standard RB experiments"""
    def __init__(self,
                 nseeds: int = 1,
                 qubits: List[int] = (0,),
                 lengths: List[int] = (1, 10, 20),
                 group_gates: Optional[str] = None,
                 rand_seed: Optional[Union[int, RandomState]] = None,
                 ):
        """Initialize the circuit generator for a standard randomized banchmarking experiment.
            Args:
                nseeds: number of different seeds (random circuits) to generate
                qubits: the qubits particiapting in the experiment
                lengths: for each seed, the lengths of the circuits used for that seed.
                group_gates: which group the circuits is based on
                rand_seed: optional random number seed
        """
        super().__init__(nseeds,
                         qubits,
                         lengths,
                         group_gates,
                         rand_seed,
                         name="randomized benchmarking")
        self.generate_circuits()

    def generate_circuits_for_seed(self) -> List[QuantumCircuit]:
        """Generates the standard RB circuits for a single seed
        Returns:
            The list of circuits.
        Additional information:
            For standard RB, the only addition is specifying "standard" in the metadata
        """
        circuits = super().generate_circuits_for_seed()
        self.add_extra_meta(circuits, {
            'experiment_type': RBExperiment.__name__,
        })
        return circuits

class RBAnalysisResult(RBAnalysisResultBase):
    def params(self) -> Tuple[float]:
        """Returns the parameters (A, alpha, B) of the fitting function"""
        return (self['A'], self['alpha'], self['B'])

    def plot_label(self) -> str:
        """Add the string to be used as the plot's header label"""
        return "alpha: %.3f(%.1e) EPC: %.3e(%.1e)" % (self['alpha'],
                                                      self['alpha_err'],
                                                      self['epc'],
                                                      self['epc_err'])

    def plot_data_series(self, ax, error_bar=False, color='blue', label=None):
        """Plots the RB data of a single series to a matplotlib axis"""
        for one_seed_data in self['xdata']:
            ax.plot(self['lengths'], one_seed_data, color=color, linestyle='none',
                    marker='x')
        if error_bar:
            ax.errorbar(self['lengths'], self['ydata']['mean'],
                        yerr=self['ydata']['std'],
                        color='red', linestyle='--', linewidth=3)
        ax.plot(self['lengths'],
                self['fit_function'](self['lengths'], *self.params()),
                color=color, linestyle='-', linewidth=2,
                label=label)

    def plot_all_data_series(self, ax):
        """Plots the data series of the RB"""
        self.plot_data_series(ax, error_bar=True)

class CNOTDihedralRBResult(RBAnalysisResultBase):
    """Class for cnot-dihedral RB analysis results"""
    def __init__(self, results):
        z_fit_result, x_fit_result, cnotdihedral_result = results
        self._z_fit_result = RBAnalysisResult(z_fit_result)
        self._x_fit_result = RBAnalysisResult(x_fit_result)
        super().__init__(cnotdihedral_result)

    def num_qubits(self) -> int:
        """Returns the number of qubits used in the RB experiment"""
        return self._z_fit_result.num_qubits()

    def plot_all_data_series(self, ax):
        """Plots the Z and X basis data series"""
        self._z_fit_result.plot_data_series(ax, color='blue', label='Measure state |0...0>')
        self._x_fit_result.plot_data_series(ax, color='red', label='Measure state |+...+>')
        ax.legend(loc='lower left')

    def plot_label(self):
        """Plots cnot-dihedral fit results"""
        return "alpha: %.3f(%.1e) EPG_est: %.3e(%.1e)" % (self['alpha'],
                                                          self['alpha_err'],
                                                          self['epg_est'],
                                                          self['epg_est_err'])

class RBAnalysis(RBAnalysisBase):
    """Analysis class for standard RB experiments"""
    __analysis_result_class__ = RBAnalysisResult

    def fit(self, experiment_data):
        """Computes the RB fit for the given data"""
        num_qubits, lengths, group_type = self.get_experiment_params(experiment_data)
        xdata = self.organize_data(experiment_data.data)
        ydata = self.calc_statistics(xdata)
        fit_guess = self.generate_fit_guess(ydata['mean'], num_qubits, lengths)
        params, params_err = self.run_curve_fit(ydata, fit_guess, lengths)

        alpha = params[1]  # exponent
        alpha_err = params_err[1]

        nrb = 2 ** num_qubits
        epc = (nrb - 1) / nrb * (1 - alpha)
        epc_err = (nrb - 1) / nrb * alpha_err / alpha

        return {
            'A': params[0],
            'alpha': params[1],
            'B': params[2],
            'A_err': params_err[0],
            'alpha_err': params_err[1],
            'B_err': params_err[2],
            'epc': epc,
            'epc_err': epc_err,
            'xdata': xdata,
            'ydata': ydata,
            'num_qubits': num_qubits,
            'lengths': lengths,
            'fit_function': self._rb_fit_fun,
            'group_type': group_type
        }

class CNOTDihedralRBAnalysis(RBAnalysisBase):
    """Analysis class for cnot-dihedral RB experiments"""
    __analysis_result_class__ = CNOTDihedralRBResult

    def split_experiment_data(self, experiment_data):
        z_data = ExperimentData(experiment_data.experiment())
        x_data = ExperimentData(experiment_data.experiment())
        for d in experiment_data.data:
            if d['metadata']['cnot_basis'] == 'Z':
                z_data.data.append(d)
            if d['metadata']['cnot_basis'] == 'X':
                x_data.data.append(d)
        return (z_data, x_data)

    def fit(self, experiment_data) -> CNOTDihedralRBResult:
        """Computes the cnot-dihedral fit from the results of the Z, X basis fits
            Args:
                experiment_data: The data from the experiment
            Returns:
                The cnot-dihedral result (which contains the Z, X results)
        """
        # pylint: disable=invalid-name

        num_qubits, lengths, group_type = self.get_experiment_params(experiment_data)
        (z_data, x_data) = self.split_experiment_data(experiment_data)
        z_fit_results = RBAnalysis().fit(z_data)
        x_fit_results = RBAnalysis().fit(x_data)

        # calculate nrb=d=2^n:
        nrb = 2 ** num_qubits

        # Calculate alpha_Z and alpha_R:
        alpha_Z = z_fit_results['alpha']
        alpha_R = x_fit_results['alpha']
        # Calculate their errors:
        alpha_Z_err = z_fit_results['alpha_err']
        alpha_R_err = x_fit_results['alpha_err']

        # Calculate alpha:
        alpha = (alpha_Z + nrb * alpha_R) / (nrb + 1)

        # Calculate alpha_err:
        alpha_Z_err_sq = (alpha_Z_err / alpha_Z / (nrb + 1)) ** 2
        alpha_R_err_sq = (nrb * alpha_R_err / alpha_R / (nrb + 1)) ** 2
        alpha_err = np.sqrt(alpha_Z_err_sq + alpha_R_err_sq)

        # Calculate epg_est:
        epg_est = (nrb - 1) * (1 - alpha) / nrb

        # Calculate epg_est_error
        epg_est_err = (nrb - 1) / nrb * alpha_err / alpha

        cnotdihedral_result = {'alpha': alpha,
                               'alpha_err': alpha_err,
                               'epg_est': epg_est,
                               'epg_est_err': epg_est_err,
                               'lengths': lengths,
                               'num_qubits': num_qubits,
                               'group_type': group_type}

        return (z_fit_results, x_fit_results, cnotdihedral_result)

class RBExperiment(RBExperimentBase):
    """Experiment class for standard RB experiment"""
    def __init__(self,
                 nseeds: int = 1,
                 qubits: List[int] = (0,),
                 lengths: List[int] = (1, 10, 20),
                 group_gates: Optional[str] = None,
                 rand_seed: Optional[Union[int, RandomState]] = None,
                 ):
        """Initialize the RB experiment
            Args:
                nseeds: number of different seeds (random circuits) to generate
                qubits: the qubits particiapting in the experiment
                lengths: for each seed, the lengths of the circuits used for that seed.
                group_gates: which group the circuits is based on
                rand_seed: optional random number seed
        """
        generator = RBGenerator(nseeds, qubits, lengths, group_gates, rand_seed)
        if generator.rb_group_type() == 'clifford':
            self.__analysis_class__ = RBAnalysis
        if generator.rb_group_type() == 'cnot_dihedral':
            self.__analysis_class__ = CNOTDihedralRBAnalysis

        super().__init__(generator=generator)