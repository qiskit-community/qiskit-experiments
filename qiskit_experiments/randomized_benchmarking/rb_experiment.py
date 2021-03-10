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
from qiskit import QuantumCircuit
from .base_rb_generator import RBGeneratorBase
from .base_rb_analysis import RBAnalysisBase, RBAnalysisResultBase
from .base_rb_experiment import RBExperimentBase

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

class RBAnalysis(RBAnalysisBase):
    """Analysis class for standard RB experiments"""
    __analysis_result_class__ = RBAnalysisResult

    def fit(self, experiment_data):
        """Computes the RB fit for the given data"""
        num_qubits, lengths, group_type = self.get_experiment_params(experiment_data)
        xdata = self.organize_data(experiment_data._data)
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
            #analysis = CNOTDihedralRBAnalysis(qubits, lengths)
            analysis = None

        super().__init__(generator=generator)