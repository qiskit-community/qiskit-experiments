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

from typing import Dict, List, Union, Optional, Tuple
from itertools import product
from copy import copy
from numpy.random import RandomState
import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info.analysis.average import average_data
from qiskit.result import Counts
from .base_rb_generator import RBGeneratorBase
from .base_rb_analysis import RBAnalysisBase
from .base_rb_experiment import RBExperimentBase
from .rb_experiment import RBAnalysisResult


class PurityRBGenerator(RBGeneratorBase):
    """Generates circuits for purity RB

    In purity RB, circuits are generated as in standard RB, but then for each circuit we create
    3^n copies, where n is the number of measured qubits. The copies correspond to the
    3^n possible measurement operators constructed from Z, X, Y; for each circuit
    we add corresponding rotations to the qubits just before the measurement gates
    """
    def __init__(self,
                 nseeds: int = 1,
                 qubits: List[int] = (0,),
                 lengths: List[int] = (1, 10, 20),
                 rand_seed: Optional[Union[int, RandomState]] = None,
                 ):
        """Initialize the circuit generator for a purity randomized banchmarking experiment.
                Args:
                    nseeds: number of different seeds (random circuits) to generate
                    qubits: the qubits particiapting in the experiment
                    lengths: for each seed, the lengths of the circuits used for that seed.
                    rand_seed: optional random number seed
        """
        super().__init__(nseeds,
                         qubits,
                         lengths,
                         rand_seed=rand_seed)
        self.generate_circuits()

    def generate_circuits_for_seed(self) -> List[Dict]:
        """Generates the purity RB circuits for a single seed
            Returns:
                The list of {'circuit': c, 'metadata': m} pairs
            Additional information:
                Purity RB creates new circuits with the corresponding purity measurements
                based on standard RB circuits
        """
        circuits = super().generate_circuits_for_seed()
        # each standard circuit gives rise to 3**qubits circuits
        # with corresponding pre-measure operators
        new_circuits = []
        for c in circuits:
            new_circuits += self.add_purity_measurements(c)
        self.add_extra_meta(new_circuits, {
            'experiment_type': PurityRBExperiment.__name__,
        })
        return new_circuits

    def add_purity_measurements(self, circuit: QuantumCircuit) -> List[QuantumCircuit]:
        """Add all combinations of purity measurement to the given circuit
        Args:
            circuit: The circuit to add purity measurement to
        Returns:
            The list of new circuits with the purity measurements
        """
        meas_op_names = ['Z', 'X', 'Y']
        result = []
        for meas_ops in product(meas_op_names, repeat=self.num_meas_qubits()):
            new_meta = copy(circuit.metadata)
            new_meta['purity_meas_ops'] = "".join(meas_ops)
            new_circuit = QuantumCircuit(circuit.qregs[0], circuit.cregs[0])
            new_circuit.metadata = new_meta
            new_circuit += circuit
            for qubit_index, meas_op in enumerate(meas_ops):
                qubit = self._meas_qubits[qubit_index]
                if meas_op == 'Z':
                    pass  # do nothing
                if meas_op == 'X':
                    new_circuit.rx(np.pi / 2, qubit)
                if meas_op == 'Y':
                    new_circuit.ry(np.pi / 2, qubit)
            result.append(new_circuit)
        return result

    def circuit_type_string(self, meta: Dict[str, any]) -> str:
        """Adds the "purity" label to the circuit type with the additional data
            of the specific measurement ops used for this circuit
            Args:
                meta: The metadata of the circuit
            Returns:
                The type string for the circuit
        """
        return "purity_{}".format(meta['purity_meas_ops'])


class PurityRBAnalysisResult(RBAnalysisResult):
    """Class for purity RB analysis results"""
    def plot_all_data_series(self, ax):
        """Plots the purity RB data series"""
        self.plot_data_series(ax, error_bar=True)

    def plot_label(self):
        """Plots the purity RB fit results"""
        return "alpha: %.3f(%.1e) PEPC: %.3e(%.1e)" % (self['alpha_pur'],
                                                       self['alpha_pur_err'],
                                                       self['pepc'],
                                                       self['pepc_err'])

    def plot_y_axis_label(self) -> str:
        """Plots the y label for purity rB results"""
        return "Trace of Rho Square"


class PurityRBAnalysis(RBAnalysisBase):
    """Analysis class for purity RB experiments"""
    __analysis_result_class__ = PurityRBAnalysisResult

    def __init__(self):
        self.num_qubits = None
        self._zdict_ops = None

    def add_zdict_ops(self):
        """Creating all Z-correlators
        in order to compute the expectation values."""
        self._zdict_ops = []
        statedict = {("{0:0%db}" % self.num_qubits).format(i): 1 for i in
                     range(2 ** self.num_qubits)}

        for i in range(2 ** self.num_qubits):
            self._zdict_ops.append(statedict.copy())
            for j in range(2 ** self.num_qubits):
                if bin(i & j).count('1') % 2 != 0:
                    self._zdict_ops[-1][("{0:0%db}"
                                         % self.num_qubits).format(j)] = -1

    @staticmethod
    def F234(n: int, a: int, b: int) -> int:
        """Function that maps:
        2^n x 3^n --> 4^n ,
        namely:
        (a,b) --> c where
        a in 2^n, b in 3^n, c in 4^n
        """
        # pylint: disable=invalid-name
        # 0 <--> I
        # 1 <--> X
        # 2 <--> Y
        # 3 <--> Z
        LUT = [[0, 0, 0], [3, 1, 2]]

        # compute bits
        aseq = []
        bseq = []

        aa = a
        bb = b
        for i in range(n):
            aseq.append(np.mod(aa, 2))
            bseq.append(np.mod(bb, 3))
            aa = np.floor_divide(aa, 2)
            bb = np.floor_divide(bb, 3)

        c = 0
        for i in range(n):
            c += (4 ** i) * LUT[aseq[i]][bseq[i]]

        return c

    def purity_op_key(self, op: str) -> int:
        """Key function to help sort the op array
        The order is: ZZ < XZ < YZ < ZX < XX < YX < ZY < XY < YY etc.
        """
        op_vals = {'Z': 0, 'X': 1, 'Y': 2}
        return sum([op_vals[o] * (3**k) for (k, o) in enumerate(op)])

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
        purity_ops = sorted(list({d['metadata']['purity_meas_ops'] for d in data}),
                            key=self.purity_op_key)
        shots_dict = self.collect_data(data,
                                       key_fn=lambda m: (m['seed'],
                                                         m['length_index'],
                                                         m['purity_meas_ops']))
        return np.array([[self.compute_purity_data(shots_dict, seed, length_index, purity_ops)
                          for length_index in length_indices]
                         for seed in seeds])

    def compute_purity_data(self,
                            shots_dict: Dict[Tuple, Counts],
                            seed: int,
                            length_index: int,
                            purity_ops: List[str]
                            ) -> float:
        """Computes the purity data from the shots dictionary for a given seed and length index
            Args:
                shots_dict: The shots dictionary for the experiment
                seed: The seed
                length_index: The length index
                purity_ops: A correctly-ordered list of the purity ops for the experiment
            Returns:
                The purity value corresponding to (seed, length_index)
        """
        corr_vec = [0] * (4 ** self.num_qubits)
        count_vec = [0] * (4 ** self.num_qubits)
        for i, purity_op in enumerate(purity_ops):
            # vector of the 4^n correlators and counts
            # calculating the vector of 4^n correlators
            for indcorr in range(2 ** self.num_qubits):
                zcorr = average_data(shots_dict[(seed, length_index, purity_op)],
                                     self._zdict_ops[indcorr])
                zind = self.F234(self.num_qubits, indcorr, i)

                corr_vec[zind] += zcorr
                count_vec[zind] += 1

        # calculating the purity
        purity = 0
        for idx, _ in enumerate(corr_vec):
            purity += (corr_vec[idx] / count_vec[idx]) ** 2
        purity = purity / (2 ** self.num_qubits)
        return purity

    def fit(self, experiment_data: List):
        """Computes the purity RB fit for the given data"""
        num_qubits, lengths, group_type = self.get_experiment_params(experiment_data)
        self.num_qubits = num_qubits
        self.add_zdict_ops()
        xdata = self.organize_data(experiment_data.data)
        ydata = self.calc_statistics(xdata)
        fit_guess = self.generate_fit_guess(ydata['mean'], num_qubits, lengths)
        params, params_err = self.run_curve_fit(ydata, fit_guess, lengths)

        alpha = params[1]  # exponent
        alpha_err = params_err[1]

        # Calculate alpha (=p):
        # fitting the curve: A*p^(2m)+B
        # where m is the Clifford length
        alpha_pur = np.sqrt(alpha)

        # calculate the error of alpha
        alpha_pur_err = alpha_err / (2 * np.sqrt(alpha_pur))

        # calculate purity error per clifford (pepc)
        nrb = 2 ** self.num_qubits
        pepc = (nrb - 1) / nrb * (1 - alpha_pur)
        pepc_err = (nrb - 1) / nrb * alpha_pur_err / alpha_pur

        return {
            'A': params[0],
            'alpha': params[1],
            'B': params[2],
            'A_err': params_err[0],
            'alpha_err': params_err[1],
            'B_err': params_err[2],
            'alpha_pur': alpha_pur,
            'alpha_pur_err': alpha_pur_err,
            'pepc': pepc,
            'pepc_err': pepc_err,
            'num_qubits': num_qubits,
            'xdata': xdata,
            'ydata': ydata,
            'lengths': lengths,
            'fit_function': self._rb_fit_fun,
            'group_type': group_type
        }


class PurityRBExperiment(RBExperimentBase):
    """Experiment class for purity RB experiment"""
    def __init__(self,
                 nseeds: int = 1,
                 qubits: List[int] = (0,),
                 lengths: List[int] = (1, 10, 20),
                 rand_seed: Optional[Union[int, RandomState]] = None,
                 ):
        """Initialize the purity RB experiment
            Args:
                nseeds: number of different seeds (random circuits) to generate
                qubits: the qubits particiapting in the experiment
                lengths: for each seed, the lengths of the circuits used for that seed.
                rand_seed: optional random number seed
        """
        generator = PurityRBGenerator(nseeds, qubits, lengths, rand_seed)
        self.__analysis_class__ = PurityRBAnalysis
        super().__init__(generator=generator)
