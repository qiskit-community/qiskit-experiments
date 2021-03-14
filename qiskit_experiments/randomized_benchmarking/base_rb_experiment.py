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
from typing import List, Optional
from .base_rb_generator import RBGeneratorBase
from ..base_experiment import BaseExperiment


class RBExperimentBase(BaseExperiment):
    """Base experiment class for randomized benchmarking experiments"""
    def __init__(self, generator: Optional[RBGeneratorBase] = None):
        self._generator = generator
        circuit_options = {"active_seeds": self._generator.seeds()}
        super().__init__(max(generator.meas_qubits())+1, circuit_options=circuit_options)
#        self.reset()

    def circuits(self, backend=None, **circuit_options):
        active_seeds = circuit_options.get('active_seeds', self._generator.seeds())
        circuits = [c for c in self._generator._circuits if c.metadata['seed'] in active_seeds]
        return circuits

    @property
    def num_qubits(self):
        """Returns the number of qubits involved in the experiment"""
        return self._generator.num_meas_qubits()

    def lengths(self):
        """Returns the length of the RB-sequences for the experiment"""
        return self._generator.lengths()

    def group_type(self):
        """Returns the group type for the experiment"""
        return self._generator.rb_group_type()

    def run(self, backend, experiment_data=None, **kwargs):
        if 'basis_gates' not in kwargs:
            kwargs['basis_gates'] = self.default_basis_gates()
        return super().run(backend, experiment_data, active_seeds=self._generator.seeds(), **kwargs)

    def run_on_new_seeds(self, backend, experiment_data, num_of_seeds, **kwargs):
        """Adds new seeds and runs on them"""
        new_seeds = self._generator.add_seeds(num_of_seeds)
        return super().run(backend, experiment_data, active_seeds=new_seeds, **kwargs)

    def run_additional_shots(self, backend, experiment_data, **kwargs):
        """Runs more shot on the existing seeds"""
        return super().run(backend, experiment_data, **kwargs)

    def default_basis_gates(self) -> List[str]:
        """The default gate basis used when transpiling the RB circuits"""
        return ['id', 'rz', 'x', 'sx', 'cx']

    def gates_per_clifford(self, backend, basis_gates=None) -> float:
        """Computes the average number of gates per group element in the transpiled circuits"""
        if basis_gates is None:
            basis_gates = self.default_basis_gates()
        qubits = self._generator.meas_qubits()
        ngates = {qubit: {base: 0 for base in basis_gates} for qubit in qubits}
        transpiled_circuits_list = self.transpiled_circuits(backend, basis_gates=basis_gates)

        for transpiled_circuit in transpiled_circuits_list:
            for instr, qregs, _ in transpiled_circuit.data:
                for qreg in qregs:
                    if qreg.index in ngates and instr.name in ngates[qreg.index]:
                        ngates[qreg.index][instr.name] += 1

        # include inverse, ie + 1 for all clifford length
        length_per_seed = sum([length + 1 for length in self._generator.lengths()])
        total_ncliffs = self._generator.nseeds() * length_per_seed
        for qubit in qubits:
            for base in basis_gates:
                ngates[qubit][base] /= total_ncliffs

        return ngates
