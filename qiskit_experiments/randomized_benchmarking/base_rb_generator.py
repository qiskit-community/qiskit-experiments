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

from typing import List, Dict, Union, Optional
import copy
from numpy.random import RandomState
from qiskit import (QuantumCircuit, QuantumRegister, ClassicalRegister)
from .rb_groups import RBgroup


class RBGeneratorBase():
    """Base generator class for randomized benchmarking experiments"""
    def __init__(self,
                 nseeds: int = 1,
                 qubits: List[int] = (0,),
                 lengths: List[int] = (1, 10, 20),
                 group_gates: Optional[str] = None,
                 rand_seed: Optional[Union[int, RandomState]] = None,
                 ):
        """Initialize the circuit generator for a randomized banchmarking experiment.

            Args:
                nseeds: number of different seeds (random circuits) to generate
                qubits: the qubits particiapting in the experiment
                lengths: for each seed, the lengths of the circuits used for that seed.
                group_gates: which group the circuits is based on
                rand_seed: optional random number seed
                name: the name of the generator

            Additional Information:
                Randomized benchmarking circuit generation usually consists of the following steps:
                1) Generate a long circuit composed from random gates from a group allowing easy
                inversion of the resulting unitary. We support the Clifford and the CNOT-dihedral
                groups.
                2) Create a sequence of circuits, C1, C2, ... , Cn such that Cn is the original
                circuit and C1, C2, ... are prefixes of Cn with lengths corresponding to the values
                given in the `lengths` parameter.
                3) For each gate in the circuits, add an inverting gate and a measurement at the
                end.
            """
        self._nseeds = nseeds
        self._physical_qubits = list(set(qubits))
        self._lengths = lengths
        self._rb_group = RBgroup(group_gates)
        if self._rb_group.group_gates_type() == 0:
            self._rb_group_type = 'clifford'
        if self._rb_group.group_gates_type() == 1:
            self._rb_group_type = 'cnot_dihedral'
        self._rand_seed = rand_seed
        self._circuits = []

    def num_physical_qubits(self):
        """Returns the number of qubits participating in the experiment"""
        return len(self._physical_qubits)

    def physical_qubits(self):
        """Returns the qubits participating in the experiment"""
        return self._physical_qubits

    def num_qubits(self):
        return self.num_physical_qubits()

    def qubits(self):
        return list(range(self.num_qubits()))

    def lengths(self):
        """Returns the lengths of the RB circuits (for each seed)"""
        return self._lengths

    def nseeds(self):
        """Returns the number of seeds in the RB experiment"""
        return self._nseeds

    def seeds(self):
        """Returns all the seeds in the RB experiment"""
        return list(range(self.nseeds()))

    def rb_group_type(self):
        """Returns the type of the group for the experiment's circuits"""
        return self._rb_group_type

    def add_seeds(self, num_of_seeds: int) -> List[int]:
        """Adds additional seeds to the experiment
        Args:
            num_of_seeds: The number of seeds to add to the experiment
        Returns:
            The numbers of the generated seeds
        """
        current_seed_number = self._nseeds
        self._nseeds += num_of_seeds
        self.generate_circuits(start_seed=current_seed_number)
        return list(range(current_seed_number, self._nseeds))

    def generate_circuits(self, start_seed: int = 0):
        """Generates the circuits for the experiment based on the stored parameters
        Args:
            start_seed: The number of seed to start generating from, if not all circuits
            needs to be generated
        """
        for seed in range(start_seed, self._nseeds):
            circuits = self.generate_circuits_for_seed()
            for circuit in circuits:
                circuit.metadata['seed'] = seed
                circuit.metadata['group_type'] = self._rb_group_type
                self.add_measurements(circuit)
                self.set_circuit_name(circuit)
                circuit.metadata['circuit_name'] = circuit.name
                self._circuits.append(circuit)

    def set_circuit_name(self, circuit: QuantumCircuit):
        """Sets the name of an RB circuit based on its metadata
        Args:
            circuit: The circuit to set
        Additional information:
            The general name format is "rb_length_A_seed_B"
            Where A is the index of the length in the length lists (not the length itself)
            and B is the number of the seed.

            When the underlying group is CNOT-Dihedral and not Clifford, the format is
            "rb_cnotdihedral_C_length_A_seed_B"
            Where C is either "X" or "Z", depending on the measurement basis used in the circuit

            For non-standard types of RB experiments,
            an additional TYPE string is part of the format:
            "rb_TYPE_length_A_seed_B"
        """
        meta = circuit.metadata
        name = "rb_"
        if meta['group_type'] == 'cnot_dihedral':
            name += "cnotdihedral_{}_".format(meta['cnot_basis'])
        if self.circuit_type_string(meta) is not None:
            name += (self.circuit_type_string(meta) + "_")
        name += "length_{}_seed_{}".format(meta['length_index'], meta['seed'])
        circuit.name = name

    def circuit_type_string(self, meta: Optional[Dict[str, any]]) -> str:
        """Returns an additional string describing the exact type of the circuit
        for non-standard RB experiment; meant to be overridden by derived classes
        Args:
            meta: The metadata of the circuit
        """
        # pylint: disable=unused-argument
        return None

    def generate_circuits_for_seed(self) -> List[QuantumCircuit]:
        """Generates the RB circuits for a single seed
        Returns:
            The list of circuits.
        Additional information:
            Generates for each seed the sequence C1, C2, ..., Cn of circuits
            corresponding to the `lengths` parameter.

            Meant to be overriden by derived classes
        """
        element_list = self.generate_random_element_list(self._lengths[-1])
        element_lists = self.split_element_list(element_list, self._lengths)
        circuits = self.generate_circuits_from_elements(element_lists)
        return circuits

    def generate_circuits_from_elements(self, element_lists: List) -> List[QuantumCircuit]:
        """Generates the circuits corresponding to the group elements
        Args:
            element_lists: Lists of group elements to build the circuits from
        Returns:
            A list of circuits.
        Additional information:
            If the element lists are L1, L2, ..., Ln then the resulting circuits
            are obtained from the gates in L1, L1+L2, L1+L2+L3, ... L1+...+Ln
            With an additional inverting gate and measurement at the end of each circuit
        """
        result = []
        qr = QuantumRegister(self.num_qubits(), 'qr')
        cr = ClassicalRegister(self.num_qubits(), 'cr')
        circ = QuantumCircuit(qr, cr)
        current_element = self._rb_group.iden(self.num_qubits())
        for length_index, element_list in enumerate(element_lists):
            for element in element_list:
                current_element = self._rb_group.compose(current_element, element['group_element'])
                circ += self.replace_q_indices(
                    element['circuit_element'],
                    self.qubits(), qr)
                # add a barrier
                circ.barrier(*[qr[x] for x in self.qubits()])
            # finished with the current list - output a circuit based on what we have
            output_circ = QuantumCircuit(qr, cr)
            output_meta = {'length_index': length_index}
            output_circ += circ
            inv_circuit = self._rb_group.inverse(current_element)
            output_circ += self.replace_q_indices(inv_circuit, self.qubits(), qr)
            if self._rb_group_type == 'cnot_dihedral':
                output_meta['cnot_basis'] = 'Z'
                cnot_circuit = self.generate_cnot_x_circuit(output_circ, output_meta)
                result.append(cnot_circuit)
            output_circ.metadata = output_meta
            result.append(output_circ)
        return result

    def generate_cnot_x_circuit(self,
                                circuit: QuantumCircuit,
                                meta: Dict[str, any]
                                ) -> QuantumCircuit:
        """Generates an additional circuit with preperation and measurement in the X basis
        Args:
            circuit: The circuit to duplicate and transform to the X basis
            meta: The metadata of the circuit, to copy and mofity accordingly
        Returns:
            The new circuit
        Additional information:
            When using the CNOT-dihedral group, we need two copies of each circuit;
            one is identical to the one used in standard RB, the other one
            is in the X-basis, an effect obtained by adding Hadamard gates in the beginning
            and end (pre-measurement) of the circuit.
        """
        cnot_circuit = QuantumCircuit(circuit.qregs[0], circuit.cregs[0])
        cnot_meta = copy.copy(meta)
        for qubit in self.qubits():
            cnot_circuit.h(qubit)
            cnot_circuit.barrier(qubit)
        cnot_circuit += circuit
        for qubit in self.qubits():
            cnot_circuit.barrier(qubit)
            cnot_circuit.h(qubit)

        cnot_meta['cnot_basis'] = 'X'
        cnot_circuit.metadata = cnot_meta
        return cnot_circuit

    def generate_random_element_list(self, length: int) -> List[Dict]:
        """Generates a random list of groups elements
        Args:
            length: The length of the list
        Returns:
            The element list. Each element is given via a dictionary as the pair
            {'group_element': g, 'circuit_element': c}
            where g is the group representation of the element (used when computing the inverting
            gate), and c is the actual circuit to add to the RB circuit.
        """
        element_list = []
        for _ in range(length):
            if self._rand_seed is not None:
                self._rand_seed += 1
            element = self._rb_group.random(self.num_qubits(), self._rand_seed)
            element_list.append({'group_element': element,
                                 'circuit_element': self._rb_group.to_circuit(element)})
        return element_list

    def split_element_list(self, element_list: List, lengths: List[int]) -> List[List]:
        """Splits the element list according to the given lengths
        Args:
            element_list: The element list to be split
            lengths: the required lengths of the circuits generated from the lists
        Returns:
            The list of element lists after the split
        Additional information:
            The output list [L1, L2, ..., Ln] satisfies the following conditions:
            1) L1 + L2 + ... + Ln is the original list
            2) len(L1 + L2 + ... + Lk) is equal to the kth element in lengths

            e.g. if list = `[1,2,3,4,5,6]` and lengths = `[1,3,6]` then the output is
            `[[1],[2,3],[4,5,6]]`
        """
        element_lists = []
        current_element_list = []
        stop_indexes = [x - 1 for x in lengths]
        for index, element in enumerate(element_list):
            current_element_list.append(element)
            if index == stop_indexes[0]:
                element_lists.append(current_element_list)
                current_element_list = []
                stop_indexes.pop(0)
        return element_lists

    def replace_q_indices(self,
                          circuit: QuantumCircuit,
                          q_nums: List[int],
                          qr: QuantumRegister) -> QuantumCircuit:
        """Take a circuit that is ordered from 0,1,2 qubits and replace 0 with the
        qubit label in the first index of q_nums, 1 with the second index...

        Args:
            circuit: circuit to operate on
            q_nums: list of qubit indices
            qr: A quantum register to use for the output circuit

        Returns:
            Updated circuit
        """

        new_circuit = QuantumCircuit(qr)
        for instr, qargs, cargs in circuit.data:
            new_qargs = [
                qr[q_nums[x]] for x in [arg.index for arg in qargs]]
            new_op = copy.deepcopy((instr, new_qargs, cargs))
            new_circuit.data.append(new_op)

        return new_circuit

    def add_measurements(self, circuit: QuantumCircuit):
        """Adds a measurement gate to each measured qubit in the circuit
        Args:
            circuit: The circuit to add measurement gates to
        """
        for clbit, qubit in enumerate(self.qubits()):
            circuit.measure(qubit, clbit)

    def add_extra_meta(self, circuits: List[QuantumCircuit],
                       extra_meta: Dict[str, any]):
        """Adds additional metadata to each element of the circuit list
        Args:
            circuits: A list of circuits
            extra_meta: The additional metdata to add to each m in the list
        """
        for circuit in circuits:
            for key, value in extra_meta.items():
                circuit.metadata[key] = value

    def circuits(self) -> List[QuantumCircuit]:
        """Return a list of experiment circuits."""
        return self._circuits
