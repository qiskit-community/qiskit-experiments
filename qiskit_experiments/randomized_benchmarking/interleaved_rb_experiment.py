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
Interleaved RB Experiment class.
"""
from typing import Union, Iterable, Optional, List

from numpy.random import Generator

from qiskit import QuantumCircuit
from qiskit.circuit import Instruction
from qiskit.quantum_info import Clifford, random_clifford

from .rb_experiment import RBExperiment
from .interleaved_rb_analysis import InterleavedRBAnalysis


class InterleavedRBExperiment(RBExperiment):
    """Interleaved RB Experiment class"""

    # Analysis class for experiment
    __analysis_class__ = InterleavedRBAnalysis

    def __init__(
        self,
        interleaved_element: Union[QuantumCircuit, Instruction, Clifford],
        qubits: Union[int, Iterable[int]],
        lengths: Iterable[int],
        num_samples: int = 1,
        seed: Optional[Union[int, Generator]] = None,
        full_sampling: bool = False,
    ):
        """Interleaved randomized benchmarking experiment.

        Args:
            interleaved_element: the element to interleave,
                    given either as a group element or as an instruction/circuit
            qubits: the number of qubits or list of
                    physical qubits for the experiment.
            lengths: A list of RB sequences lengths.
            num_samples: number of samples to generate for each
                         sequence length
            seed: Seed or generator object for random number
                  generation. If None default_rng will be used.
            full_sampling: If True all Cliffords are independently sampled for
                           all lengths. If False for sample of lengths longer
                           sequences are constructed by appending additional
                           Clifford samples to shorter sequences.
        """
        self._interleaved_element = interleaved_element
        super().__init__(qubits, lengths, num_samples, seed, full_sampling)

    def _sample_circuits(self, lengths, seed=None):
        circuits = []
        for length in lengths if self._full_sampling else [lengths[-1]]:
            elements = [random_clifford(self.num_qubits, seed=seed) for _ in range(length)]
            element_lengths = [len(elements)] if self._full_sampling else lengths
            std_circuits = self._generate_circuit(elements, element_lengths)
            for circuit in std_circuits:
                circuit.metadata["series"] = 0
                circuit.metadata["series_name"] = "standard"
            circuits += std_circuits

            int_elements = self._interleave(elements)
            int_elements_lengths = [length * 2 for length in element_lengths]
            int_circuits = self._generate_circuit(int_elements, int_elements_lengths)
            for circuit in int_circuits:
                circuit.metadata["series"] = 1
                circuit.metadata["series_name"] = "interleaved"
                circuit.metadata["xval"] = circuit.metadata["xval"] // 2
            circuits += int_circuits
        return circuits

    def _interleave(self, element_list: List) -> List:
        """Interleaving the interleaved element inside the element list.

        Args:
            element_list: The list of elements we add the interleaved element to.

        Returns:
            The new list with the element interleaved.
        """
        new_element_list = []
        for element in element_list:
            new_element_list.append(element)
            new_element_list.append(self._interleaved_element)
        return new_element_list
