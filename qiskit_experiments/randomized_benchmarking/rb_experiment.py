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
Standard RB Experiment class.
"""
from typing import Union, Iterable, Optional, List

import numpy as np
from numpy.random import Generator, default_rng

from qiskit import QuantumCircuit
from qiskit.providers import Backend
from qiskit.quantum_info import Clifford
from qiskit.providers.options import Options

from qiskit_experiments.base_experiment import BaseExperiment
from .rb_analysis import RBAnalysis
from .clifford_utils import CliffordUtils


class RBExperiment(BaseExperiment):
    """RB Experiment class.

    Experiment Options:
        lengths: A list of RB sequences lengths.
        num_samples: number of samples to generate for each sequence length.
    """

    # Analysis class for experiment
    __analysis_class__ = RBAnalysis

    def __init__(
        self,
        qubits: Union[int, Iterable[int]],
        lengths: Iterable[int],
        num_samples: int = 1,
        seed: Optional[Union[int, Generator]] = None,
        full_sampling: bool = False,
    ):
        """Standard randomized benchmarking experiment.

        Args:
            qubits: the number of qubits or list of
                    physical qubits for the experiment.
            lengths: A list of RB sequences lengths.
            num_samples: number of samples to generate for each sequence length.
            seed: Seed or generator object for random number
                  generation. If None default_rng will be used.
            full_sampling: If True all Cliffords are independently sampled for
                           all lengths. If False for sample of lengths longer
                           sequences are constructed by appending additional
                           Clifford samples to shorter sequences.
        """
        # Initialize base experiment
        super().__init__(qubits)

        # Set configurable options
        self.set_experiment_options(lengths=list(lengths), num_samples=num_samples)

        # Set fixed options
        self._full_sampling = full_sampling
        self._clifford_utils = CliffordUtils()

        if not isinstance(seed, Generator):
            self._rng = default_rng(seed=seed)
        else:
            self._rng = seed

    @classmethod
    def _default_experiment_options(cls):
        return Options(lengths=None, num_samples=None)

    # pylint: disable = arguments-differ
    def circuits(self, backend: Optional[Backend] = None) -> List[QuantumCircuit]:
        """Return a list of RB circuits.
        Args:
            backend (Backend): Optional, a backend object.
        Returns:
            A list of :class:`QuantumCircuit`.
        """
        circuits = []
        for _ in range(self.experiment_options.num_samples):
            circuits += self._sample_circuits(self.experiment_options.lengths, seed=self._rng)
        return circuits

    def _sample_circuits(
        self, lengths: Iterable[int], seed: Optional[Union[int, Generator]] = None
    ) -> List[QuantumCircuit]:
        """Return a list RB circuits for the given lengths.

        Args:
            lengths: A list of RB sequences lengths.
            seed: Seed or generator object for random number
                  generation. If None default_rng will be used.

        Returns:
            A list of :class:`QuantumCircuit`.
        """
        circuits = []
        for length in lengths if self._full_sampling else [lengths[-1]]:
            elements = self._clifford_utils.random_clifford_circuits(self.num_qubits, length, seed)
            element_lengths = [len(elements)] if self._full_sampling else lengths
            circuits += self._generate_circuit(elements, element_lengths)
        return circuits

    def _generate_circuit(
        self, elements: Iterable[Clifford], lengths: Iterable[int]
    ) -> List[QuantumCircuit]:
        """Return the RB circuits constructed from the given element list.

        Args:
            elements: A list of Clifford elements
            lengths: A list of RB sequences lengths.

        Returns:
            A list of :class:`QuantumCircuit`s.

        Additional information:
            The circuits are constructed iteratively; each circuit is obtained
            by extending the previous circuit (without the inversion and measurement gates)
        """
        qubits = list(range(self.num_qubits))
        circuits = []

        circs = [QuantumCircuit(self.num_qubits) for _ in range(len(lengths))]
        for circ in circs:
            circ.barrier(qubits)
        circ_op = Clifford(np.eye(2 * self.num_qubits))

        for current_length, group_elt_circ in enumerate(elements):
            group_elt_gate = group_elt_circ.to_gate()
            circ_op = circ_op.compose(Clifford(group_elt_circ))
            for circ in circs:
                circ.append(group_elt_gate, qubits)
                circ.barrier(qubits)
            if current_length + 1 in lengths:
                # copy circuit and add inverse
                inv = circ_op.adjoint()
                rb_circ = circs.pop()
                rb_circ.append(inv, qubits)
                rb_circ.barrier(qubits)
                rb_circ.metadata = {
                    "experiment_type": self._type,
                    "xval": current_length + 1,
                    "ylabel": self.num_qubits * "0",
                    "group": "Clifford",
                    "qubits": self.physical_qubits,
                }
                rb_circ.measure_all()
                circuits.append(rb_circ)
        return circuits
