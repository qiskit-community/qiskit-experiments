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

from qiskit import QuantumCircuit, QiskitError
from qiskit.providers import Backend
from qiskit.quantum_info import Clifford
from qiskit.providers.options import Options
from qiskit.circuit import Gate

from qiskit_experiments.framework import BaseExperiment, ParallelExperiment
from qiskit_experiments.curve_analysis.data_processing import probability
from .rb_analysis import RBAnalysis
from .clifford_utils import CliffordUtils
from .rb_utils import RBUtils


class StandardRB(BaseExperiment):
    """Standard Randomized Benchmarking Experiment class.

    Overview
        Randomized Benchmarking (RB) is an efficient and robust method
        for estimating the average error-rate of a set of quantum gate operations.
        See `Qiskit Textbook
        <https://qiskit.org/textbook/ch-quantum-hardware/randomized-benchmarking.html>`_
        for an explanation on the RB method.

        A standard RB experiment generates sequences of random Cliffords
        such that the unitary computed by the sequences is the identity.
        After running the sequences on a backend, it calculates the probabilities to get back to
        the ground state, fits an exponentially decaying curve, and estimates
        the Error Per Clifford (EPC), as described in Ref. [1, 2].

        See :class:`RBAnalysis` documentation for additional
        information on RB experiment analysis.

        See :class:`RBUtils` documentation for additional information
        on estimating the Error Per Gate (EPG) for 1-qubit and 2-qubit gates,
        from 1-qubit and 2-qubit standard RB experiments, by Ref. [3].

    References
        1. Easwar Magesan, J. M. Gambetta, and Joseph Emerson,
           Robust randomized benchmarking of quantum processes,
           `arXiv:quant-ph/1009.3639 <https://arxiv.org/pdf/1009.3639>`_
        2. Easwar Magesan, Jay M. Gambetta, and Joseph Emerson,
           Characterizing Quantum Gates via Randomized Benchmarking,
           `arXiv:quant-ph/1009.6887 <https://arxiv.org/pdf/1109.6887>`_
        3. David C. McKay, Sarah Sheldon, John A. Smolin, Jerry M. Chow, and Jay M. Gambetta,
           Three Qubit Randomized Benchmarking, `arXiv:quant-ph/1712.06550
           <https://arxiv.org/pdf/1712.06550>`_

    Analysis Class
        :class:`RBAnalysis`

    Experiment Options
        - **lengths**: A list of RB sequences lengths.
        - **num_samples**: Number of samples to generate for each sequence length.

    Analysis Options
        - **error_dict**: Optional. Error estimates for gates from the backend properties.
        - **epg_1_qubit**: Optional. EPG data for the 1-qubit gate involved, assumed to
          have been obtained from previous experiments. This is used to estimate the 2-qubit EPG.
        - **gate_error_ratio**: An estimate for the ratios between errors on different gates.
    """

    # Analysis class for experiment
    __analysis_class__ = RBAnalysis

    def __init__(
        self,
        qubits: Union[int, Iterable[int]],
        lengths: Iterable[int],
        num_samples: int = 3,
        seed: Optional[Union[int, Generator]] = None,
        full_sampling: Optional[bool] = False,
    ):
        """Initialize a standard randomized benchmarking experiment.

        Args:
            qubits: The number of qubits or list of
                    physical qubits for the experiment.
            lengths: A list of RB sequences lengths.
            num_samples: Number of samples to generate for each sequence length.
            seed: Seed or generator object for random number
                  generation. If None default_rng will be used.
            full_sampling: If True all Cliffords are independently sampled for
                           all lengths. If False for sample of lengths longer
                           sequences are constructed by appending additional
                           Clifford samples to shorter sequences.
                           The default is False.
        """
        # Initialize base experiment
        super().__init__(qubits)
        self._verify_parameters(lengths, num_samples)

        # Set configurable options
        self.set_experiment_options(lengths=list(lengths), num_samples=num_samples)
        self.set_analysis_options(data_processor=probability(outcome="0" * self.num_qubits))

        # Set fixed options
        self._full_sampling = full_sampling
        self._clifford_utils = CliffordUtils()

        if not isinstance(seed, Generator):
            self._rng = default_rng(seed=seed)
        else:
            self._rng = seed

    def _verify_parameters(self, lengths, num_samples):
        """Verify input correctness, raise QiskitError if needed"""
        if any(length <= 0 for length in lengths):
            raise QiskitError(
                f"The lengths list {lengths} should only contain " "positive elements."
            )
        if len(set(lengths)) != len(lengths):
            raise QiskitError(
                f"The lengths list {lengths} should not contain " "duplicate elements."
            )
        if num_samples <= 0:
            raise QiskitError(f"The number of samples {num_samples} should " "be positive.")

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
            group_elt_gate = group_elt_circ
            if not isinstance(group_elt_gate, Gate):
                group_elt_gate = group_elt_gate.to_gate()
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
                    "group": "Clifford",
                    "physical_qubits": self.physical_qubits,
                }
                rb_circ.measure_all()
                circuits.append(rb_circ)
        return circuits

    def _get_circuit_metadata(self, circuit):
        if circuit.metadata["experiment_type"] == self._type:
            return circuit.metadata
        if circuit.metadata["experiment_type"] == ParallelExperiment.__name__:
            for meta in circuit.metadata["composite_metadata"]:
                if meta["physical_qubits"] == self.physical_qubits:
                    return meta
        return None

    def _postprocess_transpiled_circuits(self, circuits, backend, **run_options):
        """Additional post-processing of transpiled circuits before running on backend"""
        for c in circuits:
            meta = self._get_circuit_metadata(c)
            if meta is not None:
                c_count_ops = RBUtils.count_ops(c, self.physical_qubits)
                circuit_length = meta["xval"]
                count_ops = [(key, (value, circuit_length)) for key, value in c_count_ops.items()]
                meta.update({"count_ops": count_ops})
