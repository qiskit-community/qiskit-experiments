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
import logging
from collections import defaultdict
from typing import Union, Iterable, Optional, List, Sequence

import numpy as np
from numpy.random import Generator, default_rng
from numpy.random.bit_generator import BitGenerator, SeedSequence
import time

from qiskit import QuantumCircuit, QuantumRegister, QiskitError
from qiskit.circuit import Instruction
from qiskit.circuit.quantumregister import Qubit
from qiskit.quantum_info import Clifford
from qiskit.providers.backend import Backend

from qiskit_experiments.framework import BaseExperiment, Options
from qiskit_experiments.framework.restless_mixin import RestlessMixin
from .rb_analysis import RBAnalysis
from .clifford_utils import CliffordUtils
from .clifford_data import CLIFF_COMPOSE_DATA, CLIFF_INVERSE_DATA

LOG = logging.getLogger(__name__)

class StandardRB(BaseExperiment, RestlessMixin):
    """Standard randomized benchmarking experiment.

    # section: overview
        Randomized Benchmarking (RB) is an efficient and robust method
        for estimating the average error-rate of a set of quantum gate operations.
        See `Qiskit Textbook
        <https://qiskit.org/textbook/ch-quantum-hardware/randomized-benchmarking.html>`_
        for an explanation on the RB method.

        A standard RB experiment generates sequences of random Cliffords
        such that the unitary computed by the sequences is the identity.
        After running the sequences on a backend, it calculates the probabilities to get back to
        the ground state, fits an exponentially decaying curve, and estimates
        the Error Per Clifford (EPC), as described in Refs. [1, 2].

    # section: analysis_ref
        :py:class:`RBAnalysis`

    # section: reference
        .. ref_arxiv:: 1 1009.3639
        .. ref_arxiv:: 2 1109.6887

    """

    def __init__(
        self,
        qubits: Sequence[int],
        lengths: Iterable[int],
        backend: Optional[Backend] = None,
        num_samples: int = 3,
        seed: Optional[Union[int, SeedSequence, BitGenerator, Generator]] = None,
        full_sampling: Optional[bool] = False,
        transpiled_rb: Optional[bool] = False
    ):
        """Initialize a standard randomized benchmarking experiment.

        Args:
            qubits: list of physical qubits for the experiment.
            lengths: A list of RB sequences lengths.
            backend: The backend to run the experiment on.
            num_samples: Number of samples to generate for each sequence length.
            seed: Optional, seed used to initialize ``numpy.random.default_rng``.
                when generating circuits. The ``default_rng`` will be initialized
                with this seed value everytime :meth:`circuits` is called.
            full_sampling: If True all Cliffords are independently sampled for
                all lengths. If False for sample of lengths longer sequences are constructed by
                appending additional Clifford samples to shorter sequences. The default is ``False``.
        """
        # Initialize base experiment
        super().__init__(qubits, analysis=RBAnalysis(), backend=backend)
        self._verify_parameters(lengths, num_samples)

        # Set configurable options
        self.set_experiment_options(lengths=list(lengths), num_samples=num_samples, seed=seed)
        self.analysis.set_options(outcome="0" * self.num_qubits)

        # Set fixed options
        self._full_sampling = full_sampling
        self._transpiled_rb = transpiled_rb
        self._clifford_utils = CliffordUtils()
        self._transpiled_cliff_circuits = None

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
    def _default_experiment_options(cls) -> Options:
        """Default experiment options.

        Experiment Options:
            lengths (List[int]): A list of RB sequences lengths.
            num_samples (int): Number of samples to generate for each sequence length.
            seed (None or int or SeedSequence or BitGenerator or Generator): A seed
                used to initialize ``numpy.random.default_rng`` when generating circuits.
                The ``default_rng`` will be initialized with this seed value everytime
                :meth:`circuits` is called.
        """
        options = super()._default_experiment_options()

        options.lengths = None
        options.num_samples = None
        options.seed = None

        return options

    def circuits(self) -> List[QuantumCircuit]:
        """Return a list of RB circuits.

        Returns:
            A list of :class:`QuantumCircuit`.
        """
        rng = default_rng(seed=self.experiment_options.seed)
        circuits = []
        if self._transpiled_rb and self._transpiled_cliff_circuits == None:
            self._transpiled_cliff_circuits = \
                CliffordUtils.generate_1q_transpiled_clifford_circuits(basis_gates=self.transpile_options.basis_gates)
        for _ in range(self.experiment_options.num_samples):
            if self.num_qubits == 1 and self._transpiled_rb:
                rb_circuits, _ = self._build_rb_circuits(self.experiment_options.lengths, rng)
                circuits += rb_circuits
            else:
                circuits += self._sample_circuits(self.experiment_options.lengths, rng)
        return circuits

    def _sample_circuits(self, lengths: Iterable[int], rng: Generator) -> List[QuantumCircuit]:
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
            elements = self._clifford_utils.random_clifford_circuits(self.num_qubits, length, rng)
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
            if isinstance(group_elt_circ, tuple):
                group_elt_gate = group_elt_circ[0]
                group_elt_op = group_elt_circ[1]
            else:
                group_elt_gate = group_elt_circ
                group_elt_op = Clifford(group_elt_circ)

            if not isinstance(group_elt_gate, Instruction):
                group_elt_gate = group_elt_gate.to_instruction()
            circ_op = circ_op.compose(group_elt_op)
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

    def _build_rb_circuits(self, lengths, rng, is_interleaved=False,
                           interleaved_element=None):
        """
            build_rb_circuits
            Args:
                    lengths: A list of RB sequence lengths. We create random circuits
                             where the number of cliffords in each is defined in lengths.
                    rng: Generator object for random number generation.
                         If None, default_rng will be used.
            To create the RB circuit, we use a mapping between Cliffords and integers
            defined in the file clifford_data.py. The operations compose and inverse  are much faster
            when performed on the integers rather than on the Cliffords themselves.
            """
        if self._full_sampling:
            return self._build_rb_circuits_full_sampling(lengths, rng, is_interleaved,
                                                         interleaved_element)

        max_qubit = max(self.physical_qubits) + 1
        all_rb_circuits = []
        if is_interleaved:
            all_rb_interleaved_circuits = []
            interleaved_elem_num = \
                CliffordUtils.num_from_1_qubit_clifford(
                    interleaved_element,
                    self.transpile_options.basis_gates,
                )
        else:
            all_rb_interleaved_circuits = None

        # When full_sampling==False, each circuit is the prefix of the next circuit (without the
        # inverse Clifford at the end of the circuit. The variable 'circ' will contain the growing circuit.
        # When each circuit reaches its length, we copy it to rb_circ, append the inverse,
        # and add it to the list of circuits.

        if is_interleaved:
            interleaved_circ = QuantumCircuit(max_qubit, 1)
            interleaved_circ.barrier(0)
        else:
            interleaved_circ = None

        random_samples = rng.integers(24, size=lengths[-1])
        circ = QuantumCircuit(max_qubit, 1)
        circ.barrier(0)

        composed_cliff_num = 0   # 0 is the Clifford that is Id
        composed_interleaved_num = 0
        prev_length = 0

        for length in lengths:
            for i in range(prev_length, length):
                rand = random_samples[i]
                # choose random clifford
                next_circ = self._transpiled_cliff_circuits[rand]
                circ.compose(next_circ, inplace=True)
                composed_cliff_num = CLIFF_COMPOSE_DATA[(composed_cliff_num, rand)]
                circ.barrier(0)
                if is_interleaved:
                    interleaved_circ.compose(next_circ, inplace=True)
                    composed_interleaved_num = CLIFF_COMPOSE_DATA[(composed_interleaved_num, rand)]
                    interleaved_circ.barrier(0)
                    interleaved_circ.compose(interleaved_element, inplace=True)
                    composed_interleaved_num = CLIFF_COMPOSE_DATA[(composed_interleaved_num, interleaved_elem_num)]
                    interleaved_circ.barrier(0)

                if i == length-1:
                    rb_circ = circ.copy()    # circ is used as the prefix of the next circuit
                    inverse_clifford_num = CLIFF_INVERSE_DATA[composed_cliff_num]
                    # append the inverse
                    rb_circ.compose(self._transpiled_cliff_circuits[inverse_clifford_num], inplace=True)
                    rb_circ.measure(0, 0)

                    rb_circ.metadata = {
                        "experiment_type": "rb",
                        "xval": length,
                        "group": "Clifford",
                        "physical_qubits": self.physical_qubits,
                        "interleaved" : False
                    }
                    all_rb_circuits.append(rb_circ)
                    if is_interleaved:
                        rb_interleaved_circ = interleaved_circ.copy()  # interleaved_circ is used as the prefix of the next circuit
                        # append the inverse
                        inverse_interleaved_num = CLIFF_INVERSE_DATA[composed_interleaved_num]
                        rb_interleaved_circ.compose(self._transpiled_cliff_circuits[inverse_interleaved_num], inplace=True)
                        rb_interleaved_circ.measure(0, 0)

                        rb_interleaved_circ.metadata = {
                            "experiment_type": "rb",
                            "xval": length,
                            "group": "Clifford",
                            "physical_qubits": self.physical_qubits,
                            "interleaved": True,
                        }
                        all_rb_interleaved_circuits.append(rb_interleaved_circ)

                prev_length = i + 1
        return all_rb_circuits, all_rb_interleaved_circuits

    def _build_rb_circuits_full_sampling(self, lengths, rng, is_interleaved=False,
                                         interleaved_element=None):
        all_rb_circuits = []
        if is_interleaved:
            all_rb_interleaved_circuits = []
            interleaved_elem_num = \
                CliffordUtils.num_from_1_qubit_clifford(
                    interleaved_element,
                    self.transpile_options.basis_gates
                )
        else:
            all_rb_interleaved_circuits = None

        max_qubit = max(self.physical_qubits) + 1
        for length in lengths:
            # We define the circuit size here, for the layout that will
            # be created later
            rb_circ = QuantumCircuit(max_qubit, 1)
            rb_circ.barrier(0)
            if is_interleaved:
                rb_interleaved_circ = QuantumCircuit(max_qubit, 1)
                rb_interleaved_circ.barrier(0)
            else:
                rb_interleaved_circ = None

            random_samples = rng.integers(24, size=length)
            # For full_sampling, we create each circuit independently.
            composed_cliff_num = 0
            composed_interleaved_num = 0
            for i in range(length):
                # choose random clifford
                rand = random_samples[i]
                next_circ = self._transpiled_cliff_circuits[rand].copy()
                rb_circ.compose(next_circ, inplace=True)
                composed_cliff_num = CLIFF_COMPOSE_DATA[(composed_cliff_num, rand)]
                rb_circ.barrier(0)
                if is_interleaved:
                    rb_interleaved_circ.compose(next_circ, inplace=True)
                    composed_interleaved_num = CLIFF_COMPOSE_DATA[(composed_interleaved_num, rand)]
                    rb_interleaved_circ.barrier(0)
                    rb_interleaved_circ.compose(interleaved_element, inplace=True)
                    composed_interleaved_num = CLIFF_COMPOSE_DATA[(composed_interleaved_num, interleaved_elem_num)]
                    rb_interleaved_circ.barrier(0)

            inverse_clifford_num = CLIFF_INVERSE_DATA[composed_cliff_num]
            # append the inverse
            rb_circ.compose(self._transpiled_cliff_circuits[inverse_clifford_num], inplace=True)
            rb_circ.measure(0, 0)
            rb_circ.metadata = {
                "experiment_type": "rb",
                "xval": length,
                "group": "Clifford",
                "physical_qubits": self.physical_qubits,
                "interleaved": False,
            }
            if is_interleaved:
                inverse_interleaved_num = CLIFF_INVERSE_DATA[composed_interleaved_num]
                rb_interleaved_circ.compose(self._transpiled_cliff_circuits[inverse_interleaved_num], inplace=True)
                rb_interleaved_circ.measure(0, 0)
                rb_interleaved_circ.metadata = {
                    "experiment_type": "rb",
                    "xval": length,
                    "group": "Clifford",
                    "physical_qubits": self.physical_qubits,
                    "interleaved" : True,
                }
            all_rb_circuits.append(rb_circ)
            if is_interleaved:
                all_rb_interleaved_circuits.append(rb_interleaved_circ)
        return all_rb_circuits, all_rb_interleaved_circuits

    # This method does a quick layout to avoid calling 'transpile()' which is
    # very costly in performance
    # We simply copy the circuit to a new circuit where we define the mapping
    # of the qubit to the single physical qubit that was requested by the user
    # This is a hack, and would be better if transpile() implemented it.
    def _layout_for_rb_single_qubit(self):
        circuits = self.circuits()
        transpiled = []
        for c in circuits:
            qr = QuantumRegister(c.num_qubits, name='q')
            qubit = Qubit(qr, self.physical_qubits[0])
            c_new = QuantumCircuit(
                *c.qregs,
                *c.cregs,
                name=c.name,
                global_phase=c.global_phase,
                metadata=c.metadata
            )
            new_data = []
            for inst, qargs, cargs in c.data:
                new_data.append((inst, [qubit], cargs))
            c_new.data = new_data
            transpiled.append(c_new)
        return transpiled

    def _transpiled_circuits(self) -> List[QuantumCircuit]:
        """Return a list of experiment circuits, transpiled."""
        if self.num_qubits==1 and self._transpiled_rb:
            transpiled = self._layout_for_rb_single_qubit()
        else:
            transpiled = super()._transpiled_circuits()

        if self.analysis.options.get("gate_error_ratio", None) is None:
            # Gate errors are not computed, then counting ops is not necessary.
            return transpiled

        # Compute average basis gate numbers per Clifford operation
        # This is probably main source of performance regression.
        # This should be integrated into transpile pass in future.
        for circ in transpiled:
            count_ops_result = defaultdict(int)
            # This is physical circuits, i.e. qargs is physical index
            for inst, qargs, _ in circ.data:
                if inst.name in ("measure", "reset", "delay", "barrier", "snapshot"):
                    continue
                qinds = [circ.find_bit(q).index for q in qargs]
                if not set(self.physical_qubits).issuperset(qinds):
                    continue
                # Not aware of multi-qubit gate direction
                formatted_key = tuple(sorted(qinds)), inst.name
                count_ops_result[formatted_key] += 1
            circ.metadata["count_ops"] = tuple(count_ops_result.items())

        return transpiled

    def _metadata(self):
        metadata = super()._metadata()
        # Store measurement level and meas return if they have been
        # set for the experiment
        for run_opt in ["meas_level", "meas_return"]:
            if hasattr(self.run_options, run_opt):
                metadata[run_opt] = getattr(self.run_options, run_opt)

        return metadata

