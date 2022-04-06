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
import warnings
from typing import Union, Iterable, Optional, List, Sequence
from collections import defaultdict

import numpy as np
from numpy.random import Generator, default_rng
from numpy.random.bit_generator import BitGenerator, SeedSequence

from qiskit import QuantumCircuit, QiskitError
from qiskit.quantum_info import Clifford
from qiskit.circuit import Gate
from qiskit.providers.backend import Backend

from qiskit_experiments.framework import BaseExperiment, Options
from qiskit_experiments.framework.restless_mixin import RestlessMixin
from .rb_analysis import RBAnalysis
from .clifford_utils import CliffordUtils
from .rb_utils import RBUtils, lookup_epg_ratio


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

        See :class:`RBUtils` documentation for additional information
        on estimating the Error Per Gate (EPG) for 1-qubit and 2-qubit gates,
        from 1-qubit and 2-qubit standard RB experiments, by Ref. [3].

    # section: analysis_ref
        :py:class:`RBAnalysis`

    # section: reference
        .. ref_arxiv:: 1 1009.3639
        .. ref_arxiv:: 2 1109.6887
        .. ref_arxiv:: 3 1712.06550

    """

    def __init__(
        self,
        qubits: Sequence[int],
        lengths: Iterable[int],
        backend: Optional[Backend] = None,
        num_samples: int = 3,
        seed: Optional[Union[int, SeedSequence, BitGenerator, Generator]] = None,
        full_sampling: Optional[bool] = False,
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
        self._clifford_utils = CliffordUtils()

        # Evaluated in the execution stage. Values are copied to experiment data metadata.
        self._gate_error_ratio = None
        self._gate_counts_per_clifford = None

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
            gate_error_ratio (Union[str, Dict[Tuple[int, str], float]]): The assumption of error ratio
                of basis gates constituting RB Clifford sequences. When this value is set,
                the error per gate (EPG) values are computed from the estimated
                error per Clifford (EPC) parameter in the RB analysis.
                This value defaults to "default". When explicit gate error ratio is not provided,
                typical error ratio is provided by :func:`~qiskit_experiments.library.\
                randomized_benchmarking.rb_utils.lookup_epg_ratio`.
                The dictionary is keyed on a tuple of qubit index and string label of instruction.
                Defined instructions should appear in the ``basis_gates`` in the transpile options.
                If this value is set to ``False``, the computation of EPG values is skipped.
        """
        options = super()._default_experiment_options()

        options.lengths = None
        options.num_samples = None
        options.seed = None
        options.gate_error_ratio = "default"

        return options

    def circuits(self) -> List[QuantumCircuit]:
        """Return a list of RB circuits.

        Returns:
            A list of :class:`QuantumCircuit`.
        """
        rng = default_rng(seed=self.experiment_options.seed)
        circuits = []
        for _ in range(self.experiment_options.num_samples):
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

            if not isinstance(group_elt_gate, Gate):
                group_elt_gate = group_elt_gate.to_gate()
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

    def _finalize(self):
        super()._finalize()

        # Set constraints to compute basis gates EPGs from an estimated EPC
        gate_error_ratio = self.experiment_options.gate_error_ratio

        if gate_error_ratio != "default":
            self._gate_error_ratio = gate_error_ratio
            return

        # When 'default' is set, create standard gate error ratio from basis gates
        basis_gates = self.transpile_options.get("basis_gates", None)

        if basis_gates is None:
            try:
                basis_gates = self.backend.configuration().basis_gates
            except AttributeError:
                # When basis gates is not provided, disable EPG computation
                warnings.warn(
                    "The basis gates infromation is not available. Cannot compute EPGs.",
                    UserWarning,
                )
                self._gate_error_ratio = False

        gate_error_ratio = {}
        for basis_gate in basis_gates:
            r_epg = lookup_epg_ratio(basis_gate, self.num_qubits)
            if r_epg is None:
                continue
            gate_error_ratio[(tuple(self.physical_qubits), basis_gate)] = r_epg

        self._gate_error_ratio = gate_error_ratio

    def _transpiled_circuits(self) -> List[QuantumCircuit]:
        """Return a list of experiment circuits, transpiled."""
        transpiled = super()._transpiled_circuits()

        if self._gate_error_ratio == False:
            # Gate errors are not computed, then counting ops is not necessary.
            return transpiled

        # Compute average basis gate numbers per Clifford operation
        # This is probably main source of performance regression.
        # This should be integrated into transpile pass in future.
        gate_counts_per_clifford = defaultdict(int)
        total_cliffs = np.sum(self.experiment_options.lengths)
        for circ in transpiled:
            for (qubits, instr), count in RBUtils.count_ops(circ, self.physical_qubits).items():
                if instr in ("measure", "reset", "delay", "barrier", "snapshot"):
                    continue
                # This is qubit aware count opts
                gate_counts_per_clifford[(qubits, instr)] += count / total_cliffs

        # Directly copy the value to experiment data metadata via instance state
        self._gate_counts_per_clifford = dict(gate_counts_per_clifford)

        return transpiled

    def _metadata(self):
        metadata = super()._metadata()
        # Store measurement level and meas return if they have been
        # set for the experiment
        for run_opt in ["meas_level", "meas_return"]:
            if hasattr(self.run_options, run_opt):
                metadata[run_opt] = getattr(self.run_options, run_opt)

        def _to_tuple(value):
            # For JSON serialization. The dict key is not string.
            if isinstance(value, dict):
                return tuple(value.items())
            return value

        metadata["gate_error_ratio"] = _to_tuple(self._gate_error_ratio)
        metadata["gate_counts_per_clifford"] = _to_tuple(self._gate_counts_per_clifford)
        return metadata
