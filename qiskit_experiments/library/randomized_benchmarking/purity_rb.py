# This code is part of Qiskit.
#
# (C) Copyright IBM 2026.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
"""
Purity RB Experiment class.
"""

from numbers import Integral
from typing import Union, Iterable, Optional, List, Sequence

import numpy as np
from numpy.random import Generator
from numpy.random.bit_generator import BitGenerator, SeedSequence

from qiskit import QuantumCircuit
from qiskit.quantum_info import Clifford
from qiskit.providers.backend import Backend
from qiskit.circuit import CircuitInstruction, Barrier

from .standard_rb import StandardRB
from .purity_rb_analysis import PurityRBAnalysis


SequenceElementType = Union[Clifford, Integral, QuantumCircuit]


class PurityRB(StandardRB):
    """An experiment to characterize the error rate of a gate set on a device
    using purity RB.

    # section: overview

    Randomized Benchmarking (RB) is an efficient and robust method
    for estimating the average error rate of a set of quantum gate operations.
    See `Qiskit Textbook
    <https://github.com/Qiskit/textbook/blob/main/notebooks/quantum-hardware/randomized-benchmarking.ipynb>`_
    for an explanation on the RB method.

    A standard RB experiment generates sequences of random Cliffords
    such that the unitary computed by the sequences is the identity.
    After running the sequences on a backend, it calculates the probabilities to get back to
    the ground state, fits an exponentially decaying curve, and estimates
    the Error Per Clifford (EPC), as described in Refs. [1, 2].

    Purity RB extends standard RB by appending post-rotations to the RB sequences
    to calculate Tr(rho^2), providing an alternative measure of gate fidelity [3].

    .. note::
        In 0.5.0, the default value of ``optimization_level`` in ``transpile_options`` changed
        from ``0`` to ``1`` for RB experiments. That may result in shorter RB circuits
        hence slower decay curves than before.

    # section: analysis_ref
        :class:`PurityRBAnalysis`

    # section: manual
        :doc:`/manuals/verification/randomized_benchmarking`

    # section: reference
        .. ref_arxiv:: 1 1009.3639
        .. ref_arxiv:: 2 1109.6887
        .. ref_arxiv:: 3 2302.10881
    """

    def __init__(
        self,
        physical_qubits: Sequence[int],
        lengths: Iterable[int],
        backend: Optional[Backend] = None,
        num_samples: int = 3,
        seed: Optional[Union[int, SeedSequence, BitGenerator, Generator]] = None,
        full_sampling: Optional[bool] = False,
    ):
        """Initialize a purity randomized benchmarking experiment.

        Args:
            physical_qubits: List of physical qubits for the experiment.
            lengths: A list of RB sequences lengths.
            backend: The backend to run the experiment on.
            num_samples: Number of samples to generate for each sequence length.
            seed: Optional, seed used to initialize ``numpy.random.default_rng``.
                  when generating circuits. The ``default_rng`` will be initialized
                  with this seed value every time :meth:`circuits` is called.
            full_sampling: If True all Cliffords are independently sampled for all lengths.
                           If False for sample of lengths longer sequences are constructed
                           by appending additional samples to shorter sequences.
                           The default is False.

        Raises:
            QiskitError: If any invalid argument is supplied.
        """
        # Initialize base experiment (RB)
        super().__init__(physical_qubits, lengths, backend, num_samples, seed, full_sampling)

        # override the analysis
        self.analysis = PurityRBAnalysis()
        self.analysis.set_options(outcome="0" * self.num_qubits)
        self.analysis.plotter.set_figure_options(
            xlabel="Clifford Length",
            ylabel="Purity",
        )

    def circuits(self) -> List[QuantumCircuit]:
        """Return a list of RB circuits.

        Returns:
            A list of :class:`QuantumCircuit`.
        """
        # Sample random Clifford sequences
        sequences = self._sample_sequences()
        # Convert each sequence into circuit and append the inverse to the end.
        # and the post-rotations
        circuits = self._sequences_to_circuits(sequences)
        # Add metadata for each circuit
        # trial links all from the same trial
        # needed for post processing the purity RB
        for circ_i, circ in enumerate(circuits):
            circ.metadata = {
                "xval": len(sequences[int(circ_i / 3**self.num_qubits)]),
                "trial": int(circ_i / 3**self.num_qubits),
                "group": "Clifford",
            }
        return circuits

    def _sequences_to_circuits(
        self, sequences: List[Sequence[SequenceElementType]]
    ) -> List[QuantumCircuit]:
        """Convert an RB sequence into circuit and append the inverse to the end and
        then the post rotations for purity RB

        Returns:
            A list of purity RB circuits.
        """
        synthesis_opts = self._get_synthesis_options()

        # post rotations as cliffords
        post_rot = []
        for i in range(3**self.num_qubits):
            ##find clifford
            qc = QuantumCircuit(self.num_qubits)
            for j in range(self.num_qubits):
                qg_ind = np.mod(int(i / 3**j), 3)
                if qg_ind == 1:
                    qc.sx(j)
                elif qg_ind == 2:
                    qc.sdg(j)
                    qc.sx(j)
                    qc.s(j)

            post_rot.append(self._to_instruction(Clifford(qc), synthesis_opts))

        # Circuit generation
        circuits = []
        for i, seq in enumerate(sequences):
            if (
                self.experiment_options.full_sampling
                or i % len(self.experiment_options.lengths) == 0
            ):
                prev_elem, prev_seq = (
                    self._StandardRB__identity_clifford(),
                    [],
                )  # pylint: disable=no-member

            circ = QuantumCircuit(self.num_qubits)
            for elem in seq:
                circ.append(self._to_instruction(elem, synthesis_opts), circ.qubits)
                circ._append(CircuitInstruction(Barrier(self.num_qubits), circ.qubits))

            # Compute inverse, compute only the difference from the previous shorter sequence
            prev_elem = self._StandardRB__compose_clifford_seq(  # pylint: disable=no-member
                prev_elem, seq[len(prev_seq) :]
            )
            prev_seq = seq
            inv = self._StandardRB__adjoint_clifford(prev_elem)  # pylint: disable=no-member

            circ.append(self._to_instruction(inv, synthesis_opts), circ.qubits)

            # copy the circuit and apply post rotations
            for j in range(3**self.num_qubits):
                circ2 = circ.copy()
                circ2.append(post_rot[j], circ.qubits)
                circ2.measure_all()  # includes insertion of the barrier before measurement
                circuits.append(circ2)

        return circuits
