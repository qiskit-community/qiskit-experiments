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
from typing import Union, Iterable, Optional, List, Sequence

from numpy.random import Generator, default_rng
from numpy.random.bit_generator import BitGenerator, SeedSequence

from qiskit import QuantumCircuit
from qiskit.circuit import Instruction
from qiskit.quantum_info import Clifford
from qiskit.exceptions import QiskitError
from qiskit.providers.backend import Backend
from qiskit.compiler import transpile

from .rb_experiment import StandardRB
from .interleaved_rb_analysis import InterleavedRBAnalysis
from .clifford_utils import CliffordUtils


class InterleavedRB(StandardRB):
    """Interleaved randomized benchmarking experiment.

    # section: overview
        Interleaved Randomized Benchmarking (RB) is a method
        to estimate the average error-rate of a certain quantum gate.

        An interleaved RB experiment generates a standard RB sequences of random Cliffords
        and another sequence with the interleaved given gate.
        After running the two sequences on a backend, it calculates the probabilities to get back to
        the ground state, fits the two exponentially decaying curves, and estimates
        the interleaved gate error. See Ref. [1] for details.

    # section: analysis_ref
        :py:class:`InterleavedRBAnalysis`

    # section: reference
        .. ref_arxiv:: 1 1203.4550

    """

    def __init__(
        self,
        interleaved_element: Union[QuantumCircuit, Instruction, Clifford],
        qubits: Sequence[int],
        lengths: Iterable[int],
        backend: Optional[Backend] = None,
        num_samples: int = 3,
        seed: Optional[Union[int, SeedSequence, BitGenerator, Generator]] = None,
        full_sampling: bool = False,
    ):
        """Initialize an interleaved randomized benchmarking experiment.

        Args:
            interleaved_element: The element to interleave,
                    given either as a group element or as an instruction/circuit
            qubits: list of physical qubits for the experiment.
            lengths: A list of RB sequences lengths.
            backend: The backend to run the experiment on.
            num_samples: Number of samples to generate for each
                         sequence length
            seed: Optional, seed used to initialize ``numpy.random.default_rng``.
                  when generating circuits. The ``default_rng`` will be initialized
                  with this seed value everytime :meth:`circuits` is called.
            full_sampling: If True all Cliffords are independently sampled for
                           all lengths. If False for sample of lengths longer
                           sequences are constructed by appending additional
                           Clifford samples to shorter sequences.
        """

        super().__init__(
            qubits,
            lengths,
            backend=backend,
            num_samples=num_samples,
            seed=seed,
            full_sampling=full_sampling,
        )
        self._set_interleaved_element(interleaved_element)
        self._transpiled_interleaved_elem = None
        self.analysis = InterleavedRBAnalysis()
        self.analysis.set_options(outcome="0" * self.num_qubits)

    def circuits(self) -> List[QuantumCircuit]:
        """Return a list of RB circuits.

        Returns:
            A list of :class:`QuantumCircuit`.

        Raises:
            QiskitError: if basis_gates is not set in transpile_options nor in backend configuration.
        """
        rng = default_rng(seed=self.experiment_options.seed)
        circuits = []
        if not hasattr(self.transpile_options, "basis_gates"):
            if self.backend.configuration.basis_gates:
                self.set_transpile_options(basis_gates=self.backend.configuration.basis_gates)
            else:
                raise QiskitError("transpile_options.basis_gates must be set for rb_experiment")
        if self._clifford_utils is None:
            self._clifford_utils = CliffordUtils(self.num_qubits, self.transpile_options.basis_gates)

        for _ in range(self.experiment_options.num_samples):
            self._set_transpiled_interleaved_element()
            std_circuits, int_circuits = self._build_rb_circuits(
                self.experiment_options.lengths,
                rng,
                interleaved_element=self._transpiled_interleaved_elem,
            )
            circuits += std_circuits
            circuits += int_circuits
        return circuits

    def _set_interleaved_element(self, interleaved_element):
        """Handle the various types of the interleaved element

        Args:
            interleaved_element: The element to interleave

        Raises:
            QiskitError: if there is no known conversion of interleaved_element
            to a Clifford group element
        """
        try:
            interleaved_element_op = Clifford(interleaved_element)
            self._interleaved_element = (interleaved_element, interleaved_element_op)
        except QiskitError as error:
            raise QiskitError(
                "Interleaved element {} could not be converted to Clifford element".format(
                    interleaved_element.name
                )
            ) from error

    def _set_transpiled_interleaved_element(self):
        """
        Create the transpiled interleaved element. If it is a single gate,
        create a circuit comprising this gate.
        """
        if not isinstance(self._interleaved_element, QuantumCircuit):
            if self.num_qubits == 1:
                qc_interleaved = QuantumCircuit(1, 1)
                qubits = [0]
            else:
                qc_interleaved = QuantumCircuit(2, 2)
                qubits = [0, 1]

            qc_interleaved.append(self._interleaved_element[0], qubits)
            self._transpiled_interleaved_elem = qc_interleaved
        else:
            qc_interleaved = self._interleaved_element
        if hasattr(self.transpile_options, "basis_gates"):
            basis_gates = self.transpile_options.basis_gates
        else:
            basis_gates = None
        self._transpiled_interleaved_elem = transpile(
            circuits=qc_interleaved, optimization_level=1, basis_gates=basis_gates
        )
