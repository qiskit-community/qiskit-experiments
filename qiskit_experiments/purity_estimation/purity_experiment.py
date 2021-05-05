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
from typing import Union, Optional, Iterable, List
from numpy.random import default_rng, Generator

from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import Clifford
from qiskit_experiments.base_experiment import BaseExperiment
from .purity_analysis import PurityEstimationAnalysis


class PurityEstimation(BaseExperiment):
    r"""Reduced density matrix purity estimation experiment.

    This estimates the purity of an N-qubit reduced density matrix
    :math:`\rho_A` using the identity

    .. math::
        Tr[\rho^2] = 2^N \sum_{i,j}\sum_{C} (-2)^{w_{i,j}} p_i(C) p_j(C)

    where :math:`w_{i,j}` is the Hamming weight distance between two
    bitstrings *i* and *j*, and :math:`p_i(C)` is the measurement outcome
    probability

    .. math::
        p_i(C) &= \langle i | C\rho C^\dagger | i \rangle \\
        C &= \bigotimes_{i=1}^N C_i

    and the sum is taken over all tensor products of 1-qubit Cliffords
    :math:`C = \bigotimes_{i=1}^N C_i` applied before Z-basis measurements.

    Note that while there are 24 single-qubit Cliffords there are only 6 that need to be sampled uniformly from
    to generate all possible measurement outcome distributions and implement averaging over the full unitary
    2-design.
    """

    __analysis_class__ = PurityEstimationAnalysis

    # Pre-synthesized 1-qubit Clifford circuits
    _CLIFFORD1_INST = [
        Clifford.from_dict({"stabilizer": stab, "destabilizer": destab}).to_instruction()
        for stab, destab in [
            ["+X", "+Z"],
            ["+X", "-Z"],
            ["+X", "+Y"],
            ["+Y", "+X"],
            ["+Z", "+X"],
            ["-Z", "-X"],
        ]
    ]

    def __init__(
        self,
        circuit: Union[QuantumCircuit, "InstructionLike"],
        qubits: Optional[Iterable[int]] = None,
        meas_qubits: Optional[Iterable[int]] = None,
        num_samples: Optional[int] = None,
        seed: Optional[Union[Generator, int]] = None,
    ):
        """Initialize new state purity estimation experiment

        Args:
            circuit: the quantum state preparation circuit. If not a quantum
                circuit it must be a class that can be appended to a quantum
                circuit.
            qubits: Optional, the physical qubits for the initial state circuit.
            meas_qubits: Optional, the qubits to be measured. These should refer
                to the logical qubits in the state circuit. If None all qubits
                in the state circuit will be measured.
            num_samples: Optional, the number of random measurement circuit samples to
                generate. If not specified the full set of :math:`6^n` circuits will
                be generated.
            seed: Optional, the random number generator, or generator seed to
                use for sampling.
        """
        num_qubits = circuit.num_qubits

        # Get physical qubits
        if qubits is None:
            qubits = num_qubits

        # Get measured qubits
        if meas_qubits is None:
            self._meas_qubits = tuple(range(num_qubits))
        else:
            self._meas_qubits = tuple(meas_qubits)

        # Get preparation circuit
        if isinstance(circuit, QuantumCircuit):
            prep_circuit = circuit
        else:
            # Convert input to a circuit
            prep_circuit = QuantumCircuit(num_qubits)
            prep_circuit.append(circuit, range(num_qubits))
        self._circuit = prep_circuit

        # Number of samples
        if num_samples is None:
            self._num_samples = 6 ** len(self._meas_qubits)
        else:
            self._num_samples = num_samples

        # RNG
        if isinstance(seed, Generator):
            self._rng = seed
        else:
            self._rng = default_rng(seed)

        super().__init__(qubits, circuit_options=["num_samples"])

    # pylint: disable = arguments-differ
    def circuits(self, backend=None, num_samples=None):

        # Get qubits and clbits
        num_meas = len(self._meas_qubits)
        total_clbits = self._circuit.num_clbits + num_meas
        circ_qubits = list(range(self._circuit.num_qubits))
        circ_clbits = list(range(self._circuit.num_clbits))
        meas_clbits = list(range(self._circuit.num_clbits, total_clbits))

        # Generate indices of measurement circuits
        max_size = 6 ** num_meas
        if num_samples is None:
            num_samples = self._num_samples
        index_lst = self._sampler(num_samples, num_meas)

        # Build circuits
        circuits = []
        for index in index_lst:
            # Add prep circuit
            circ = QuantumCircuit(self.num_qubits, total_clbits)
            circ.append(self._circuit, circ_qubits, circ_clbits)

            # Add 1-qubit random Cliffords
            for i in range(num_meas):
                circ.append(self._CLIFFORD1_INST[index[i]], [self._meas_qubits[i]])

            # Measurement
            circ.measure(self._meas_qubits, meas_clbits)

            # Add metadata
            circ.metadata = {
                "experiment_type": self._type,
                "qubits": self.physical_qubits,
                "clbits": meas_clbits,
                "index": list(index),
            }
            circuits.append(circ)
        return circuits

    def _sampler(self, num_samples, num_meas):
        max_size = 6 ** num_meas
        if max_size <= 2 ** 63:
            # We sample without replacement
            samples = self._rng.choice(max_size, size=num_samples, replace=False)
            return [self._int2indices(i) for i in samples]
        else:
            # We can't use numpy random choice without replacement since range
            # is larger than 64-bit ints. Hence we sample with replacement and
            # rely on the sample space being so large that we are extremely
            # unlikely to sample the same index twice
            return self._rng.choice(6, size=(num_samples, num_meas), replace=True)

    def _int2indices(self, i: int) -> List[int]:
        """Convert an integer to list of indices"""
        size = len(self._meas_qubits)
        if i == 0:
            return size * [0]
        indices = []
        while i:
            indices.append(int(i % 6))
            i //= 6
        indices = indices[::-1]
        return (size - len(indices)) * [0] + indices
