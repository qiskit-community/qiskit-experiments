from typing import Union, Iterable, Optional, List

import numpy as np
from numpy.random import Generator, default_rng

from qiskit import QuantumCircuit, QiskitError, QuantumRegister, ClassicalRegister
from qiskit.ignis.characterization.characterization_utils import pad_id_gates, time_to_ngates
from qiskit.providers import Backend
from qiskit.quantum_info import Clifford
from qiskit.providers.options import Options
from qiskit.circuit import Gate

from qiskit_experiments.framework import BaseExperiment, ParallelExperiment
from qiskit_experiments.curve_analysis.data_processing import probability
from .rb_analysis import RBAnalysis
from .clifford_utils import CliffordUtils
from .rb_utils import RBUtils


class T2Hahn(BaseExperiment):
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

    # section: reference
        .. ref_arxiv:: 1 1009.3639
        .. ref_arxiv:: 2 1109.6887
        .. ref_arxiv:: 3 1712.06550

    """

    # Analysis class for experiment
    __analysis_class__ = T2Analysis  # need to add T2Analysis

    def __init__(
            self,
            qubits: Union[int, Iterable[int]],
            lengths: Union[List[int], np.array],  # need to change name?
            gate_time: float,
            n_echos: int = 1,
            phase_alt_echo: bool = False,
    ):
        """Initialize a T2 experiment with Hahn echo.

         Args:
                num_of_gates:
                    Each element of the list corresponds to a circuit.
                    `num_of_gates[i]` is the number of identity gates in each section
                    "t" of the pulse sequence in circuit no. i.
                    Must be in an increasing order.
                gate_time: time of running a single identity gate.
                qubits: indices of the qubits whose
                    T\ :sub:`2`:sup:`*`\ 's are to be measured.
                n_echos: number of echo gates (`X` or `Y`).
                phase_alt_echo: if True then alternate the echo between
                    `X` and `Y`.
        """
        # Initialize base experiment
        super().__init__(qubits)
        self._verify_parameters(qubits, lengths, n_echos,gate_time)

        # Set configurable options
        self.set_experiment_options(lengths=list(lengths), n_echos=n_echos, phase_alt_echo=phase_alt_echo)
        self.set_analysis_options(data_processor=probability(
            outcome="0" * self.num_qubits))  # Need to rewrite after making the analysis class


    def _verify_parameters(self, qubits, lengths, n_echos, gate_time):
        """Verify input correctness, raise QiskitError if needed"""
        if any(length <= 0 for length in lengths):
            raise QiskitError(
                f"The lengths list {lengths} should only contain " "positive elements."
            )
        if len(set(lengths)) != len(lengths):
            raise QiskitError(
                f"The lengths list {lengths} should not contain " "duplicate elements."
            )
        if any(lengths[idx - 1] >= lengths[idx] for idx in range(1, lengths)):
            raise QiskitError(f"The number of identity gates {lengths} should " "be increasing.")

        if any(qubit < 0 for qubit in qubits):
            raise QiskitError(f"The index of the qubits {qubits} should " "be non-negative.")

        if n_echos < 1:
            raise QiskitError(f"The number of echoes {n_echos} should " "be at least 1.")

        if gate_time <= 0:
            raise QiskitError(f"The gate time {gate_time} should " "be positive.")

    def _generate_circuits(self, num_of_gates: Union[List[int], np.array],
                        gate_time: float,
                        qubits: List[int],
                        n_echos: int = 1,
                        phase_alt_echo: bool = False):
        if n_echos < 1:
            raise ValueError('Must be at least one echo')

        xdata = 2 * gate_time * np.array(num_of_gates) * n_echos
        qr = QuantumRegister(max(qubits) + 1)
        cr = ClassicalRegister(len(qubits))
        circuits = []
        for circ_index, circ_length in enumerate(num_of_gates):
            circ = QuantumCircuit()
            circ.name = 't2circuit_' + str(circ_index) + '_0'
            for qind, qubit in enumerate(qubits):

                # First Y90 and Y echo
                circ.append(circ.ry(np.pi/4, [qr[qubit]]))  # Y90
                circ = pad_id_gates(circ, qr, qubit, circ_length)  # ids
                circ.y(qr[qubit])

                for echoid in range(n_echos - 1):  # repeat
                    circ = pad_id_gates(circ, qr, qubit, 2 * circ_length)  # ids
                    if phase_alt_echo and (not echoid % 2):  # optionally
                        circ.x(qr[qubit])  # X
                    else:
                        circ.y(qr[qubit])

                circ = pad_id_gates(circ, qr, qubit, circ_length)  # ids
                circ.append(circ.y, [qr[qubit]])  # Y90
            circ.barrier(qr)
            for qind, qubit in enumerate(qubits):
                circ.measure(qr[qubit], cr[qind])  # measure
            circuits.append(circ)

        return circuits, xdata
