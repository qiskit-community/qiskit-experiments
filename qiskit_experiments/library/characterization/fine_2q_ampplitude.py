# This code is part of Qiskit.
#
# (C) Copyright IBM 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Fine amplitude characterization experiment for two-qubit gates."""

from typing import Optional, Tuple
import numpy as np

from qiskit.circuit import QuantumCircuit
from qiskit.circuit.library import RZXGate
from qiskit.providers.backend import Backend

from qiskit_experiments.library.characterization.fine_amplitude import FineAmplitude


class FineTwoQubitAmplitude(FineAmplitude):
    r"""Experiment to characterize the amplitude of the two-qubit CR gate.

    # section: overview

        This experiment is designed to measure over and under rotations in the
        amplitude of the cross-resonance gate (CR).
    """

    @classmethod
    def _default_experiment_options(cls):
        """
        Experiment options:
            repetitions (List[int]): A list of the number of times that the gate is repeated.
        """
        options = super()._default_experiment_options()

        options.repetitions = [0, 1, 2, 3, 4, 5, 7, 9, 11, 13]
        options.add_xp_circuit = False

        return options

    def _pre_circuit(self) -> QuantumCircuit:
        """Return the quantum circuit to which the repeated gate sequence will be appended.

        Sub-classes can override this class if they wish to add gates before the main gate
        sequence.
        """
        return QuantumCircuit(2, 2)

    def _add_measure(self, circuit: QuantumCircuit):
        """In this experiment we only measure the target qubit."""
        circuit.measure(1, 1)


class FineRZXAmplitude(FineTwoQubitAmplitude):
    r"""A fine amplitude experiment with all the options set for the RZX(pi/2) gate.

    # section: overview

        :class:`FineXAmplitude` is a subclass of :class:`FineAmplitude` and is used to set
        the appropriate values for the default options.
    """

    def __init__(self, qubits: Tuple[int, int], backend: Optional[Backend] = None):
        """Initialize the experiment."""
        super().__init__(qubits, RZXGate(np.pi/2), backend=backend)
        # Set default analysis options
        self.analysis.set_options(
            angle_per_gate=np.pi,
            phase_offset=np.pi / 2,
            amp=1,
        )
