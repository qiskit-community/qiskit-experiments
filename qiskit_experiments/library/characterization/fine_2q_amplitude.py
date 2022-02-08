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

from qiskit import QiskitError
from qiskit.circuit import QuantumCircuit, Gate
from qiskit.providers.backend import Backend

from qiskit_experiments.library.characterization.fine_amplitude import FineAmplitude
from qiskit_experiments.framework import Options


class FineTwoQubitAmplitude(FineAmplitude):
    r"""A fine amplitude experiment with all the options set for a two-qubit `R(theta)` gate.

    # section: overview

        :class:`FineTwoQubitAmplitude` is a subclass of :class:`FineAmplitude` and is used
        to set the appropriate values for the default options. The following example
        illustrates how this experiment can be used

        .. code:: python

            inst_map = InstructionScheduleMap()
            inst_map.add("rzx", qubits, my_rzx_schedule)

            fine_amp = FineTwoQubitAmplitude(qubits, RZXGate(np.pi/2), backend)
            fine_amp.set_transpile_options(inst_map=inst_map)

            exp_data = fine_amp.run()

        Here, the user provides the schedule for the `RZXGate` by passing it to the
        experiment using the instruction schedule map in the transpile options. The example
        above will create the circuits

        .. parsed-literal::

                 ┌────────────┐     ┌────────────┐
            q_0: ┤0           ├ ... ┤0           ├───
                 │  rzx(1.57) │     │  rzx(1.57) │┌─┐
            q_1: ┤1           ├ ... ┤1           ├┤M├
                 └────────────┘     └────────────┘└╥┘
            c: 1/══════════════ ...  ══════════════╩═
                                                   0

    """

    def __init__(self, qubits: Tuple[int, int], gate: Gate, backend: Optional[Backend] = None):
        """Initialize the experiment.

        Args:
            qubits: A length two tuple of the qubits on which to run.
            gate: A Qiskit Gate instruction with one parameter corresponding to the rotation angle.
            backend: The backend to run on.

        Raises:
            QiskitError: If the gate does not have exactly one parameter.
        """

        if len(gate.params) != 1:
            raise QiskitError(f"A gate with a single rotation angle is required.")

        angle = gate.params[0]
        label = gate.name + f"({angle})"
        super().__init__(qubits, Gate(gate.name, 2, [], label=label), backend=backend)
        # Set default analysis options
        self.analysis.set_options(
            angle_per_gate=angle,
            phase_offset=-np.pi,
            outcome="1",
        )
        self.set_transpile_options(basis_gates=[gate.name])

    @classmethod
    def _default_experiment_options(cls) -> Options:
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
        return QuantumCircuit(2, 1)

    def _add_measure(self, circuit: QuantumCircuit):
        """In this experiment we only measure the target qubit."""
        circuit.measure(1, 0)
