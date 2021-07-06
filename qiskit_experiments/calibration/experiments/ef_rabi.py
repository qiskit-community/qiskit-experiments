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

"""Rabi amplitude experiment for the e-f transition."""

from typing import Optional, Tuple

from qiskit import QiskitError, QuantumCircuit
from qiskit.circuit import Gate, Parameter
from qiskit.providers import Backend
import qiskit.pulse as pulse

from qiskit_experiments.calibration.experiments import Rabi


class EFRabi(Rabi):
    """An experiment that scans the amplitude of a pulse to calibrate rotations between 1 and 2.

    The circuits are of the form:

    .. parsed-literal::

                   ┌───┐┌───────────┐ ░ ┌─┐
              q_0: ┤ X ├┤ Rabi(amp) ├─░─┤M├
                   └───┘└───────────┘ ░ └╥┘
        measure: 1/══════════════════════╩═
                                         0
    """

    def __init__(self, qubit: int, frequency: float, absolute: bool = True):
        """Setup a Rabi experiment on the given qubit.

        Args:
            qubit: The qubit on which to run the Rabi experiment.
            frequency: The frequency at which the Rabi pulse between 1 and 2 is played
            absolute: Boolean to specify if the frequency is absolute or relative to the
                qubit frequency in the backend.
        """

        super().__init__(qubit)

        self._frequency = frequency
        self._absolute = absolute

    def _rabi_gate_schedule(
        self, backend: Optional[Backend] = None
    ) -> Tuple[pulse.Schedule, Parameter]:
        """Create the rabi schedule."""

        if backend is None and self._absolute:
            raise QiskitError("Cannot determine frequency absolute to qubit without a backend.")

        freq_shift = self._frequency
        if self._absolute:
            center_freq = backend.defaults().qubit_freq_est[self.physical_qubits[0]]
            freq_shift -= center_freq

        amp_param = Parameter("amp")
        with pulse.build(backend=backend, name="rabi") as schedule:

            pulse.shift_frequency(freq_shift, pulse.DriveChannel(self.physical_qubits[0]))
            pulse.play(
                pulse.Gaussian(
                    duration=self.experiment_options.duration,
                    amp=amp_param,
                    sigma=self.experiment_options.sigma,
                ),
                pulse.DriveChannel(self.physical_qubits[0]),
            )
            pulse.shift_frequency(-freq_shift, pulse.DriveChannel(self.physical_qubits[0]))

        return schedule, amp_param

    def _template_circuit(self, amp_param) -> QuantumCircuit:
        """Return the template quantum circuit."""
        circuit = QuantumCircuit(1)
        circuit.x(0)
        circuit.append(Gate(name=self.__rabi_gate_name__, num_qubits=1, params=[amp_param]), (0,))
        circuit.measure_active()

        return circuit
