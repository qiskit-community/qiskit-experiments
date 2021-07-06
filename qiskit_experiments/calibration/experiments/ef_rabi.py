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

from qiskit import QuantumCircuit
from qiskit.circuit import Gate, Parameter
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

    def __init__(self, qubit: int, frequency_shift: float):
        """Setup a Rabi experiment between levels 1 and 2 on the given qubit.

        Args:
            qubit: The qubit on which to run the Rabi experiment.
            frequency_shift: The frequency by which the Rabi pulse between 1 and 2 is shifted
        """

        super().__init__(qubit)

        # create schedule of the Rabi gate with a shifted pulse frequency
        amp_param = Parameter("amp")
        with pulse.build(name="rabi") as sched:
            pulse.shift_frequency(frequency_shift, pulse.DriveChannel(self.physical_qubits[0]))
            pulse.play(
                pulse.Gaussian(
                    duration=self.experiment_options.duration,
                    amp=amp_param,
                    sigma=self.experiment_options.sigma,
                ),
                pulse.DriveChannel(self.physical_qubits[0]),
            )
            pulse.shift_frequency(-frequency_shift, pulse.DriveChannel(self.physical_qubits[0]))

        self.set_experiment_options(schedule=sched)

    def _template_circuit(self, amp_param) -> QuantumCircuit:
        """Return the template quantum circuit."""
        circuit = QuantumCircuit(1)
        circuit.x(0)
        circuit.append(Gate(name=self.__rabi_gate_name__, num_qubits=1, params=[amp_param]), (0,))
        circuit.measure_active()

        return circuit
