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

import warnings
from typing import Optional
import numpy as np

from qiskit.providers import Backend
from qiskit.providers.options import Options
from qiskit import QuantumCircuit
from qiskit.circuit import Gate, Parameter
import qiskit.pulse as pulse

from qiskit_experiments.calibration.experiments import Rabi


class EFRabi(Rabi):
    """An experiment that scans the amplitude of a pulse to calibrate rotations between 1 and 2.

    This experiment is similar to the Rabi experiment between the ground and first excited state.
    The difference is in the initial X gate used to populate the first excited state. The Rabi pulse
    is then applied on the 1 <-> 2 transition (sometimes also labeled the e <-> f transition) which
    implies that frequency shift instructions are used. The necessary frequency shift (typically the
    qubit anharmonicity) should be specified through the experiment options.

    The circuits are of the form:

    .. parsed-literal::

                   ┌───┐┌───────────┐ ░ ┌─┐
              q_0: ┤ X ├┤ Rabi(amp) ├─░─┤M├
                   └───┘└───────────┘ ░ └╥┘
        measure: 1/══════════════════════╩═
                                         0
    """

    @classmethod
    def _default_experiment_options(cls) -> Options:
        """Default values for the pulse if no schedule is given.

        Users can set a schedule by doing

        .. code-block::

            ef_rabi.set_experiment_options(schedule=rabi_schedule)

        """
        return Options(
            duration=160,
            sigma=40,
            amplitudes=np.linspace(-0.95, 0.95, 51),
            schedule=None,
            normalization=True,
            frequency_shift=0.0,
        )

    def _default_gate_schedule(self, backend: Optional[Backend] = None):
        """Create the default schedule for the EFRabi gate with a frequency shift to the 1-2
        transition."""
        amp = Parameter("amp")
        with pulse.build(backend=backend, name="rabi") as default_schedule:

            if self.experiment_options.frequency_shift == 0.0:
                warnings.warn(
                    "With no applied frequency shift, this experiment will drive Rabi "
                    "oscillations between levels 0 and 1. Use "
                    "ef_rabi.set_experiment_options(frequency_shift=anharm), where anharm is the qubit "
                    "anharmonicity, to drive 1-2 Rabi oscillations."
                )
            pulse.shift_frequency(
                self.experiment_options.frequency_shift, pulse.DriveChannel(self.physical_qubits[0])
            )
            pulse.play(
                pulse.Gaussian(
                    duration=self.experiment_options.duration,
                    amp=amp,
                    sigma=self.experiment_options.sigma,
                ),
                pulse.DriveChannel(self.physical_qubits[0]),
            )
            pulse.shift_frequency(
                -self.experiment_options.frequency_shift,
                pulse.DriveChannel(self.physical_qubits[0]),
            )

        return default_schedule

    def _template_circuit(self, amp_param) -> QuantumCircuit:
        """Return the template quantum circuit."""
        circuit = QuantumCircuit(1)
        circuit.x(0)
        circuit.append(Gate(name=self.__rabi_gate_name__, num_qubits=1, params=[amp_param]), (0,))
        circuit.measure_active()

        return circuit
