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

"""Spectroscopy experiment class for resonators."""

from typing import Optional, Iterable, Tuple

from qiskit import QuantumCircuit
from qiskit.circuit import Parameter, Gate
from qiskit.providers import Backend
import qiskit.pulse as pulse
from qiskit.providers.options import Options
from qiskit.pulse import ScheduleBlock

from qiskit_experiments.library.characterization import QubitSpectroscopy
from qiskit_experiments.curve_analysis import ResonanceAnalysis


class ResonatorSpectroscopyAnalysis(ResonanceAnalysis):
    """Class to analysis resonator spectroscopy."""

    @classmethod
    def _default_options(cls):
        options = super()._default_options()
        options.dimensionality_reduction="ToAbs"
        return options


class ResonatorSpectroscopy(QubitSpectroscopy):
    """Perform spectroscopy on the readout resonator."""

    __spec_gate_name__ = "MSpec"

    def _template_circuit(self, freq_param) -> QuantumCircuit:
        """Return the template quantum circuit."""
        circuit = QuantumCircuit(1)
        circuit.append(Gate(name=self.__spec_gate_name__, num_qubits=1, params=[freq_param]), (0,))

        return circuit

    def _spec_gate_schedule(
        self, backend: Optional[Backend] = None
    ) -> Tuple[pulse.ScheduleBlock, Parameter]:
        """Create the spectroscopy schedule."""

        qubit = self.physical_qubits[0]

        freq_param = Parameter("frequency")

        with pulse.build(backend=backend, name="spectroscopy") as schedule:
            pulse.shift_frequency(freq_param, pulse.MeasureChannel(qubit))
            pulse.play(
                pulse.GaussianSquare(
                    duration=self.experiment_options.duration,
                    amp=self.experiment_options.amp,
                    sigma=self.experiment_options.sigma,
                    width=self.experiment_options.width,
                ),
                pulse.MeasureChannel(qubit),
            )
            pulse.acquire(self.experiment_options.duration, qubit, pulse.MemorySlot(qubit))

        return schedule, freq_param
