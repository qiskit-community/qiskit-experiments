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

from typing import Optional

from qiskit import QuantumCircuit
from qiskit.circuit import Gate, Parameter
from qiskit.exceptions import QiskitError
from qiskit.providers import Backend
import qiskit.pulse as pulse
from qiskit.providers.options import Options
from qiskit.pulse.transforms import block_to_schedule

from qiskit_experiments.characterization.qubit_spectroscopy import QubitSpectroscopy, SpectroscopyAnalysis


class ResonatorSpectroscopyAnalysis(SpectroscopyAnalysis):
    """Class to analysis resonator spectroscopy."""

    @classmethod
    def _default_options(cls):
        options = super()._default_options()
        options.dimensionality_reduction="ToAbs"
        return options


class ResonatorSpectroscopy(QubitSpectroscopy):
    """Perform spectroscopy on the readout resonator."""

    __analysis_class__ = ResonatorSpectroscopyAnalysis

    @classmethod
    def _default_experiment_options(cls) -> Options:
        """Default option values used for the spectroscopy pulse."""
        return Options(
            amp=0.1,
            duration=2688,
            sigma=64,
            width=2432,
        )

    def circuits(self, backend: Optional[Backend] = None):
        """Create the circuit for the spectroscopy experiment.

        The circuits are based on a GaussianSquare pulse and a frequency_shift instruction
        encapsulated in a gate.

        Args:
            backend: A backend object.

        Returns:
            circuits: The circuits that will run the spectroscopy experiment.

        Raises:
            QiskitError:
                - If relative frequencies are used but no backend was given.
                - If the backend configuration does not define dt.
        """
        if not backend and not self._absolute:
            raise QiskitError("Cannot run spectroscopy relative to resonator without a backend.")

        # Create a template circuit
        freq_param = Parameter("frequency")
        duration = self.experiment_options.duration
        qubit = self.physical_qubits[0]

        # TODO get this to work with circuits
        circuits = []
        for freq in self._frequencies:
            with pulse.build(backend=backend, name="spectroscopy") as circuit:
                pulse.set_frequency(freq, pulse.measure_channel(qubit))
                pulse.play(
                    pulse.GaussianSquare(duration, 0.1, 64, duration - 4 * 64),
                    pulse.measure_channel(qubit)
                )
                pulse.acquire(duration, qubit, pulse.MemorySlot(qubit))

            circuit.metadata = {
                "experiment_type": self._type,
                "qubit": self.physical_qubits[0],
                "xval": freq,
                "unit": "Hz",
                "amplitude": self.experiment_options.amp,
                "duration": self.experiment_options.duration,
                "sigma": self.experiment_options.sigma,
                "width": self.experiment_options.width,
                #"schedule": str(circuit),
                "meas_level": self.run_options.meas_level,
                "meas_return": self.run_options.meas_return,
            }

            try:
                circuit.metadata["dt"] = getattr(backend.configuration(), "dt")
            except AttributeError as no_dt:
                raise QiskitError("Dt parameter is missing in backend configuration") from no_dt

            circuits.append(block_to_schedule(circuit))

        return circuits
