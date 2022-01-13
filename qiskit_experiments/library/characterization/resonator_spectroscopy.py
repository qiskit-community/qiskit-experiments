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

from typing import Iterable, Optional, Tuple

from qiskit import QuantumCircuit
from qiskit.circuit import Parameter, Gate
from qiskit.exceptions import QiskitError
from qiskit.providers import Backend
import qiskit.pulse as pulse

from qiskit_experiments.library.characterization import QubitSpectroscopy
from .analysis.resonator_spectroscopy_analysis import ResonatorSpectroscopyAnalysis


class ResonatorSpectroscopy(QubitSpectroscopy):
    """Perform spectroscopy on the readout resonator."""

    __spec_gate_name__ = "MSpec"

    def __init__(
        self,
        qubit: int,
        frequencies: Iterable[float],
        backend: Optional[Backend] = None,
        absolute: bool = True,
    ):
        """
        A spectroscopy experiment run by setting the frequency of the readout drive.
        The parameters of the GaussianSquare spectroscopy pulse can be specified at run-time.
        The spectroscopy pulse has the following parameters:
        - amp: The amplitude of the pulse must be between 0 and 1, the default is 0.1.
        - duration: The duration of the spectroscopy pulse in samples, the default is 1000 samples.
        - sigma: The standard deviation of the pulse, the default is duration / 4.
        - width: The width of the flat-top in the pulse, the default is 0, i.e. a Gaussian.

        Args:
            qubit: The qubit on which to run readout spectroscopy.
            frequencies: The frequencies to scan in the experiment, in Hz.
            backend: Optional, the backend to run the experiment on.
            absolute: Boolean to specify if the frequencies are absolute or relative to the
                qubit frequency in the backend.
        """
        super().__init__(qubit, frequencies, backend, absolute)
        self.analysis = ResonatorSpectroscopyAnalysis()

    @property
    def center_frequency(self) -> float:
        """Returns the center frequency of the experiment.

        Returns:
            The center frequency of the experiment.

        Raises:
            QiskitError: If the experiment does not have a backend set.
        """
        if self.backend is None:
            raise QiskitError("backend not set. Cannot call center_frequency.")

        return self.backend.defaults().meas_freq_est[self.physical_qubits[0]]

    def _template_circuit(self, freq_param) -> QuantumCircuit:
        """Return the template quantum circuit."""
        if self.backend is not None:
            cbits = self.backend.configuration().num_qubits
        else:
            cbits = 1

        circuit = QuantumCircuit(1, cbits)
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
