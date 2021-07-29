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

"""Spectroscopy experiment class."""

from typing import List, Optional, Tuple, Union

import numpy as np
import qiskit.pulse as pulse
from qiskit import QuantumCircuit
from qiskit.circuit import Gate, Parameter
from qiskit.exceptions import QiskitError
from qiskit.providers import Backend
from qiskit.providers.options import Options
from qiskit.qobj.utils import MeasLevel
from qiskit.utils import apply_prefix

from qiskit_experiments.framework import BaseExperiment
from qiskit_experiments.library.characterization.resonance_analysis import ResonanceAnalysis


class QubitSpectroscopy(BaseExperiment):
    """Class that runs spectroscopy by sweeping the qubit frequency.

    The circuits produced by spectroscopy, i.e.

    .. parsed-literal::

                   ┌────────────┐ ░ ┌─┐
              q_0: ┤ Spec(freq) ├─░─┤M├
                   └────────────┘ ░ └╥┘
        measure: 1/══════════════════╩═
                                     0

    have a spectroscopy pulse-schedule embedded in a spectroscopy gate. The
    pulse-schedule consists of a set frequency instruction followed by a GaussianSquare
    pulse. A list of circuits is generated, each with a different frequency "freq".
    """

    __analysis_class__ = ResonanceAnalysis
    __spec_gate_name__ = "Spec"

    @classmethod
    def _default_run_options(cls) -> Options:
        """Default options values for the experiment :meth:`run` method."""
        return Options(
            meas_level=MeasLevel.KERNELED,
            meas_return="single",
        )

    @classmethod
    def _default_experiment_options(cls) -> Options:
        """Default option values used for the spectroscopy pulse."""
        return Options(
            amp=0.1,
            duration=1024,
            sigma=256,
            width=0,
        )

    @classmethod
    def _default_analysis_options(cls) -> Options:
        """Default analysis options."""
        options = super()._default_analysis_options()
        options.db_parameters = {"freq": ("f01", "Hz")}
        options.normalization = True

        return options

    def __init__(
        self,
        qubit: int,
        frequencies: Union[List[float], np.array],
        unit: Optional[str] = "Hz",
        absolute: bool = True,
    ):
        """
        A spectroscopy experiment run by setting the frequency of the qubit drive.
        The parameters of the GaussianSquare spectroscopy pulse can be specified at run-time.
        The spectroscopy pulse has the following parameters:
        - amp: The amplitude of the pulse must be between 0 and 1, the default is 0.1.
        - duration: The duration of the spectroscopy pulse in samples, the default is 1000 samples.
        - sigma: The standard deviation of the pulse, the default is duration / 4.
        - width: The width of the flat-top in the pulse, the default is 0, i.e. a Gaussian.

        Args:
            qubit: The qubit on which to run spectroscopy.
            frequencies: The frequencies to scan in the experiment.
            unit: The unit in which the user specifies the frequencies. Can be one
                of 'Hz', 'kHz', 'MHz', 'GHz'. Internally, all frequencies will be converted
                to 'Hz'.
            absolute: Boolean to specify if the frequencies are absolute or relative to the
                qubit frequency in the backend.

        Raises:
            QiskitError: if there are less than three frequency shifts or if the unit is not known.

        """
        if len(frequencies) < 3:
            raise QiskitError("Spectroscopy requires at least three frequencies.")

        if unit == "Hz":
            self._frequencies = frequencies
        else:
            self._frequencies = [apply_prefix(freq, unit) for freq in frequencies]

        super().__init__([qubit])

        self._absolute = absolute

        if not self._absolute:
            self.set_analysis_options(xlabel="Frequency shift [Hz]")
        else:
            self.set_analysis_options(xlabel="Frequency [Hz]")

        self.set_analysis_options(ylabel="Signal [arb. unit]")

    def _spec_gate_schedule(
        self, backend: Optional[Backend] = None
    ) -> Tuple[pulse.ScheduleBlock, Parameter]:
        """Create the spectroscopy schedule."""
        freq_param = Parameter("frequency")
        with pulse.build(backend=backend, name="spectroscopy") as schedule:
            pulse.shift_frequency(freq_param, pulse.DriveChannel(self.physical_qubits[0]))
            pulse.play(
                pulse.GaussianSquare(
                    duration=self.experiment_options.duration,
                    amp=self.experiment_options.amp,
                    sigma=self.experiment_options.sigma,
                    width=self.experiment_options.width,
                ),
                pulse.DriveChannel(self.physical_qubits[0]),
            )
            pulse.shift_frequency(-freq_param, pulse.DriveChannel(self.physical_qubits[0]))

        return schedule, freq_param

    def _template_circuit(self, freq_param) -> QuantumCircuit:
        """Return the template quantum circuit."""
        circuit = QuantumCircuit(1)
        circuit.append(Gate(name=self.__spec_gate_name__, num_qubits=1, params=[freq_param]), (0,))
        circuit.measure_active()

        return circuit

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
                - If absolute frequencies are used but no backend is given.
                - If the backend configuration does not define dt.
        """
        if backend is None and self._absolute:
            raise QiskitError("Cannot run spectroscopy absolute to qubit without a backend.")

        # Create a template circuit
        sched, freq_param = self._spec_gate_schedule(backend)
        circuit = self._template_circuit(freq_param)
        circuit.add_calibration("Spec", (self.physical_qubits[0],), sched, params=[freq_param])

        # Create the circuits to run
        circs = []
        for freq in self._frequencies:

            freq_shift = freq
            if self._absolute:
                center_freq = backend.defaults().qubit_freq_est[self.physical_qubits[0]]
                freq_shift -= center_freq

            freq_shift = np.round(freq_shift, decimals=3)

            assigned_circ = circuit.assign_parameters({freq_param: freq_shift}, inplace=False)
            assigned_circ.metadata = {
                "experiment_type": self._type,
                "qubits": (self.physical_qubits[0],),
                "xval": np.round(freq, decimals=3),
                "unit": "Hz",
                "amplitude": self.experiment_options.amp,
                "duration": self.experiment_options.duration,
                "sigma": self.experiment_options.sigma,
                "width": self.experiment_options.width,
                "schedule": str(sched),
            }

            try:
                assigned_circ.metadata["dt"] = getattr(backend.configuration(), "dt")
            except AttributeError as no_dt:
                raise QiskitError("Dt parameter is missing in backend configuration") from no_dt

            circs.append(assigned_circ)

        return circs
