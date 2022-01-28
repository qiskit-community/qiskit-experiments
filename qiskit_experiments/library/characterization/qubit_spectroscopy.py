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

from typing import Iterable, Optional, Tuple

import numpy as np
import qiskit.pulse as pulse
from qiskit import QuantumCircuit
from qiskit.circuit import Gate, Parameter
from qiskit.exceptions import QiskitError
from qiskit.providers import Backend
from qiskit.qobj.utils import MeasLevel

from qiskit_experiments.framework import BaseExperiment, Options
from qiskit_experiments.curve_analysis import ResonanceAnalysis


class QubitSpectroscopy(BaseExperiment):
    """Class that runs spectroscopy by sweeping the qubit frequency.

    # section: overview
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

    # section: analysis_ref
        :py:class:`~qiskit_experiments.curve_analysis.ResonanceAnalysis`
    """

    __spec_gate_name__ = "Spec"

    @classmethod
    def _default_run_options(cls) -> Options:
        """Default options values for the experiment :meth:`run` method."""
        options = super()._default_run_options()

        options.meas_level = MeasLevel.KERNELED
        options.meas_return = "single"

        return options

    @classmethod
    def _default_experiment_options(cls) -> Options:
        """Default option values used for the spectroscopy pulse.

        Experiment Options:
            amp (float): The amplitude of the spectroscopy pulse. Defaults to 0.1.
            duration (int): The duration of the spectroscopy pulse. Defaults to 1024 samples.
            sigma (float): The standard deviation of the flanks of the spectroscopy pulse.
                Defaults to 256.
            width (int): The width of the flat-top part of the GaussianSquare pulse.
                Defaults to 0.
        """
        options = super()._default_experiment_options()

        options.amp = 0.1
        options.duration = 1024
        options.sigma = 256
        options.width = 0

        return options

    def __init__(
        self,
        qubit: int,
        frequencies: Iterable[float],
        backend: Optional[Backend] = None,
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
            frequencies: The frequencies to scan in the experiment, in Hz.
            backend: Optional, the backend to run the experiment on.
            absolute: Boolean to specify if the frequencies are absolute or relative to the
                qubit frequency in the backend.

        Raises:
            QiskitError: if there are less than three frequency shifts.

        """
        super().__init__([qubit], analysis=ResonanceAnalysis(), backend=backend)

        if len(frequencies) < 3:
            raise QiskitError("Spectroscopy requires at least three frequencies.")

        self._frequencies = frequencies
        self._absolute = absolute

        if not self._absolute:
            self.analysis.set_options(xlabel="Frequency shift")
        else:
            self.analysis.set_options(xlabel="Frequency")

        self.analysis.set_options(ylabel="Signal [arb. unit]")

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

    def circuits(self):
        """Create the circuit for the spectroscopy experiment.

        The circuits are based on a GaussianSquare pulse and a frequency_shift instruction
        encapsulated in a gate.

        Returns:
            circuits: The circuits that will run the spectroscopy experiment.

        Raises:
            QiskitError:
                - If absolute frequencies are used but no backend is given.
                - If the backend configuration does not define dt.
            AttributeError: If backend to run on does not contain 'dt' configuration.
        """
        if self.backend is None and self._absolute:
            raise QiskitError("Cannot run spectroscopy absolute to qubit without a backend.")

        # Create a template circuit
        sched, freq_param = self._spec_gate_schedule(self.backend)
        circuit = self._template_circuit(freq_param)
        circuit.add_calibration("Spec", (self.physical_qubits[0],), sched, params=[freq_param])

        # Get dt
        try:
            dt_factor = getattr(self.backend.configuration(), "dt")
        except AttributeError as no_dt:
            raise AttributeError("dt parameter is missing in backend configuration") from no_dt

        # Get center frequency from backend
        if self._absolute:
            center_freq = self.backend.defaults().qubit_freq_est[self.physical_qubits[0]]
        else:
            center_freq = None

        # Create the circuits to run
        circs = []
        for freq in self._frequencies:
            freq_shift = freq
            if self._absolute:
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
                "dt": dt_factor,
            }

            circs.append(assigned_circ)

        return circs
