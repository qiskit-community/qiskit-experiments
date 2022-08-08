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
import numpy as np

from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.exceptions import QiskitError
from qiskit.providers import Backend
import qiskit.pulse as pulse

from qiskit_experiments.framework import Options, BackendData
from qiskit_experiments.library.characterization.spectroscopy import Spectroscopy
from .analysis.resonator_spectroscopy_analysis import ResonatorSpectroscopyAnalysis


class ResonatorSpectroscopy(Spectroscopy):
    """Perform spectroscopy on the readout resonator.

    # section: overview
        This experiment does spectroscopy on the readout resonator. It applies the following
        circuit

        .. parsed-literal::

                 ┌─┐
              q: ┤M├
                 └╥┘
            c: 1/═╩═
                  0

        where a spectroscopy pulse is attached to the measurement instruction.

        Side note: when doing readout resonator spectroscopy, each measured IQ point has a
        frequency dependent phase. Close to the resonance, the IQ points start rotating around
        in the IQ plan. This effect must be accounted for in the data processing to produce a
        meaningful signal. The default data processing workflow will therefore reduce the two-
        dimensional IQ data to one-dimensional data using the magnitude of each IQ point.

        # section: warning
            Some backends may not have the required functionality to properly support resonator
            spectroscopy experiments. The experiment may not work or the resulting resonance
            may not properly reflect the properties of the readout resonator.

    # section: example

        The resonator spectroscopy experiment can be run by doing:

        .. code:: python

            qubit = 1
            spec = ResonatorSpectroscopy(qubit, backend)
            exp_data = spec.run().block_for_results()
            exp_data.figure(0)

        This will measure the resonator attached to qubit 1 and report the resonance frequency
        as well as the kappa, i.e. the line width, of the resonator.

    # section: analysis_ref
        :py:class:`ResonatorSpectroscopyAnalysis`

    # section: see_also
        qiskit_experiments.library.characterization.qubit_spectroscopy.QubitSpectroscopy
    """

    @classmethod
    def _default_experiment_options(cls) -> Options:
        """Default option values used for the spectroscopy pulse.

        All units of the resonator spectroscopy experiment are given in seconds.

        Experiment Options:
            amp (float): The amplitude of the spectroscopy pulse. Defaults to 1 and must
                be between 0 and 1.
            duration (float): The duration in seconds of the spectroscopy pulse.
            sigma (float): The standard deviation of the spectroscopy pulse in seconds.
            width (float): The width of the flat-top part of the GaussianSquare pulse in
                seconds. Defaults to 0.
        """
        options = super()._default_experiment_options()

        options.amp = 1
        options.duration = 480e-9
        options.sigma = 60e-9
        options.width = 360e-9

        return options

    def __init__(
        self,
        qubit: int,
        backend: Optional[Backend] = None,
        frequencies: Optional[Iterable[float]] = None,
        absolute: bool = True,
        **experiment_options,
    ):
        """Initialize a resonator spectroscopy experiment.

        A spectroscopy experiment run by setting the frequency of the readout drive.
        The parameters of the GaussianSquare spectroscopy pulse can be specified at run-time
        through the experiment options.

        Args:
            qubit: The qubit on which to run readout spectroscopy.
            backend: Optional, the backend to run the experiment on.
            frequencies: The frequencies to scan in the experiment, in Hz. The default values
                range from -20 MHz to 20 MHz in 51 steps. If the ``absolute`` variable is
                set to True then a center frequency obtained from the backend's defaults is
                added to each value of this range.
            absolute: Boolean to specify if the frequencies are absolute or relative to the
                resonator frequency in the backend. The default value is True.
            experiment_options: Key word arguments used to set the experiment options.

        Raises:
            QiskitError: if no frequencies are given and absolute frequencies are desired and
                no backend is given.
        """
        analysis = ResonatorSpectroscopyAnalysis()

        if frequencies is None:
            frequencies = np.linspace(-20.0e6, 20.0e6, 51)

            if absolute:
                if backend is None:
                    raise QiskitError(
                        "Cannot automatically compute absolute frequencies without a backend."
                    )

                center_freq = BackendData(backend).meas_freqs[qubit]
                frequencies += center_freq

        super().__init__(qubit, frequencies, backend, absolute, analysis, **experiment_options)

    @property
    def _backend_center_frequency(self) -> float:
        """Returns the center frequency of the experiment.

        Returns:
            The center frequency of the experiment.

        Raises:
            QiskitError: If the experiment does not have a backend set.
        """
        if self.backend is None:
            raise QiskitError("backend not set. Cannot call center_frequency.")

        return self._backend_data.meas_freqs[self.physical_qubits[0]]

    def _template_circuit(self) -> QuantumCircuit:
        """Return the template quantum circuit."""
        circuit = QuantumCircuit(1, 1)
        circuit.measure(0, 0)

        return circuit

    def _schedule(self) -> Tuple[pulse.ScheduleBlock, Parameter]:
        """Create the spectroscopy schedule."""

        dt, granularity = self._dt, self._granularity

        duration = int(granularity * (self.experiment_options.duration / dt // granularity))
        sigma = granularity * (self.experiment_options.sigma / dt // granularity)
        width = granularity * (self.experiment_options.width / dt // granularity)

        qubit = self.physical_qubits[0]

        freq_param = Parameter("frequency")

        with pulse.build(backend=self.backend, name="spectroscopy") as schedule:
            pulse.shift_frequency(freq_param, pulse.MeasureChannel(qubit))
            pulse.play(
                pulse.GaussianSquare(
                    duration=duration,
                    amp=self.experiment_options.amp,
                    sigma=sigma,
                    width=width,
                ),
                pulse.MeasureChannel(qubit),
            )
            pulse.acquire(duration, qubit, pulse.MemorySlot(0))

        return schedule, freq_param

    def circuits(self):
        """Create the circuit for the spectroscopy experiment.

        The circuits are based on a GaussianSquare pulse and a frequency_shift instruction
        encapsulated in a measurement instruction.

        Returns:
            circuits: The circuits that will run the spectroscopy experiment.
        """
        sched, freq_param = self._schedule()

        circs = []
        for freq in self._frequencies:
            freq_shift = freq - self._backend_center_frequency if self._absolute else freq
            freq_shift = np.round(freq_shift, decimals=3)

            sched_ = sched.assign_parameters({freq_param: freq_shift}, inplace=False)

            circuit = self._template_circuit()
            circuit.add_calibration("measure", self.physical_qubits, sched_)
            self._add_metadata(circuit, freq, sched)

            circs.append(circuit)

        return circs
