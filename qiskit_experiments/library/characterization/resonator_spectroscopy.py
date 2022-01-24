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

from qiskit_experiments.framework import Options
from qiskit_experiments.library.characterization.spectroscopy import Spectroscopy
from qiskit_experiments.data_processing.processor_library import get_processor, ProjectorType
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

        where a spectroscopy pulse is attached to the measurement instruction. When doing
        readout resonator spectroscopy, each measured IQ point has a frequency dependent
        phase. Close to the resonance, the IQ points start rotating around in the IQ plan.
        This effect must be accounted for in the data processing to produce a meaningful
        signal.

    # section: analysis_ref
        :py:class:`ResonatorSpectroscopyAnalysis`

    # section: see_also
        qiskit_experiments.library.characterization.qubit_spectroscopy.QubitSpectroscopy
    """

    @classmethod
    def _default_experiment_options(cls) -> Options:
        """Default option values used for the spectroscopy pulse.

        Experiment Options:
            amp (float): The amplitude of the spectroscopy pulse. Defaults to 1 and must
                be between 0 and 1.
            acquisition_duration (int): The duration of the acquisition instruction. By default
                is lasts 1024 samples, i.e. the same duration as the measurement pulse.
        """
        options = super()._default_experiment_options()

        options.amp = 1
        options.acquire_duration = 1024
        options.acquire_delay = 0

        return options

    def __init__(
        self,
        qubit: int,
        frequencies: Iterable[float],
        backend: Optional[Backend] = None,
        absolute: bool = True,
        **experiment_options,
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
                resonator frequency in the backend.
            experiment_options: Key word arguments used to set the experiment options.
        """
        analysis = ResonatorSpectroscopyAnalysis()
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

        return self.backend.defaults().meas_freq_est[self.physical_qubits[0]]

    def _template_circuit(self) -> QuantumCircuit:
        """Return the template quantum circuit."""
        circuit = QuantumCircuit(1, 1)
        circuit.measure(0, 0)

        return circuit

    def _schedule(self) -> Tuple[pulse.ScheduleBlock, Parameter]:
        """Create the spectroscopy schedule."""

        qubit = self.physical_qubits[0]

        freq_param = Parameter("frequency")

        with pulse.build(backend=self.backend, name="spectroscopy") as schedule:
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

            with pulse.align_left():
                pulse.delay(self.experiment_options.acquire_delay, pulse.AcquireChannel(qubit))
                pulse.acquire(self.experiment_options.acquire_duration, qubit, pulse.MemorySlot(0))

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
