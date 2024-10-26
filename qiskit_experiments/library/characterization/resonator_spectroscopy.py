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

import warnings
from typing import Iterable, Optional, Sequence, Tuple
import numpy as np

from qiskit import QuantumCircuit
from qiskit import pulse
from qiskit.circuit import Parameter
from qiskit.exceptions import QiskitError
from qiskit.providers import Backend

from qiskit_experiments.framework import BackendData, BackendTiming, Options
from qiskit_experiments.library.characterization.spectroscopy import Spectroscopy
from qiskit_experiments.database_service import Resonator
from .analysis.resonator_spectroscopy_analysis import ResonatorSpectroscopyAnalysis


class ResonatorSpectroscopy(Spectroscopy):
    """An experiment to perform frequency spectroscopy of the readout resonator.

    # section: overview
        This experiment does spectroscopy on the readout resonator. It applies the following
        circuit

        .. parsed-literal::

                 ┌─┐
              q: ┤M├
                 └╥┘
            c: 1/═╩═
                  0

        where a spectroscopy pulse is attached to the measurement instruction. An initial circuit
        can be added before the measurement by setting the ``initial_circuit`` experiment option. If
        set, the experiment applies the following circuit:

        .. parsed-literal::

                 ┌─────────────────┐┌─┐
              q: ┤ initial_circuit ├┤M├
                 └─────────────────┘└╥┘
            c: 1/════════════════════╩═
                                     0

        Side note 1: when doing readout resonator spectroscopy, each measured IQ point has a
        frequency dependent phase. Close to the resonance, the IQ points start rotating around
        in the IQ plan. This effect must be accounted for in the data processing to produce a
        meaningful signal. The default data processing workflow will therefore reduce the two-
        dimensional IQ data to one-dimensional data using the magnitude of each IQ point.

        Side node 2: when running readout resonator spectroscopy in a parallel experiment the
        user will need to specify the memory slot to use. This can easily be done with the code
        shown below.

        .. code::

            specs = []
            for slot, qubit in enumerate(qubits):
                specs.append(ResonatorSpectroscopy(
                    physical_qubits=[qubit],
                    backend=backend2,
                    memory_slot=slot,
                ))

            exp = ParallelExperiment(specs, backend=backend2)

        # section: warning
            Some backends may not have the required functionality to properly support resonator
            spectroscopy experiments. The experiment may not work or the resulting resonance
            may not properly reflect the properties of the readout resonator.

    # section: example

        The resonator spectroscopy experiment can be run by doing:

        .. code:: python

            qubit = 1
            spec = ResonatorSpectroscopy([qubit], backend)
            exp_data = spec.run().block_for_results()
            exp_data.figure(0)

        This will measure the resonator attached to qubit 1 and report the resonance frequency
        as well as the kappa, i.e. the line width, of the resonator.

    # section: analysis_ref
        :class:`ResonatorSpectroscopyAnalysis`
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
                seconds.
            initial_circuit (QuantumCircuit): A single-qubit initial circuit to run before the
                measurement/spectroscopy pulse. The circuit must contain only a single qubit and zero
                classical bits. If None, no circuit is appended before the measurement. Defaults to None.
            memory_slot (int): The memory slot that the acquire instruction uses in the pulse schedule.
                The default value is ``0``. This argument allows the experiment to function in a
                :class:`.ParallelExperiment`.
        """
        options = super()._default_experiment_options()

        options.amp = 1
        options.duration = 480e-9
        options.sigma = 60e-9
        options.width = 360e-9
        options.initial_circuit = None
        options.memory_slot = 0

        return options

    def set_experiment_options(self, **fields):
        # Check that the initial circuit is for a single qubit only.
        if "initial_circuit" in fields:
            initial_circuit = fields["initial_circuit"]
            if (
                initial_circuit is not None
                and initial_circuit.num_qubits != 1
                or initial_circuit.num_clbits != 0
            ):
                raise QiskitError(
                    "Initial circuit must be for exactly one qubit and zero classical bits. Got "
                    f"{initial_circuit.num_qubits} qubits and {initial_circuit.num_clbits} "
                    "classical bits instead."
                )
        return super().set_experiment_options(**fields)

    def __init__(
        self,
        physical_qubits: Sequence[int],
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
            physical_qubits: List containing the resonator on which to run readout
                spectroscopy.
            backend: Optional, the backend to run the experiment on.
            frequencies: The frequencies to scan in the experiment, in Hz. The default values
                range from -20 MHz to 20 MHz in 51 steps. If the ``absolute`` variable is
                set to True then a center frequency obtained from the backend's defaults is
                added to each value of this range.
            absolute: Boolean to specify if the frequencies are absolute or relative to the
                resonator frequency in the backend. The default value is True.
            experiment_options: Key word arguments used to set the experiment options.

        Raises:
            QiskitError: If no frequencies are given and absolute frequencies are desired and
                no backend is given or the backend does not have default measurement frequencies.
        """
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="deprecation of Qiskit Pulse",
                module="qiskit_experiments",
                category=DeprecationWarning,
            )
            analysis = ResonatorSpectroscopyAnalysis()

        if frequencies is None:
            frequencies = np.linspace(-20.0e6, 20.0e6, 51)

            if absolute:
                frequencies += self._get_backend_meas_freq(
                    BackendData(backend) if backend is not None else None,
                    physical_qubits[0],
                )

        super().__init__(
            physical_qubits, frequencies, backend, absolute, analysis, **experiment_options
        )

    @staticmethod
    def _get_backend_meas_freq(backend_data: Optional[BackendData], qubit: int) -> float:
        """Get backend meas_freq with error checking"""
        if backend_data is None:
            raise QiskitError(
                "Cannot automatically compute absolute frequencies without a backend."
            )

        if len(backend_data.meas_freqs) < qubit + 1:
            raise QiskitError(
                "Cannot retrieve default measurement frequencies from backend. "
                "Please set frequencies explicitly or set `absolute` to `False`."
            )
        return backend_data.meas_freqs[qubit]

    @property
    def _backend_center_frequency(self) -> float:
        """Returns the center frequency of the experiment.

        Returns:
            The center frequency of the experiment.

        Raises:
            QiskitError: If the experiment does not have a backend set.
        """
        return self._get_backend_meas_freq(self._backend_data, self.physical_qubits[0])

    def _template_circuit(self) -> QuantumCircuit:
        """Return the template quantum circuit."""
        circuit = QuantumCircuit(1, 1)
        if self.experiment_options.initial_circuit is not None:
            circuit.append(self.experiment_options.initial_circuit, [0])
        circuit.measure(0, 0)

        return circuit

    def _schedule(self) -> Tuple[pulse.ScheduleBlock, Parameter]:
        """Create the spectroscopy schedule."""
        timing = BackendTiming(self.backend)

        if timing.dt is None:
            raise QiskitError(f"{self.__class__.__name__} requires a backend with a dt value.")

        duration = timing.round_pulse(time=self.experiment_options.duration)
        sigma = self.experiment_options.sigma / timing.dt
        width = self.experiment_options.width / timing.dt

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
            pulse.acquire(duration, qubit, pulse.MemorySlot(self.experiment_options.memory_slot))

        return schedule, freq_param

    def _metadata(self):
        """Update metadata with the resonator components."""
        metadata = super()._metadata()
        metadata["device_components"] = list(map(Resonator, self.physical_qubits))
        return metadata

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
            self._add_metadata(circuit, freq)

            circs.append(circuit)

        return circs
