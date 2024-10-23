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

from typing import Tuple

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import Gate, Parameter
from qiskit.exceptions import QiskitError
from qiskit import pulse

from qiskit_experiments.framework import BackendTiming
from qiskit_experiments.library.characterization.spectroscopy import Spectroscopy


class QubitSpectroscopy(Spectroscopy):
    """A spectroscopy experiment to obtain a frequency sweep of the qubit.

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
        :class:`~qiskit_experiments.curve_analysis.ResonanceAnalysis`

    # section: example
        .. jupyter-execute::
            :hide-code:

            import warnings

            warnings.filterwarnings(
                "ignore",
                message=".*Due to the deprecation of Qiskit Pulse.*",
                category=DeprecationWarning,
            )

            # backend
            from qiskit_experiments.test.pulse_backend import SingleTransmonTestBackend
            backend = SingleTransmonTestBackend(5.2e9,-.25e9, 1e9, 0.8e9, 1e4, noise=True, seed=199)

        .. jupyter-execute::

            import numpy as np
            from qiskit_experiments.library.characterization import QubitSpectroscopy

            qubit = 0
            freq01_estimate = backend.defaults().qubit_freq_est[qubit]
            frequencies = np.linspace(freq01_estimate-15e6, freq01_estimate+15e6, 51)

            exp = QubitSpectroscopy(physical_qubits = (qubit,),
                                    frequencies = frequencies,
                                    backend = backend,
                                    )
            exp.set_experiment_options(amp=0.005)

            exp_data = exp.run().block_for_results()
            display(exp_data.figure(0))
            exp_data.analysis_results(dataframe=True)
    """

    __spec_gate_name__ = "Spec"

    @property
    def _backend_center_frequency(self) -> float:
        """Returns the center frequency of the experiment.

        Returns:
            The center frequency of the experiment.

        Raises:
            QiskitError: If the experiment does not have a backend set.
        """
        if self.backend is None:
            raise QiskitError("backend not set. Cannot determine the center frequency.")

        return self._backend_data.drive_freqs[self.physical_qubits[0]]

    def _template_circuit(self, freq_param) -> QuantumCircuit:
        """Return the template quantum circuit."""
        circuit = QuantumCircuit(1)
        circuit.append(Gate(name=self.__spec_gate_name__, num_qubits=1, params=[freq_param]), (0,))
        circuit.measure_active()

        return circuit

    def _schedule(self) -> Tuple[pulse.ScheduleBlock, Parameter]:
        """Create the spectroscopy schedule."""
        timing = BackendTiming(self.backend)

        if timing.dt is None:
            raise QiskitError(f"{self.__class__.__name__} requires a backend with a dt value.")

        duration = timing.round_pulse(time=self.experiment_options.duration)
        sigma = self.experiment_options.sigma / timing.dt
        width = self.experiment_options.width / timing.dt

        freq_param = Parameter("frequency")

        with pulse.build(backend=self.backend, name="spectroscopy") as schedule:
            pulse.shift_frequency(freq_param, pulse.DriveChannel(self.physical_qubits[0]))
            pulse.play(
                pulse.GaussianSquare(
                    duration=duration,
                    amp=self.experiment_options.amp,
                    sigma=sigma,
                    width=width,
                ),
                pulse.DriveChannel(self.physical_qubits[0]),
            )
            pulse.shift_frequency(-freq_param, pulse.DriveChannel(self.physical_qubits[0]))

        return schedule, freq_param

    def circuits(self):
        """Create the circuit for the spectroscopy experiment.

        The circuits are based on a GaussianSquare pulse and a frequency_shift instruction
        encapsulated in a gate.

        Returns:
            circuits: The circuits that will run the spectroscopy experiment.
        """

        # Create a template circuit
        sched, freq_param = self._schedule()
        circuit = self._template_circuit(freq_param)
        circuit.add_calibration(
            self.__spec_gate_name__, self.physical_qubits, sched, params=[freq_param]
        )

        # Create the circuits to run
        circs = []
        for freq in self._frequencies:
            freq_shift = freq - self._backend_center_frequency if self._absolute else freq
            freq_shift = np.round(freq_shift, decimals=3)

            assigned_circ = circuit.assign_parameters({freq_param: freq_shift}, inplace=False)
            self._add_metadata(assigned_circ, freq)

            circs.append(assigned_circ)

        return circs
