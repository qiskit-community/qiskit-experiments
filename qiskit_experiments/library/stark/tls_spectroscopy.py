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

"""TLS Spectroscopy experiment with T1 scan."""

from qiskit_experiments.library import T1
from typing import Sequence, Optional, List
from qiskit.providers.backend import Backend
from qiskit import pulse, circuit
from qiskit_experiments.framework import BatchExperiment, Options
import numpy as np


class StarkT1(T1):

    def __init__(
        self,
        qubit: int,
        delays: Sequence[float],
        stark_amp: float,
        stark_freq_offset: float,
        stark_channel: Optional[pulse.channels.PulseChannel] = None,
        backend: Optional[Backend] = None,
        **kwargs,
    ):
        super().__init__(qubit=qubit, delays=delays, backend=backend)
        self.set_experiment_options(
            stark_channel=stark_channel or pulse.DriveChannel(qubit),
            stark_amp=stark_amp,
            stark_freq_offset=stark_freq_offset,
            **kwargs,
        )

    @classmethod
    def _default_experiment_options(cls) -> Options:
        """Default experiment options.

        Experiment Options:
            stark_amp (float): blah
            stark_freq_offset (float): blah
            stark_sigma (float): blah
            stark_risefall (float): blah
            stark_channel (PulseChannel): blah
            qubit_freq (float): blah
            dt (float): blah
            granularity (float): blah
        """
        options = super()._default_experiment_options()
        options.stark_amp = 0.0
        options.stark_freq_offset = 0.0
        options.stark_sigma = 64
        options.stark_risefall = 2
        options.stark_channel = None
        options.qubit_freq = 0
        options.dt = 1e-9
        options.granularity = 1
        return options

    def _set_backend(self, backend: Backend):
        try:
            dt=backend.configuration().dt
        except AttributeError:
            dt=self.experiment_options.dt
        try:
            granularity = backend.configuration().timing_constraints["granularity"]
        except (AttributeError, KeyError):
            granularity = self.experiment_options.granularity
        try:
            qubit_freq = backend.defaults().qubit_freq_est[self.physical_qubits[0]]
        except (AttributeError, IndexError):
            qubit_freq = self.experiment_options.qubit_freq

        self.set_experiment_options(dt=dt, granularity=granularity, qubit_freq=qubit_freq)
        super()._set_backend(backend)

    def circuits(self) -> List[circuit.QuantumCircuit]:
        opt = self.experiment_options

        delay = circuit.Parameter("delay")
        stark_gate = circuit.Gate("StarkDelay", 1, [delay])
        stark_freq = opt.qubit_freq + np.sign(opt.stark_amp) * opt.stark_freq_offset

        with pulse.build() as stark_tone:
            pulse.set_frequency(stark_freq, opt.stark_channel)
            pulse.play(
                pulse.GaussianSquare(
                    duration=delay,
                    amp=np.abs(opt.stark_amp),
                    sigma=opt.stark_sigma,
                    risefall_sigma_ratio=opt.stark_risefall,
                ),
                opt.stark_channel,
            )

        circ = circuit.QuantumCircuit(1, 1)
        circ.x(0)
        circ.barrier(0)
        circ.append(stark_gate, [0])
        circ.barrier(0)
        circ.measure(0, 0)
        circ.metadata = {
            "experiment_type": self._type,
            "qubit": self.physical_qubits[0],
            "unit": "s",
        }
        circ.add_calibration(stark_gate, self.physical_qubits, stark_tone, [delay])

        circs = []
        for delay_sec in self.experiment_options.delays:
            delay_dt = opt.granularity * int(delay_sec / opt.dt / opt.granularity)
            try:
                citc_t = circ.assign_parameters({delay: delay_dt}, inplace=False)
            except pulse.PulseError as ex:
                raise pulse.PulseError(
                    f"Assigned duration of {delay_dt} dt is shorter than pulse edge duration. "
                    "Set longer delay values than that of pulse edges."
                )
            citc_t.metadata["xval"] = delay_dt * opt.dt

            circs.append(citc_t)

        return circs


class TLSSpectroscopy(BatchExperiment):
    def __init__(
        self,
        qubit: int,
        delays: Sequence[float],
        stark_amps: Sequence[float],
        stark_freq_offset: float,
        stark_channel: Optional[pulse.channels.PulseChannel] = None,
        backend: Optional[Backend] = None,
        **kwargs,
    ):
        # TODO write analysis that draws heatmap and T1 vs freq
        #      analysis options may take calibration curve for axis conversion
        spect_experiments = [
            StarkT1(
                qubit=qubit,
                delays=delays,
                stark_amp=stark_amp,
                stark_freq_offset=stark_freq_offset,
                stark_channel=stark_channel,
                backend=backend,
                **kwargs,
            )
            for stark_amp in stark_amps
        ]

        super().__init__(spect_experiments, backend=backend)

    def set_experiment_options(self, **fields):
        for exp in self.component_experiment():
            exp.set_experiment_options(**fields)
