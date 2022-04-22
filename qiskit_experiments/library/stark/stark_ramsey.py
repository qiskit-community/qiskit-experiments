# This code is part of Qiskit.
#
# (C) Copyright IBM 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
T2Ramsey Experiment class.
"""

from typing import List, Optional, Sequence

import numpy as np
from qiskit import pulse, circuit
from qiskit.providers.backend import Backend
from qiskit.pulse import GaussianSquare

from qiskit_experiments.framework import BaseExperiment, Options, BatchExperiment
from qiskit_experiments.library.characterization.analysis.t2ramsey_analysis import T2RamseyAnalysis


class StarkRamsey(BaseExperiment):
    # Characterize Qubit Frequency under stark tone

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
        super().__init__(qubits=[qubit], analysis=T2RamseyAnalysis(), backend=backend)
        self.set_experiment_options(
            delays=delays,
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
        options.delays = None
        return options

    def _set_backend(self, backend: Backend):
        try:
            dt = backend.configuration().dt
        except AttributeError:
            dt = self.experiment_options.dt
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

        circs = []
        for delay_sec in self.experiment_options.delays:
            delay_dt = opt.granularity * int(delay_sec / opt.dt / opt.granularity)

            stark_gate = circuit.Gate("StarkDelay", 1, [delay_dt])
            stark_freq = np.sign(opt.stark_amp) * opt.stark_freq_offset

            with pulse.build() as stark_tone:
                with pulse.frequency_offset(stark_freq, opt.stark_channel):
                    pulse.play(GaussianSquare(duration=delay_dt, amp=np.abs(opt.stark_amp), sigma=opt.stark_sigma,
                                              risefall_sigma_ratio=opt.stark_risefall), opt.stark_channel)
                    pulse_len = delay_dt

                    # pulse_len = play_chunked_gaussian_square(
                    #    duration=delay_dt,
                    #    amp=np.abs(opt.stark_amp),
                    #    sigma=opt.stark_sigma,
                    #    risefall_sigma_ratio=opt.stark_risefall,
                    #    channel=opt.stark_channel)

            circ = circuit.QuantumCircuit(1, 1)
            circ.add_calibration(stark_gate, self.physical_qubits, stark_tone)

            circ.sx(0)

            circ.append(stark_gate, [0])

            circ.barrier(0)
            circ.sx(0)
            circ.barrier(0)
            circ.measure(0, 0)

            circ.metadata = {
                "experiment_type": self._type,
                "qubit": self.physical_qubits[0],
                "unit": "s",
                "xval": pulse_len * opt.dt
            }
            circs.append(circ)
        return circs

class StarkRamseyAmplitudeScan(BatchExperiment):
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

        stark_experiments = [
            StarkRamsey(
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

        super().__init__(stark_experiments, backend=backend)

    def set_experiment_options(self, **fields):
        for exp in self.component_experiment():
            exp.set_experiment_options(**fields)

