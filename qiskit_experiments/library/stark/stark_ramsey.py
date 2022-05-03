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
from qiskit.exceptions import QiskitError

from qiskit_experiments.framework import BaseExperiment, Options, BatchExperiment

from .analysis import StarkRamseyAnalysis


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
        # backend parameters required to run this experiment
        self._dt = 1
        self._granularity = 1

        super().__init__(qubits=[qubit], analysis=StarkRamseyAnalysis(), backend=backend)
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
        """
        options = super()._default_experiment_options()
        options.stark_amp = 0.0
        options.stark_freq_offset = 0.0
        options.stark_sigma = 64
        options.stark_risefall = 2
        options.stark_channel = None
        options.delays = None
        return options

    def _set_backend(self, backend: Backend):
        super()._set_backend(backend)
        try:
            self._dt = backend.configuration().dt
        except AttributeError:
            raise QiskitError(
                f"{backend.name()} doesn't provide system time resolution dt. "
                "This value is necessary to convert provided delays in SI unit into samples.",
            )
        try:
            self._granularity = backend.configuration().timing_constraints["granularity"]
        except (AttributeError, KeyError):
            pass

    def circuits(self) -> List[circuit.QuantumCircuit]:
        if self._backend is None:
            raise QiskitError(
                f"Backend must be set before generating {self.__class__.__name__} circuit. "
            )

        opt = self.experiment_options
        min_delay = 2 * opt.stark_risefall * opt.stark_sigma

        circs = []
        for delay_sec in self.experiment_options.delays:
            delay_dt = self._granularity * int(delay_sec / self._dt / self._granularity)
            if delay_dt < min_delay:
                raise ValueError(
                    f"Minimum Ramsey delay must be longer than {min_delay * self._dt: .3e} sec. "
                    "This value corresponds to the rising and falling edges of pulsed Stark tone "
                    "implemented as a flat-topped Gaussian pulse."
                )

            stark_gate = circuit.Gate("StarkDelay", 1, [delay_dt])

            # Use abstract amplitude:
            # Stark shift has quadratic dependency on the drive amplitude.
            # This means negative and positive amplitude yield the same Stark shift.
            # Since we may want to scan amplitude from positive to negative value,
            # i.e. choosing amplitude as a control parameter rather than detuning,
            # we conventionally flip the sign of frequency detuning
            # if negative drive amplitude is set.
            stark_freq = np.sign(opt.stark_amp) * opt.stark_freq_offset

            with pulse.build() as stark_tone:
                with pulse.frequency_offset(stark_freq, opt.stark_channel):
                    pulse.play(
                        GaussianSquare(
                            duration=delay_dt,
                            amp=np.abs(opt.stark_amp),
                            sigma=opt.stark_sigma,
                            risefall_sigma_ratio=opt.stark_risefall
                        ),
                        opt.stark_channel,
                    )

            circ = circuit.QuantumCircuit(1, 1)
            circ.add_calibration(stark_gate, self.physical_qubits, stark_tone)

            circ.sx(0)
            circ.append(stark_gate, [0])
            circ.sx(0)
            circ.measure(0, 0)

            circ.metadata = {
                "experiment_type": self._type,
                "qubit": self.physical_qubits[0],
                "unit": "s",
                "xval": delay_dt * self._dt
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

