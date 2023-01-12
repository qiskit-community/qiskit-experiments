# This code is part of Qiskit.
#
# (C) Copyright IBM 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
"""
Ramsey fast experiment class with pulsed Stark tone drive.
"""

from typing import List, Tuple, Dict, Optional

import numpy as np
from qiskit import pulse, circuit
from qiskit.providers.backend import Backend
from qiskit.utils import optionals as _optional
from qiskit_experiments.framework import BaseExperiment, Options, BackendTiming

from .ramsey_fast_analysis import StarkRamseyFastAnalysis

if _optional.HAS_SYMENGINE:
    import symengine as sym
else:
    import sympy as sym


class StarkRamseyFast(BaseExperiment):
    """Experiment to calibrate Stark shift as a function of amplitude.

    # section: overview

        This experiment is identical to :class:`StarkRamseyXY` but scans
        Stark tone amplitude at the fixed flat-top duration.
        This experiment consists of following two circuits:

        .. parsed-literal::

            (Ramsey X)  The pulse before measurement rotates by pi-half around the X axis

                     ┌────┐┌───────────────────┐┌───┐┌───────────────────┐┌────────┐┌────┐┌─┐
                  q: ┤ √X ├┤ StarkV(stark_amp) ├┤ X ├┤ StarkU(stark_amp) ├┤ Rz(-π) ├┤ √X ├┤M├
                     └────┘└───────────────────┘└───┘└───────────────────┘└────────┘└────┘└╥┘
                c: 1/══════════════════════════════════════════════════════════════════════╩═
                                                                                           0

            (Ramsey Y) The pulse before measurement rotates by pi-half around the Y axis

                     ┌────┐┌───────────────────┐┌───┐┌───────────────────┐┌───────────┐┌────┐┌─┐
                  q: ┤ √X ├┤ StarkV(stark_amp) ├┤ X ├┤ StarkU(stark_amp) ├┤ Rz(-3π/2) ├┤ √X ├┤M├
                     └────┘└───────────────────┘└───┘└───────────────────┘└───────────┘└────┘└╥┘
                c: 1/═════════════════════════════════════════════════════════════════════════╩═
                                                                                              0

        This experiment is usually conducted beyond other spectroscopy experiments.
        Because Stark drive cannot directly control the resulting Stark shift,
        an experimentalist must characterize the amount of the shift
        as a function of the drive amplitude to
        modulate qubit frequency to the target value.

        Because frequency offset must be zero at zero Stark tone amplitude
        up to frequency miscalibration, one can directly scan tone amplitude from zero to
        certain amplitude instead of conducting :class:`StarkRamseyXY` experiment at each amplitude.
        This drastically saves time of experiment to characterize Stark shift.
        Note that usually Stark shift is asymmetric with respect to the frequency offset,
        because of the anti-crossing occurring at around higher energy transitions,
        and thus the Stark amplitude parameter must be scanned in both direction.

        This experiment gives several coefficients to convert the tone amplitude into
        the frequency shift, which may be supplied to following spectroscopy experiments.

    # section: analysis_ref
        :py:class:`StarkRamseyFastAnalysis`

    # section: see_also
        qiskit_experiments.library.stark.ramsey_xy.StarkRamseyXY
        qiskit_experiments.library.characterization.ramsey_xy.RamseyXY

    """

    def __init__(
        self,
        qubit: int,
        backend: Optional[Backend] = None,
        **experiment_options,
    ):
        """Create new experiment.

        Args:
            qubit: Index of qubit.
            backend: Optional, the backend to run the experiment on.
            experiment_options: Extra experiment options. See self.experiment_options.
        """
        self._timing = None

        super().__init__(qubits=[qubit], analysis=StarkRamseyFastAnalysis(), backend=backend)
        self.set_experiment_options(**experiment_options)

    @classmethod
    def _default_experiment_options(cls) -> Options:
        """Default experiment options.

        Experiment Options:
            stark_channel (PulseChannel): Pulse channel to apply Stark tones.
                If not provided, the same channel with the qubit drive is used.
            stark_freq_offset (float): Offset of Stark tone frequency from the qubit frequency.
                This must be greater than zero not to apply Rabi drive.
            stark_sigma (float): Gaussian sigma of the rising and falling edges
                of the Stark tone, in seconds.
            stark_delay (float): The time to accumulate Stark shift phase in seconds.
            stark_risefall (float): Ratio of sigma to the duration of
                the rising and falling edges of the Stark tone.
            min_stark_amp: Minimum Stark tone amplitude.
            max_stark_amp: Maximum Stark tone amplitude.
            num_stark_amps (int): Number of circuits per Ramsey X and Y with different amplitude.
            stark_amps (list[float]): The list of amplitudes that will be scanned in
                the experiment. If not set, then ``num_stark_amps`` evenly spaced amplitudes
                between ``min_stark_amp`` and ``max_stark_amp`` are used. If ``stark_amps``
                is set, these parameters are ignored.
        """
        options = super()._default_experiment_options()
        options.update_options(
            stark_channel=None,
            stark_freq_offset=80e6,
            stark_sigma=15e-9,
            stark_delay=50e-9,
            stark_risefall=2,
            min_stark_amp=-0.8,
            max_stark_amp=0.8,
            num_stark_amps=101,
            stark_amps=None,
        )
        options.set_validator("stark_freq_offset", (0, np.inf))
        options.set_validator("stark_channel", pulse.channels.PulseChannel)
        return options

    def _set_backend(self, backend: Backend):
        super()._set_backend(backend)
        self._timing = BackendTiming(backend)

    def amplitudes(self) -> np.ndarray:
        """Stark tone amplitudes to use in circuits.

        Returns:
            The list of amplitudes to use for the different circuits based on the
            experiment options.
        """
        opt = self.experiment_options  # alias

        if opt.stark_amps is None:
            return np.linspace(opt.min_stark_amp, opt.max_stark_amp, opt.num_stark_amps)
        return opt.stark_amps

    def parameterized_circuits(self) -> Tuple[circuit.QuantumCircuit, circuit.QuantumCircuit]:
        """Create circuits with parameters for Ramsey XY experiment with Stark tone.

        Returns:
            Quantum template circuits for Ramsey X and Ramsey Y experiment.
        """
        opt = self.experiment_options  # alias
        param = circuit.Parameter("stark_amp")
        sym_param = param._symbol_expr

        # Pulse gates
        stark_v = circuit.Gate("StarkV", 1, [param])
        stark_u = circuit.Gate("StarkU", 1, [param])

        # Note that Stark tone yields negative (positive) frequency shift
        # when the Stark tone frequency is higher (lower) than qubit f01 frequency.
        # This choice gives positive frequency shift with positive Stark amplitude.
        qubit_f01 = self._backend_data.drive_freqs[self.physical_qubits[0]]
        neg_sign_of_amp = circuit.ParameterExpression(
            symbol_map={param: sym_param},
            expr=-sym.sign(sym_param),
        )
        abs_of_amp = circuit.ParameterExpression(
            symbol_map={param: sym_param},
            expr=sym.Abs(sym_param),
        )
        stark_freq = qubit_f01 + neg_sign_of_amp * opt.stark_freq_offset
        stark_channel = opt.stark_channel or pulse.DriveChannel(self.physical_qubits[0])
        ramps_dt = self._timing.round_pulse(time=2 * opt.stark_risefall * opt.stark_sigma)
        sigma_dt = ramps_dt / 2 / opt.stark_risefall
        delay_dt = self._timing.round_pulse(time=opt.stark_delay)

        with pulse.build() as stark_v_schedule:
            pulse.set_frequency(stark_freq, stark_channel)
            pulse.play(
                pulse.Gaussian(
                    duration=ramps_dt,
                    amp=abs_of_amp,
                    sigma=sigma_dt,
                ),
                stark_channel,
            )

        with pulse.build() as stark_u_schedule:
            pulse.set_frequency(stark_freq, stark_channel)
            pulse.play(
                pulse.GaussianSquare(
                    duration=ramps_dt + delay_dt,
                    amp=abs_of_amp,
                    sigma=sigma_dt,
                    risefall_sigma_ratio=opt.stark_risefall,
                ),
                stark_channel,
            )

        ram_x = circuit.QuantumCircuit(1, 1)
        ram_x.sx(0)
        ram_x.append(stark_v, [0])
        ram_x.x(0)
        ram_x.append(stark_u, [0])
        ram_x.rz(-np.pi, 0)
        ram_x.sx(0)
        ram_x.measure(0, 0)
        ram_x.metadata = {"series": "X"}
        ram_x.add_calibration(
            gate=stark_v,
            qubits=self.physical_qubits,
            schedule=stark_v_schedule,
        )
        ram_x.add_calibration(
            gate=stark_u,
            qubits=self.physical_qubits,
            schedule=stark_u_schedule,
        )

        ram_y = circuit.QuantumCircuit(1, 1)
        ram_y.sx(0)
        ram_y.append(stark_v, [0])
        ram_y.x(0)
        ram_y.append(stark_u, [0])
        ram_y.rz(-np.pi * 3 / 2, 0)
        ram_y.sx(0)
        ram_y.measure(0, 0)
        ram_y.metadata = {"series": "Y"}
        ram_y.add_calibration(
            gate=stark_v,
            qubits=self.physical_qubits,
            schedule=stark_v_schedule,
        )
        ram_y.add_calibration(
            gate=stark_u,
            qubits=self.physical_qubits,
            schedule=stark_u_schedule,
        )

        return ram_x, ram_y

    def circuits(self) -> List[circuit.QuantumCircuit]:
        """Create circuits.

        Returns:
            A list of circuits with a variable delay.
        """
        ramx_circ, ramy_circ = self.parameterized_circuits()
        param = next(iter(ramx_circ.parameters))

        circs = []
        for amp in self.amplitudes():
            if np.isclose(amp, 0.0):
                # To avoid singular point in the fit guess function.
                continue

            ramx_circ_assigned = ramx_circ.assign_parameters({param: amp}, inplace=False)
            ramx_circ_assigned.metadata["xval"] = amp
            ramx_circ_assigned.metadata["direction"] = "pos" if amp > 0 else "neg"

            ramy_circ_assigned = ramy_circ.assign_parameters({param: amp}, inplace=False)
            ramy_circ_assigned.metadata["xval"] = amp
            ramy_circ_assigned.metadata["direction"] = "pos" if amp > 0 else "neg"

            circs.extend([ramx_circ_assigned, ramy_circ_assigned])

        return circs

    def _metadata(self) -> Dict[str, any]:
        """Return experiment metadata for ExperimentData."""
        return {
            "stark_delay": self._timing.pulse_time(time=self.experiment_options.stark_delay),
            "stark_freq_offset": self.experiment_options.stark_freq_offset,
        }
