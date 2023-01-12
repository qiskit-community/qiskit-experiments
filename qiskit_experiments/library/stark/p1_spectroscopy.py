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
Ramsey T1 experiment class with pulsed Stark tone drive.
"""

from typing import List, Dict, Optional

import numpy as np
from qiskit import pulse, circuit
from qiskit.exceptions import QiskitError
from qiskit.providers.backend import Backend
from qiskit.utils import optionals as _optional

from qiskit_experiments.framework import BaseExperiment, Options, BackendTiming
from .p1_spectroscopy_analysis import StarkP1SpectroscopyAnalysis

if _optional.HAS_SYMENGINE:
    import symengine as sym
else:
    import sympy as sym


class StarkP1Spectroscopy(BaseExperiment):
    """P1 spectroscopy experiment with Stark tone.

    # section: overview

        This experiment scans the survival probability of T1 experiment at certain
        delay time against various qubit frequencies.
        This experiment consists of following circuit:

        .. parsed-literal::

                 ┌───┐┌──────────────────┐┌─┐
              q: ┤ X ├┤ Stark(stark_amp) ├┤M├
                 └───┘└──────────────────┘└╥┘
            c: 1/══════════════════════════╩═
                                           0

        Qubit frequency is modulated by the Stark tone applied during the delay
        in between the X gate and measurement.

        This experiment is often used to characterize TLS in the vicinity of qubit frequency.

    # section: note

        Parameters necessary to convert a Stark shift into tone amplitude must be
        calibrated before running this experiment with :class:`.StarkRamseyFast`.

    """

    def __init__(
        self,
        qubit: int,
        backend: Optional[Backend] = None,
        **experiment_options,
    ):
        self._timing = None

        super().__init__(qubits=[qubit], analysis=StarkP1SpectroscopyAnalysis(), backend=backend)
        self.set_experiment_options(**experiment_options)

    @classmethod
    def _default_experiment_options(cls) -> Options:
        """Default experiment options.

        Experiment Options:
            t1_delay (float): The T1 delay time after excitation pulse. The delay must be
                sufficiently greater than the edge duration determined by the stark_sigma.
            stark_channel (PulseChannel): Pulse channel to apply Stark tones.
                If not provided, the same channel with the qubit drive is used.
            stark_freq_offset (float): Offset of Stark tone frequency from the qubit frequency.
                This must be greater than zero not to apply Rabi drive.
            stark_sigma (float): Gaussian sigma of the rising and falling edges
                of the Stark tone, in seconds.
            stark_risefall (float): Ratio of sigma to the duration of
                the rising and falling edges of the Stark tone.
            min_freq (float): Minimum Stark shift frequency.
            max_freq (float): Maximum Stark shift frequency.
            num_freqs (int): Number of circuits with different Stark shift.
            frequencies (list[float]): The list of Stark shifts that will be scanned in
                the experiment. If not set, then ``num_freqs`` evenly spaced frequency shifts
                between ``min_freq`` and ``max_freq`` are used. If ``frequencies``
                is set, these parameters are ignored.
            stark_amp_limit (float): Power limit of the Stark tone.
                If estimated Stark tone amplitude exceeds this threshold,
                an experiment with such frequency shift is just omitted.
            stark_pos_coef_o2 (float): Calibration parameter from :class:`StarkRamseyFast`.
            stark_pos_coef_o3 (float): Calibration parameter from :class:`StarkRamseyFast`.
            stark_neg_coef_o2 (float): Calibration parameter from :class:`StarkRamseyFast`.
            stark_neg_coef_o3 (float): Calibration parameter from :class:`StarkRamseyFast`.
            stark_offset (float): Calibration parameter from :class:`StarkRamseyFast`
        """
        options = super()._default_experiment_options()
        options.update_options(
            t1_delay=20e-6,
            stark_channel=None,
            stark_freq_offset=80e6,
            stark_sigma=15e-9,
            stark_risefall=2,
            min_freq=-30e6,
            max_freq=30e6,
            num_freqs=201,
            frequencies=None,
            stark_amp_limit=1.0,
            stark_pos_coef_o2=None,
            stark_pos_coef_o3=None,
            stark_neg_coef_o2=None,
            stark_neg_coef_o3=None,
            stark_offset=None,
        )
        options.set_validator("stark_freq_offset", (0, np.inf))
        options.set_validator("stark_channel", pulse.channels.PulseChannel)
        return options

    def _set_backend(self, backend: Backend):
        super()._set_backend(backend)
        self._timing = BackendTiming(backend)

    def frequencies(self) -> np.ndarray:
        """Stark shifts to use in circuits.

        Returns:
            The list of Stark shift to use for the different circuits based on the
            experiment options.
        """
        opt = self.experiment_options  # alias

        if opt.frequencies is None:
            return np.linspace(opt.min_freq, opt.max_freq, opt.num_freqs)
        return opt.frequencies

    def parametrized_circuits(self) -> circuit.QuantumCircuit:
        """Create circuit with parameters for P1 experiment with Stark tone.

        Returns:
            Quantum template circuit of Stark T1 at fixed delay.
        """
        opt = self.experiment_options  # alias
        param = circuit.Parameter("stark_amp")
        sym_param = param._symbol_expr

        # Pulse gates
        stark = circuit.Gate("Stark", 1, [param])

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
        sigma_dt = opt.stark_sigma / self._backend_data.dt
        delay_dt = self._timing.round_pulse(time=opt.t1_delay)

        with pulse.build() as stark_schedule:
            pulse.set_frequency(stark_freq, stark_channel)
            pulse.play(
                pulse.GaussianSquare(
                    duration=delay_dt,
                    amp=abs_of_amp,
                    sigma=sigma_dt,
                    risefall_sigma_ratio=opt.stark_risefall,
                ),
                stark_channel,
            )

        temp_t1 = circuit.QuantumCircuit(1, 1)
        temp_t1.x(0)
        temp_t1.append(stark, [0])
        temp_t1.measure(0, 0)
        temp_t1.add_calibration(
            gate=stark,
            qubits=self.physical_qubits,
            schedule=stark_schedule,
        )

        return temp_t1

    def _freq_to_amp(self, freq_shift: float) -> float:
        """Compute optimal Stark tone amplitude from the frequency.

        Args:
            freq_shift: Target frequency shift in Hz.

        Returns:
            Stark tone amplitude.

        Raises:
            QiskitError: When calibration parameter is not provided.
            QiskitError: When no valid Stark tone amplitude is found.
        """
        opt = self.experiment_options  # alias

        if np.isclose(freq_shift, 0.0):
            return 0.0

        if freq_shift > 0:
            coeff2, coeff3 = opt.stark_pos_coef_o2, opt.stark_pos_coef_o3
        else:
            coeff2, coeff3 = opt.stark_neg_coef_o2, opt.stark_neg_coef_o3
        offset = opt.stark_offset

        if coeff2 is None or coeff3 is None or offset is None:
            raise QiskitError(
                "Calibration parameter is not specified. "
                "Please run StarkRamseyFast experiment and provide experiment options: "
                "stark_pos_coef_o2, stark_pos_coef_o3, "
                "stark_neg_coef_o2, stark_neg_coef_o2, and stark_offset."
            )

        candidates = np.roots([coeff3, coeff2, 0.0, offset - freq_shift])
        # Note that there must be only 1 solution because Stark shift
        # is monotonically increase with the tone amplitude.
        valid = np.where(
            (np.sign(candidates.real) == np.sign(freq_shift))
            & np.isclose(candidates.imag, 0.0)
            & (np.abs(candidates.real) < 1.0)
        )
        if valid[0].size != 1:
            raise QiskitError(
                f"Optimal Stark tone amplitude cannot be found for the shift at {freq_shift} Hz. "
                f"The candidate values are {candidates}, but there is no single valid amplitude. "
                "The valid tone amplitude must be real, same sign with the shift, and less than one. "
                f"Please review calibration coefficients {coeff2, coeff3, offset}."
            )

        return float(candidates[valid].real)

    def circuits(self) -> List[circuit.QuantumCircuit]:
        """Create circuits.

        Returns:
            A list of circuits with a variable delay.
        """
        opt = self.experiment_options  # alias
        t1_circ = self.parametrized_circuits()
        param = next(iter(t1_circ.parameters))

        circs = []
        for freq in self.frequencies():
            amp = self._freq_to_amp(freq)
            if np.abs(amp) > opt.stark_amp_limit:
                continue
            if np.isclose(amp, 0.0):
                # Use standard delay when tone amplitude is zero
                t1_assigned = circuit.QuantumCircuit(1, 1)
                t1_assigned.x(0)
                t1_assigned.delay(self._timing.round_delay(time=opt.t1_delay), 0)
                t1_assigned.measure(0, 0)
            else:
                t1_assigned = t1_circ.assign_parameters({param: amp}, inplace=False)
            t1_assigned.metadata = {
                "xval": freq,
                "amp": amp,
            }
            circs.append(t1_assigned)

        return circs

    def _metadata(self) -> Dict[str, any]:
        """Return experiment metadata for ExperimentData."""
        return {
            "t1_delay": self._timing.pulse_time(time=self.experiment_options.t1_delay),
            "stark_freq_offset": self.experiment_options.stark_freq_offset,
        }
