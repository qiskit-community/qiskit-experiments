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

"""Ramsey XY frequency characterization experiment."""

from typing import List, Tuple, Dict, Optional

import numpy as np
from qiskit import pulse
from qiskit.circuit import QuantumCircuit, Parameter, Gate
from qiskit.providers.backend import Backend
from qiskit.qobj.utils import MeasLevel

from qiskit_experiments.framework import BaseExperiment, Options, BackendTiming
from qiskit_experiments.framework.restless_mixin import RestlessMixin
from qiskit_experiments.library.characterization.analysis import RamseyXYAnalysis


class RamseyXY(BaseExperiment, RestlessMixin):
    r"""Ramsey XY experiment to measure the frequency of a qubit.

    # section: overview

        This experiment differs from the :class:`~qiskit_experiments.characterization.\
        t2ramsey.T2Ramsey` since it is sensitive to the sign of the frequency offset from
        the main transition. This experiment consists of following two circuits:

        .. parsed-literal::

            (Ramsey X) The second pulse rotates by pi-half around the X axis

                       ┌────┐┌─────────────┐┌───────┐┌────┐ ░ ┌─┐
                  q_0: ┤ √X ├┤ Delay(τ[s]) ├┤ Rz(θ) ├┤ √X ├─░─┤M├
                       └────┘└─────────────┘└───────┘└────┘ ░ └╥┘
            measure: 1/════════════════════════════════════════╩═
                                                               0

            (Ramsey Y) The second pulse rotates by pi-half around the Y axis

                       ┌────┐┌─────────────┐┌───────────┐┌────┐ ░ ┌─┐
                  q_0: ┤ √X ├┤ Delay(τ[s]) ├┤ Rz(θ-π/2) ├┤ √X ├─░─┤M├
                       └────┘└─────────────┘└───────────┘└────┘ ░ └╥┘
            measure: 1/════════════════════════════════════════════╩═
                                                                   0

        The first and second circuits measure the expectation value along the -Y and X axes,
        respectively. This experiment therefore tracks the dynamics of the Bloch vector
        around the equator. The drive frequency of the control electronics defines a reference frame,
        which differs from the true qubit frequency by :math:`\Delta\omega`.
        The Hamiltonian during the
        ``Delay`` instruction is :math:`H^R = - \frac{1}{2} \Delta\omega` in the rotating frame,
        and the propagator will be :math:`U(\tau) = \exp(-iH^R\tau / \hbar)` where :math:`\tau` is the
        duration of the delay. By scanning this duration, we can get

        .. math::

            {\cal E}_x(\tau)
                = {\rm Re} {\rm Tr}\left( Y U \rho U^\dagger \right)
                &= - \cos(\Delta\omega\tau) = \sin(\Delta\omega\tau - \frac{\pi}{2}), \\
            {\cal E}_y(\tau)
                = {\rm Re} {\rm Tr}\left( X U \rho U^\dagger \right)
                &= \sin(\Delta\omega\tau),

        where :math:`\rho` is prepared by the first :math:`\sqrt{\rm X}` gate. Note that phase
        difference of these two outcomes :math:`{\cal E}_x, {\cal E}_y` depends on the sign and
        the magnitude of the frequency offset :math:`\Delta\omega`. By contrast, the measured
        data in the standard Ramsey experiment does not depend on the sign of :math:`\Delta\omega`,
        because :math:`\cos(-\Delta\omega\tau) = \cos(\Delta\omega\tau)`.

        The experiment also allows users to add a small frequency offset to better resolve
        any oscillations. This is implemented by a virtual Z rotation in the circuits. In the
        circuit above it appears as the delay-dependent angle θ(τ).

    # section: analysis_ref
        :py:class:`RamseyXYAnalysis`
    """

    @classmethod
    def _default_experiment_options(cls):
        """Default values for the Ramsey XY experiment.

        Experiment Options:
            delays (list): The list of delays that will be scanned in the experiment, in seconds.
            osc_freq (float): A frequency shift in Hz that will be applied by means of
                a virtual Z rotation to increase the frequency of the measured oscillation.
        """
        options = super()._default_experiment_options()
        options.delays = np.linspace(0, 1.0e-6, 51)
        options.osc_freq = 2e6

        return options

    def __init__(
        self,
        qubit: int,
        backend: Optional[Backend] = None,
        delays: Optional[List] = None,
        osc_freq: float = 2e6,
    ):
        """Create new experiment.

        Args:
            qubit: The qubit on which to run the Ramsey XY experiment.
            backend: Optional, the backend to run the experiment on.
            delays: The delays to scan, in seconds.
            osc_freq: the oscillation frequency induced by the user through a virtual
                Rz rotation. This quantity is given in Hz.
        """
        super().__init__([qubit], analysis=RamseyXYAnalysis(), backend=backend)

        if delays is None:
            delays = self.experiment_options.delays
        self.set_experiment_options(delays=delays, osc_freq=osc_freq)

    def _set_backend(self, backend: Backend):
        super()._set_backend(backend)

        # Scheduling parameters
        if not self._backend_data.is_simulator:
            scheduling_method = getattr(self.transpile_options, "scheduling_method", "alap")
            self.set_transpile_options(scheduling_method=scheduling_method)

    def _pre_circuit(self) -> QuantumCircuit:
        """Return a preparation circuit.

        This method can be overridden by subclasses e.g. to run on transitions other
        than the 0 <-> 1 transition.
        """
        return QuantumCircuit(1)

    def circuits(self) -> List[QuantumCircuit]:
        """Create the circuits for the Ramsey XY characterization experiment.

        Returns:
            A list of circuits with a variable delay.
        """
        dt_unit = False
        if self.backend:
            dt_factor = self._backend_data.dt
            dt_unit = dt_factor is not None

        # Compute the rz rotation angle to add a modulation.
        p_delay_sec = Parameter("delay_sec")
        if dt_unit:
            p_delay_dt = Parameter("delay_dt")

        rotation_angle = 2 * np.pi * self.experiment_options.osc_freq * p_delay_sec

        # Create the X and Y circuits.
        metadata = {
            "experiment_type": self._type,
            "qubits": self.physical_qubits,
            "osc_freq": self.experiment_options.osc_freq,
            "unit": "s",
        }

        ram_x = self._pre_circuit()
        ram_x.sx(0)

        if dt_unit:
            ram_x.delay(p_delay_dt, 0, "dt")
        else:
            ram_x.delay(p_delay_sec, 0, "s")

        ram_x.rz(rotation_angle, 0)
        ram_x.sx(0)
        ram_x.measure_active()
        ram_x.metadata = metadata.copy()

        ram_y = self._pre_circuit()
        ram_y.sx(0)

        if dt_unit:
            ram_y.delay(p_delay_dt, 0, "dt")
        else:
            ram_y.delay(p_delay_sec, 0, "s")

        ram_y.rz(rotation_angle - np.pi / 2, 0)
        ram_y.sx(0)
        ram_y.measure_active()
        ram_y.metadata = metadata.copy()

        circs = []
        for delay in self.experiment_options.delays:
            if dt_unit:
                delay_dt = round(delay / dt_factor)
                real_delay_in_sec = delay_dt * dt_factor
            else:
                real_delay_in_sec = delay

            # create ramsey x
            if dt_unit:
                assigned_x = ram_x.assign_parameters(
                    {p_delay_sec: real_delay_in_sec, p_delay_dt: delay_dt}, inplace=False
                )
            else:
                assigned_x = ram_x.assign_parameters(
                    {p_delay_sec: real_delay_in_sec}, inplace=False
                )

            assigned_x.metadata["series"] = "X"
            assigned_x.metadata["xval"] = real_delay_in_sec

            # create ramsey y
            if dt_unit:
                assigned_y = ram_y.assign_parameters(
                    {p_delay_sec: real_delay_in_sec, p_delay_dt: delay_dt}, inplace=False
                )
            else:
                assigned_y = ram_y.assign_parameters(
                    {p_delay_sec: real_delay_in_sec}, inplace=False
                )

            assigned_y.metadata["series"] = "Y"
            assigned_y.metadata["xval"] = real_delay_in_sec

            circs.extend([assigned_x, assigned_y])

        return circs

    def _finalize(self):
        # Set initial guess for sinusoidal offset when meas level is 2.
        # This returns probability P1 thus offset=0.5 is obvious.
        # This guarantees reasonable fit especially when data contains only less than half cycle.
        meas_level = self.run_options.get("meas_level", MeasLevel.CLASSIFIED)
        if meas_level == MeasLevel.CLASSIFIED:
            init_guess = self.analysis.options.get("p0", {})
            if "base" not in init_guess:
                init_guess["base"] = 0.5
            self.analysis.set_options(p0=init_guess)

    def _metadata(self):
        metadata = super()._metadata()
        # Store measurement level and meas return if they have been
        # set for the experiment
        for run_opt in ["meas_level", "meas_return"]:
            if hasattr(self.run_options, run_opt):
                metadata[run_opt] = getattr(self.run_options, run_opt)
        return metadata


class StarkRamseyXY(BaseExperiment):
    """Ramsey XY experiment with pulsed Stark tone.

    # section: overview

        This experiment is a variant of :class:`.RamseyXY` with a pulsed Stark tone
        and consists of the following two circuits:

        .. parsed-literal::

            (Ramsey X)  The pulse before measurement rotates by pi-half around the X axis

                     ┌────┐┌────────┐┌───┐┌───────────────┐┌────────┐┌────┐┌─┐
                  q: ┤ √X ├┤ StarkV ├┤ X ├┤ StarkU(delay) ├┤ Rz(-π) ├┤ √X ├┤M├
                     └────┘└────────┘└───┘└───────────────┘└────────┘└────┘└╥┘
                c: 1/═══════════════════════════════════════════════════════╩═
                                                                            0

            (Ramsey Y) The pulse before measurement rotates by pi-half around the Y axis

                     ┌────┐┌────────┐┌───┐┌───────────────┐┌───────────┐┌────┐┌─┐
                  q: ┤ √X ├┤ StarkV ├┤ X ├┤ StarkU(delay) ├┤ Rz(-3π/2) ├┤ √X ├┤M├
                     └────┘└────────┘└───┘└───────────────┘└───────────┘└────┘└╥┘
                c: 1/══════════════════════════════════════════════════════════╩═
                                                                               0

        In principle, the sequence is a variant of :class:`.RamseyXY` circuit.
        However, the delay in between √X gates is replaced with the off-resonant drive.
        This off-resonant drive shifts the qubit frequency due to the
        Stark effect and causes it to accumulate phase during the
        Ramsey sequence. This frequency shift is a function of the
        offset of the Stark tone frequency from the qubit frequency
        and of the magnitude of the tone.

        Note that the Stark tone pulse (StarkU) takes the form of a flat-topped Gaussian envelope.
        The magnitude of the pulse varies in time during its rising and falling edges.
        It is difficult to characterize the net phase accumulation of the qubit during the
        edges of the pulse when the frequency shift is varying with the pulse amplitude.
        In order to simplify the analysis, an additional pulse (StarkV)
        involving only the edges of StarkU is added to the sequence.
        The sign of the phase accumulation is inverted for StarkV relative to that of StarkU
        by inserting an X gate in between the two pulses.

        This technique allows the experiment to accumulate only the net phase
        during the flat-top part of the StarkU pulse with constant magnitude.

    # section: analysis_ref
        :py:class:`RamseyXYAnalysis`

    # section: see_also
        qiskit_experiments.library.characterization.ramsey_xy.RamseyXY

    """

    def __init__(
        self,
        qubit: int,
        stark_amp: float,
        backend: Optional[Backend] = None,
        **experiment_options,
    ):
        """Create new experiment.

        Args:
            qubit: Index of qubit.
            stark_amp: A single float parameter to represent the magnitude of the Stark tone
                and the sign of expected the Stark shift.
                See :ref:`stark_tone_implementation` for details.
            backend: Optional, the backend to run the experiment on.
            experiment_options: Extra experiment options. See ``self.experiment_options``.
        """
        self._timing = None

        super().__init__(qubits=[qubit], analysis=RamseyXYAnalysis(), backend=backend)
        self.set_experiment_options(stark_amp=stark_amp, **experiment_options)

    @classmethod
    def _default_experiment_options(cls) -> Options:
        """Default experiment options.

        Experiment Options:
            stark_amp (float): A single float parameter to represent the magnitude of Stark tone
                and the sign of expected Stark shift.
                See :ref:`stark_tone_implementation` for details.
            stark_channel (PulseChannel): Pulse channel to apply Stark tones.
                If not provided, the same channel with the qubit drive is used.
                See :ref:`stark_channel_consideration` for details.
            stark_freq_offset (float): Offset of Stark tone frequency from the qubit frequency.
                This offset should be sufficiently large so that the Stark pulse
                does not Rabi drive the qubit.
                See :ref:`stark_frequency_consideration` for details.
            stark_sigma (float): Gaussian sigma of the rising and falling edges
                of the Stark tone, in seconds.
            stark_risefall (float): Ratio of sigma to the duration of
                the rising and falling edges of the Stark tone.
            min_freq (float): Minimum frequency that this experiment is guaranteed to resolve.
                Note that fitter algorithm :class:`.RamseyXYAnalysis` of this experiment
                is still capable of fitting experiment data with lower frequency.
            max_freq (float): Maximum frequency that this experiment can resolve.
            delays (list[float]): The list of delays if set that will be scanned in the
                experiment. If not set, then evenly spaced delays with interval
                computed from ``min_freq`` and ``max_freq`` are used.
                See :meth:`StarkRamseyXY.delays` for details.
        """
        options = super()._default_experiment_options()
        options.update_options(
            stark_amp=0.0,
            stark_channel=None,
            stark_freq_offset=80e6,
            stark_sigma=15e-9,
            stark_risefall=2,
            min_freq=5e6,
            max_freq=100e6,
            delays=None,
        )
        options.set_validator("stark_freq_offset", (0, np.inf))
        options.set_validator("stark_channel", pulse.channels.PulseChannel)
        return options

    def _set_backend(self, backend: Backend):
        super()._set_backend(backend)
        self._timing = BackendTiming(backend)

    def delays(self) -> np.ndarray:
        """Delay values to use in circuits.

        .. note::

            The delays are computed with the ``min_freq`` and ``max_freq`` experiment options.
            The maximum point is computed from the ``min_freq`` to guarantee the result
            contains at least one Ramsey oscillation cycle at this frequency.
            The interval is computed from the ``max_freq`` to sample with resolution
            such that the Nyquist frequency is twice ``max_freq``.

        Returns:
            The list of delays to use for the different circuits based on the
            experiment options.
        """
        opt = self.experiment_options  # alias

        if opt.delays is None:
            # Delay is longer enough to capture 1 cycle of the minmum frequency.
            # Fitter can still accurately fit samples shorter than 1 cycle.
            max_period = 1 / opt.min_freq
            # Inverse of interval should be greater than Nyquist frequency.
            sampling_freq = 2 * opt.max_freq
            interval = 1 / sampling_freq
            return np.arange(0, max_period, interval)
        return opt.delays

    def parameterized_circuits(self) -> Tuple[QuantumCircuit, QuantumCircuit]:
        """Create circuits with parameters for Ramsey XY experiment with Stark tone.

        Returns:
            Quantum template circuits for Ramsey X and Ramsey Y experiment.
        """
        opt = self.experiment_options  # alias
        param = Parameter("delay")

        # Pulse gates
        stark_v = Gate("StarkV", 1, [])
        stark_u = Gate("StarkU", 1, [param])

        # Note that Stark tone yields negative (positive) frequency shift
        # when the Stark tone frequency is higher (lower) than qubit f01 frequency.
        # This choice gives positive frequency shift with positive Stark amplitude.
        qubit_f01 = self._backend_data.drive_freqs[self.physical_qubits[0]]
        stark_freq = qubit_f01 - np.sign(opt.stark_amp) * opt.stark_freq_offset
        stark_amp = np.abs(opt.stark_amp)
        stark_channel = opt.stark_channel or pulse.DriveChannel(self.physical_qubits[0])
        ramps_dt = self._timing.round_pulse(time=2 * opt.stark_risefall * opt.stark_sigma)
        sigma_dt = ramps_dt / 2 / opt.stark_risefall

        with pulse.build() as stark_v_schedule:
            pulse.set_frequency(stark_freq, stark_channel)
            pulse.play(
                pulse.Gaussian(
                    duration=ramps_dt,
                    amp=stark_amp,
                    sigma=sigma_dt,
                ),
                stark_channel,
            )

        with pulse.build() as stark_u_schedule:
            pulse.set_frequency(stark_freq, stark_channel)
            pulse.play(
                pulse.GaussianSquare(
                    duration=ramps_dt + param,
                    amp=stark_amp,
                    sigma=sigma_dt,
                    risefall_sigma_ratio=opt.stark_risefall,
                ),
                stark_channel,
            )

        ram_x = QuantumCircuit(1, 1)
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

        ram_y = QuantumCircuit(1, 1)
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

    def circuits(self) -> List[QuantumCircuit]:
        """Create circuits.

        Returns:
            A list of circuits with a variable delay.
        """
        ramx_circ, ramy_circ = self.parameterized_circuits()
        param = next(iter(ramx_circ.parameters))

        circs = []
        dt = self._backend_data.dt
        granularity = self._backend_data.granularity
        for delay in self.delays():
            # Not using pulse_round method of the BackendTiming class
            # because this method considers the minimum pulse duration.
            # Valid delay here corresponds to the flat-top length and thus can be zero at minimum.
            valid_delay_dt = granularity * int(round(delay / dt / granularity))
            net_delay_sec = valid_delay_dt * dt

            ramx_circ_assigned = ramx_circ.assign_parameters({param: valid_delay_dt}, inplace=False)
            ramx_circ_assigned.metadata["xval"] = net_delay_sec

            ramy_circ_assigned = ramy_circ.assign_parameters({param: valid_delay_dt}, inplace=False)
            ramy_circ_assigned.metadata["xval"] = net_delay_sec

            circs.extend([ramx_circ_assigned, ramy_circ_assigned])

        return circs

    def _metadata(self) -> Dict[str, any]:
        """Return experiment metadata for ExperimentData."""
        return {
            "stark_amp": self.experiment_options.stark_amp,
            "stark_freq_offset": self.experiment_options.stark_freq_offset,
        }
