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
"""Stark Ramsey experiment directly scanning Stark amplitude."""

from __future__ import annotations

import warnings
from collections.abc import Sequence

import numpy as np
from qiskit import pulse
from qiskit.circuit import QuantumCircuit, Gate, ParameterExpression, Parameter
from qiskit.providers.backend import Backend
from qiskit.utils import optionals as _optional
from qiskit.utils.deprecation import deprecate_func

from qiskit_experiments.framework import BaseExperiment, Options, BackendTiming
from .ramsey_amp_scan_analysis import StarkRamseyXYAmpScanAnalysis

if _optional.HAS_SYMENGINE:
    import symengine as sym
else:
    import sympy as sym


class StarkRamseyXYAmpScan(BaseExperiment):
    r"""A fast characterization of Stark frequency shift by varying Stark tone amplitude.

    # section: overview

        This experiment scans Stark tone amplitude at a fixed tone duration.
        The experimental circuits are identical to the :class:`.StarkRamseyXY` experiment
        except that the Stark pulse amplitude is the scanned parameter rather than the pulse width.

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

        The AC Stark effect can be used to shift the frequency of a qubit with a microwave drive.
        To calibrate a specific frequency shift, the :class:`.StarkRamseyXY` experiment can be run
        to scan the Stark pulse duration at every amplitude, but such a two dimensional scan of
        the tone duration and amplitude may require many circuit executions.
        To avoid this overhead, the :class:`.StarkRamseyXYAmpScan` experiment fixes the
        tone duration and scans only amplitude.

        Recall that an observed Ramsey oscillation in each quadrature may be represented by

        .. math::

            {\cal E}_X(\Omega, t_S) = A e^{-t_S/\tau} \cos \left( 2\pi f_S(\Omega) t_S \right), \\
            {\cal E}_Y(\Omega, t_S) = A e^{-t_S/\tau} \sin \left( 2\pi f_S(\Omega) t_S \right),

        where :math:`f_S(\Omega)` denotes the amount of Stark shift
        at a constant tone amplitude :math:`\Omega`, and :math:`t_S` is the duration of the
        applied tone. For a fixed tone duration,
        one can still observe the Ramsey oscillation by scanning the tone amplitude.
        However, since :math:`f_S` is usually a higher order polynomial of :math:`\Omega`,
        one must manage to fit the y-data for trigonometric functions with
        phase which non-linearly changes with the x-data.
        The :class:`.StarkRamseyXYAmpScan` experiment thus drastically reduces the number of
        circuits to run in return for greater complexity in the fitting model.

    # section: analysis_ref
        :class:`qiskit_experiments.library.driven_freq_tuning.StarkRamseyXYAmpScanAnalysis`

    # section: see_also
        :class:`qiskit_experiments.library.driven_freq_tuning.ramsey.StarkRamseyXY`
        :class:`qiskit_experiments.library.characterization.ramsey_xy.RamseyXY`

    # section: manual
        :doc:`/manuals/characterization/stark_experiment`

    """

    @deprecate_func(
        since="0.8",
        package_name="qiskit-experiments",
        additional_msg=(
            "Due to the deprecation of Qiskit Pulse, experiments involving pulse "
            "gate calibrations like this one have been deprecated."
        ),
    )
    def __init__(
        self,
        physical_qubits: Sequence[int],
        backend: Backend | None = None,
        **experiment_options,
    ):
        """Create new experiment.

        Args:
            physical_qubits: Sequence with the index of the physical qubit.
            backend: Optional, the backend to run the experiment on.
            experiment_options: Experiment options. See the class documentation or
                ``self._default_experiment_options`` for descriptions.
        """
        self._timing = None

        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="deprecation of Qiskit Pulse",
                module="qiskit_experiments",
                category=DeprecationWarning,
            )
            analysis = StarkRamseyXYAmpScanAnalysis()

        super().__init__(
            physical_qubits=physical_qubits,
            analysis=analysis,
            backend=backend,
        )
        self.set_experiment_options(**experiment_options)

    @classmethod
    def _default_experiment_options(cls) -> Options:
        """Default experiment options.

        Experiment Options:
            stark_channel (PulseChannel): Pulse channel on which  to apply Stark tones.
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
            stark_length (float): Time to accumulate Stark shifted phase in seconds.
            min_stark_amp (float): Minimum Stark tone amplitude.
            max_stark_amp (float): Maximum Stark tone amplitude.
            num_stark_amps (int): Number of Stark tone amplitudes to scan.
            stark_amps (list[float]): The list of amplitude that will be scanned in the experiment.
                If not set, then ``num_stark_amps`` evenly spaced amplitudes
                between ``min_stark_amp`` and ``max_stark_amp`` are used. If ``stark_amps``
                is set, these parameters are ignored.
        """
        options = super()._default_experiment_options()
        options.update_options(
            stark_channel=None,
            stark_freq_offset=80e6,
            stark_sigma=15e-9,
            stark_risefall=2,
            stark_length=50e-9,
            min_stark_amp=-1.0,
            max_stark_amp=1.0,
            num_stark_amps=101,
            stark_amps=None,
        )
        options.set_validator("stark_freq_offset", (0, np.inf))
        options.set_validator("stark_channel", pulse.channels.PulseChannel)
        return options

    def _set_backend(self, backend: Backend):
        super()._set_backend(backend)
        self._timing = BackendTiming(backend)

    def parameters(self) -> np.ndarray:
        """Stark tone amplitudes to use in circuits.

        Returns:
            The list of amplitudes to use for the different circuits based on the
            experiment options.
        """
        opt = self.experiment_options  # alias

        if opt.stark_amps is None:
            params = np.linspace(opt.min_stark_amp, opt.max_stark_amp, opt.num_stark_amps)
        else:
            params = np.asarray(opt.stark_amps, dtype=float)

        return params

    def parameterized_circuits(self) -> tuple[QuantumCircuit, ...]:
        """Create circuits with parameters for Ramsey XY experiment with Stark tone.

        Returns:
            Quantum template circuits for Ramsey X and Ramsey Y experiment.
        """
        opt = self.experiment_options  # alias
        param = Parameter("stark_amp")
        sym_param = param._symbol_expr

        # Pulse gates
        stark_v = Gate("StarkV", 1, [param])
        stark_u = Gate("StarkU", 1, [param])

        # Note that Stark tone yields negative (positive) frequency shift
        # when the Stark tone frequency is higher (lower) than qubit f01 frequency.
        # This choice gives positive frequency shift with positive Stark amplitude.
        qubit_f01 = self._backend_data.drive_freqs[self.physical_qubits[0]]
        neg_sign_of_amp = ParameterExpression(
            symbol_map={param: sym_param},
            expr=-sym.sign(sym_param),
        )
        abs_of_amp = ParameterExpression(
            symbol_map={param: sym_param},
            expr=sym.Abs(sym_param),
        )
        stark_freq = qubit_f01 + neg_sign_of_amp * opt.stark_freq_offset
        stark_channel = opt.stark_channel or pulse.DriveChannel(self.physical_qubits[0])
        ramps_dt = self._timing.round_pulse(time=2 * opt.stark_risefall * opt.stark_sigma)
        sigma_dt = ramps_dt / 2 / opt.stark_risefall
        width_dt = self._timing.round_pulse(time=opt.stark_length)

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
                    duration=ramps_dt + width_dt,
                    amp=abs_of_amp,
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

    def circuits(self) -> list[QuantumCircuit]:
        """Create circuits.

        Returns:
            A list of circuits with a variable Stark tone amplitudes.
        """
        ramx_circ, ramy_circ = self.parameterized_circuits()
        param = next(iter(ramx_circ.parameters))

        circs = []
        for amp in self.parameters():
            # Add metadata "direction" to ease the filtering of the data
            # by curve analysis. Indeed, the fit parameters are amplitude sign dependent.

            ramx_circ_assigned = ramx_circ.assign_parameters({param: amp}, inplace=False)
            ramx_circ_assigned.metadata["xval"] = amp
            ramx_circ_assigned.metadata["direction"] = "pos" if amp > 0 else "neg"

            ramy_circ_assigned = ramy_circ.assign_parameters({param: amp}, inplace=False)
            ramy_circ_assigned.metadata["xval"] = amp
            ramy_circ_assigned.metadata["direction"] = "pos" if amp > 0 else "neg"

            circs.extend([ramx_circ_assigned, ramy_circ_assigned])

        return circs

    def _metadata(self) -> dict[str, any]:
        """Return experiment metadata for ExperimentData."""
        metadata = super()._metadata()
        metadata["stark_length"] = self._timing.pulse_time(
            time=self.experiment_options.stark_length
        )
        metadata["stark_freq_offset"] = self.experiment_options.stark_freq_offset

        return metadata
