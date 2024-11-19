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
"""P1 experiment at various qubit frequencies."""

from __future__ import annotations

import warnings
from collections.abc import Sequence

import numpy as np
from qiskit import pulse
from qiskit.circuit import QuantumCircuit, Gate, Parameter, ParameterExpression
from qiskit.providers.backend import Backend
from qiskit.utils import optionals as _optional
from qiskit.utils.deprecation import deprecate_func

from qiskit_experiments.framework import BackendTiming, BaseExperiment, Options
from .p1_spect_analysis import StarkP1SpectAnalysis

from .coefficients import (
    StarkCoefficients,
    retrieve_coefficients_from_backend,
)

if _optional.HAS_SYMENGINE:
    import symengine as sym
else:
    import sympy as sym


class StarkP1Spectroscopy(BaseExperiment):
    """P1 spectroscopy experiment with Stark tone.

    # section: overview

        This experiment measures a probability of the excitation state of the qubit
        with a certain delay after excitation.
        A Stark tone is applied during this delay to move the
        qubit frequency to conduct a spectroscopy of qubit relaxation quantity.

        .. parsed-literal::

                 ┌───┐┌──────────────────┐┌─┐
              q: ┤ X ├┤ Stark(stark_amp) ├┤M├
                 └───┘└──────────────────┘└╥┘
            c: 1/══════════════════════════╩═
                                           0

        Since the qubit relaxation rate may depend on the qubit frequency due to the
        coupling to nearby energy levels, this experiment is useful to find out
        lossy operation frequency that might be harmful to the gate fidelity [1].

    # section: analysis_ref
        :class:`qiskit_experiments.library.driven_freq_tuning.StarkP1SpectAnalysis`

    # section: reference
        .. ref_arxiv:: 1 2105.15201

    # section: see_also
        :class:`qiskit_experiments.library.driven_freq_tuning.ramsey.StarkRamseyXY`
        :class:`qiskit_experiments.library.driven_freq_tuning.ramsey_amp_scan.StarkRamseyXYAmpScan`

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
        """
        Initialize new experiment class.

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
            analysis = StarkP1SpectAnalysis()

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
            min_xval (float): Minimum x value.
            max_xval (float): Maximum x value.
            num_xvals (int): Number of x-values to scan.
            xval_type (str): Type of x-value. Either ``amplitude`` or ``frequency``.
                Setting to frequency requires pre-calibration of Stark shift coefficients.
            spacing (str): A policy for the spacing to create an amplitude list from
                ``min_stark_amp`` to ``max_stark_amp``. Either ``linear`` or ``quadratic``
                must be specified.
            xvals (list[float]): The list of x-values that will be scanned in the experiment.
                If not set, then ``num_xvals`` parameters spaced according to
                the ``spacing`` policy between ``min_xval`` and ``max_xval`` are used.
                If ``xvals`` is set, these parameters are ignored.
            stark_coefficients (StarkCoefficients): Calibrated Stark shift coefficients.
                This value is necessary when xval_type is "frequency".
                When this value is None, a search for the "stark_coefficients" in the
                result database is run. This requires having the experiment service
                available in the backend set for the experiment.
        """
        options = super()._default_experiment_options()
        options.update_options(
            t1_delay=20e-6,
            stark_channel=None,
            stark_freq_offset=80e6,
            stark_sigma=15e-9,
            stark_risefall=2,
            min_xval=-1.0,
            max_xval=1.0,
            num_xvals=201,
            xval_type="amplitude",
            spacing="quadratic",
            xvals=None,
            stark_coefficients=None,
        )
        options.set_validator("spacing", ["linear", "quadratic"])
        options.set_validator("xval_type", ["amplitude", "frequency"])
        options.set_validator("stark_freq_offset", (0, np.inf))
        options.set_validator("stark_channel", pulse.channels.PulseChannel)
        options.set_validator("stark_coefficients", StarkCoefficients)
        return options

    def _set_backend(self, backend: Backend):
        super()._set_backend(backend)
        self._timing = BackendTiming(backend)

    def parameters(self) -> np.ndarray:
        """Stark tone amplitudes to use in circuits.

        Returns:
            The list of amplitudes to use for the different circuits based on the
            experiment options.

        Raises:
            ValueError: When invalid xval spacing is specified.
        """
        opt = self.experiment_options  # alias

        if opt.xvals is None:
            if opt.spacing == "linear":
                params = np.linspace(opt.min_xval, opt.max_xval, opt.num_xvals)
            elif opt.spacing == "quadratic":
                min_sqrt = np.sign(opt.min_xval) * np.sqrt(np.abs(opt.min_xval))
                max_sqrt = np.sign(opt.max_xval) * np.sqrt(np.abs(opt.max_xval))
                lin_params = np.linspace(min_sqrt, max_sqrt, opt.num_xvals)
                params = np.sign(lin_params) * lin_params**2
            else:
                raise ValueError(f"Spacing option {opt.spacing} is not valid.")
        else:
            params = np.asarray(opt.xvals, dtype=float)

        if opt.xval_type == "frequency":
            coeffs = opt.stark_coefficients
            if coeffs is None:
                coeffs = retrieve_coefficients_from_backend(
                    backend=self.backend,
                    qubit=self.physical_qubits[0],
                )
            return coeffs.convert_freq_to_amp(freqs=params)
        return params

    def parameterized_circuits(self) -> tuple[QuantumCircuit, ...]:
        """Create circuits with parameters for P1 experiment with Stark shift.

        Returns:
            Quantum template circuit for P1 experiment.
        """
        opt = self.experiment_options  # alias
        param = Parameter("stark_amp")
        sym_param = param._symbol_expr

        # Pulse gates
        stark = Gate("Stark", 1, [param])

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

        temp_t1 = QuantumCircuit(1, 1)
        temp_t1.x(0)
        temp_t1.append(stark, [0])
        temp_t1.measure(0, 0)
        temp_t1.add_calibration(
            gate=stark,
            qubits=self.physical_qubits,
            schedule=stark_schedule,
        )

        return (temp_t1,)

    def circuits(self) -> list[QuantumCircuit]:
        """Create circuits.

        Returns:
            A list of P1 circuits with a variable Stark tone amplitudes.
        """
        (t1_circ,) = self.parameterized_circuits()
        param = next(iter(t1_circ.parameters))

        circs = []
        for amp in self.parameters():
            t1_assigned = t1_circ.assign_parameters({param: amp}, inplace=False)
            t1_assigned.metadata = {"xval": amp}
            circs.append(t1_assigned)

        return circs

    def _metadata(self) -> dict[str, any]:
        """Return experiment metadata for ExperimentData."""
        metadata = super()._metadata()
        metadata["stark_freq_offset"] = self.experiment_options.stark_freq_offset

        return metadata
