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
"""
T1 Experiment class.
"""

from typing import List, Tuple, Dict, Optional, Union, Sequence

import numpy as np
from qiskit import pulse
from qiskit.circuit import QuantumCircuit, Gate, Parameter, ParameterExpression
from qiskit.providers.backend import Backend
from qiskit.utils import optionals as _optional

from qiskit_experiments.framework import BackendTiming, BaseExperiment, ExperimentData, Options
from qiskit_experiments.library.characterization.analysis.t1_analysis import (
    T1Analysis,
    StarkP1SpectAnalysis,
)

if _optional.HAS_SYMENGINE:
    import symengine as sym
else:
    import sympy as sym


class T1(BaseExperiment):
    r"""An experiment to measure the qubit relaxation time.

    # section: overview

        This experiment estimates the :math:`T_1` relaxation time of the qubit by
        generating a series of circuits that excite the qubit then wait for different
        intervals before measurement. The resulting data of excited population versus
        wait time is fitted to an exponential curve to obtain an estimate for
        :math:`T_1`.

    # section: analysis_ref
        :class:`.T1Analysis`

    # section: manual
        :doc:`/manuals/characterization/t1`

    """

    @classmethod
    def _default_experiment_options(cls) -> Options:
        """Default experiment options.

        Experiment Options:
            delays (Iterable[float]): Delay times of the experiments in seconds.
        """
        options = super()._default_experiment_options()
        options.delays = None
        return options

    def __init__(
        self,
        physical_qubits: Sequence[int],
        delays: Union[List[float], np.array],
        backend: Optional[Backend] = None,
    ):
        """
        Initialize the T1 experiment class.

        Args:
            physical_qubits: a single-element sequence containing the qubit whose T1 is to be
                estimated.
            delays: Delay times of the experiments in seconds.
            backend: Optional, the backend to run the experiment on.

        Raises:
            ValueError: If the number of delays is smaller than 3
        """
        # Initialize base experiment
        super().__init__(physical_qubits, analysis=T1Analysis(), backend=backend)

        # Set experiment options
        self.set_experiment_options(delays=delays)

    def circuits(self) -> List[QuantumCircuit]:
        """
        Return a list of experiment circuits

        Returns:
            The experiment circuits
        """
        timing = BackendTiming(self.backend)

        circuits = []
        for delay in self.experiment_options.delays:
            circ = QuantumCircuit(1, 1)
            circ.x(0)
            circ.barrier(0)
            circ.delay(timing.round_delay(time=delay), 0, timing.delay_unit)
            circ.barrier(0)
            circ.measure(0, 0)

            circ.metadata = {
                "experiment_type": self._type,
                "qubit": self.physical_qubits[0],
                "unit": "s",
            }
            circ.metadata["xval"] = timing.delay_time(time=delay)

            circuits.append(circ)

        return circuits

    def _metadata(self):
        metadata = super()._metadata()
        # Store measurement level and meas return if they have been
        # set for the experiment
        for run_opt in ["meas_level", "meas_return"]:
            if hasattr(self.run_options, run_opt):
                metadata[run_opt] = getattr(self.run_options, run_opt)
        return metadata


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
        :py:class:`.StarkP1SpectAnalysis`

    # section: reference
        .. ref_arxiv:: 1 2105.15201

    # section: see_also
        :class:`qiskit_experiments.library.characterization.ramsey_xy.StarkRamseyXY`
        :class:`qiskit_experiments.library.characterization.ramsey_xy.StarkRamseyXYAmpScan`

    # section: manual
        :doc:`/manuals/characterization/stark_experiment`

    """

    def __init__(
        self,
        physical_qubits: Sequence[int],
        backend: Optional[Backend] = None,
        **experiment_options,
    ):
        """
        Initialize the T1 experiment class.

        Args:
            physical_qubits: Sequence with the index of the physical qubit.
            backend: Optional, the backend to run the experiment on.
            experiment_options: Experiment options. See the class documentation or
                ``self._default_experiment_options`` for descriptions.
        """
        self._timing = None

        super().__init__(
            physical_qubits=physical_qubits,
            analysis=StarkP1SpectAnalysis(),
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
            service (IBMExperimentService): A valid experiment service instance that can
                provide the Stark coefficients for the qubit to run experiment.
                This is required only when ``stark_coefficients`` is ``latest`` and
                ``xval_type`` is ``frequency``. This value is automatically set when
                a backend is attached to this experiment instance.
            stark_coefficients (Union[Dict, str]): Dictionary of Stark shift coefficients to
                convert tone amplitudes into amount of Stark shift. This dictionary must include
                all keys defined in :attr:`.StarkP1SpectAnalysis.stark_coefficients_names`,
                which are calibrated with :class:`.StarkRamseyXYAmpScan`.
                Alternatively, it searches for these coefficients in the result database
                when "latest" is set. This requires having the experiment service set in
                the experiment data to analyze.
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
            service=None,
            stark_coefficients="latest",
        )
        options.set_validator("spacing", ["linear", "quadratic"])
        options.set_validator("xval_type", ["amplitude", "frequency"])
        options.set_validator("stark_freq_offset", (0, np.inf))
        options.set_validator("stark_channel", pulse.channels.PulseChannel)
        return options

    def _set_backend(self, backend: Backend):
        super()._set_backend(backend)
        self._timing = BackendTiming(backend)
        if self.experiment_options.service is None:
            self.set_experiment_options(
                service=ExperimentData.get_service_from_backend(backend),
            )

    def parameters(self) -> np.ndarray:
        """Stark tone amplitudes to use in circuits.

        Returns:
            The list of amplitudes to use for the different circuits based on the
            experiment options.
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
            return self._frequencies_to_amplitudes(params)
        return params

    def _frequencies_to_amplitudes(self, params: np.ndarray) -> np.ndarray:
        """A helper method to convert frequency values to amplitude.

        Args:
            params: Parameters representing a frequency of Stark shift.

        Returns:
            Corresponding Stark tone amplitudes.

        Raises:
            RuntimeError: When service or analysis results for Stark coefficients are not available.
            TypeError: When attached analysis class is not valid.
            KeyError: When stark_coefficients dictionary is provided but keys are missing.
            ValueError: When specified Stark shift is not available.
        """
        opt = self.experiment_options  # alias

        if not isinstance(self.analysis, StarkP1SpectAnalysis):
            raise TypeError(
                f"Analysis class {self.analysis.__class__.__name__} is not a subclass of "
                "StarkP1SpectAnalysis. Use proper analysis class to scan frequencies."
            )
        coef_names = self.analysis.stark_coefficients_names

        if opt.stark_coefficients == "latest":
            if opt.service is None:
                raise RuntimeError(
                    "Experiment service is not available. Provide a dictionary of "
                    "Stark coefficients in the experiment options."
                )
            coefficients = self.analysis.retrieve_coefficients_from_service(
                service=opt.service,
                qubit=self.physical_qubits[0],
                backend=self._backend_data.name,
            )
            if coefficients is None:
                raise RuntimeError(
                    "Experiment results for the coefficients of the Stark shift is not found "
                    f"for the backend {self._backend_data.name} qubit {self.physical_qubits}."
                )
        else:
            missing = set(coef_names) - opt.stark_coefficients.keys()
            if any(missing):
                raise KeyError(
                    f"Following coefficient data is missing in the 'stark_coefficients': {missing}."
                )
            coefficients = opt.stark_coefficients
        positive = np.asarray([coefficients[coef_names[idx]] for idx in [2, 1, 0]])
        negative = np.asarray([coefficients[coef_names[idx]] for idx in [5, 4, 3]])
        offset = coefficients[coef_names[6]]

        amplitudes = np.zeros_like(params)
        for idx, tgt_freq in enumerate(params):
            stark_shift = tgt_freq - offset
            if np.isclose(stark_shift, 0):
                amplitudes[idx] = 0
                continue
            if np.sign(stark_shift) > 0:
                fit_coeffs = [*positive, -stark_shift]
            else:
                fit_coeffs = [*negative, -stark_shift]
            amp_candidates = np.roots(fit_coeffs)
            # Because the fit function is third order, we get three solutions here.
            # Only one valid solution must exist because we assume
            # a monotonic trend for Stark shift against tone amplitude in domain of definition.
            criteria = np.all(
                [
                    # Frequency shift and tone have the same sign by definition
                    np.sign(amp_candidates.real) == np.sign(stark_shift),
                    # Tone amplitude is a real value
                    np.isclose(amp_candidates.imag, 0.0),
                    # The absolute value of tone amplitude must be less than 1.0
                    np.abs(amp_candidates.real) < 1.0,
                ],
                axis=0,
            )
            valid_amps = amp_candidates[criteria]
            if len(valid_amps) == 0:
                raise ValueError(
                    f"Stark shift at frequency value of {tgt_freq} Hz is not available on "
                    f"the backend {self._backend_data.name} qubit {self.physical_qubits}."
                )
            if len(valid_amps) > 1:
                # We assume a monotonic trend but sometimes a large third-order term causes
                # inflection point and inverts the trend in larger amplitudes.
                # In this case we would have more than one solutions, but we can
                # take the smallerst amplitude before reaching to the inflection point.
                before_inflection = np.argmin(np.abs(valid_amps.real))
                valid_amp = float(valid_amps[before_inflection].real)
            else:
                valid_amp = float(valid_amps.real)
            amplitudes[idx] = valid_amp

        return amplitudes

    def parameterized_circuits(self) -> Tuple[QuantumCircuit, ...]:
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

    def circuits(self) -> List[QuantumCircuit]:
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

    def _metadata(self) -> Dict[str, any]:
        """Return experiment metadata for ExperimentData."""
        metadata = super()._metadata()
        metadata["stark_freq_offset"] = self.experiment_options.stark_freq_offset

        return metadata
