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
Cross resonance Hamiltonian tomography.
"""

import warnings
from typing import List, Tuple, Sequence, Optional, Type

import numpy as np
from qiskit import pulse, circuit, QuantumCircuit
from qiskit.circuit.parameterexpression import ParameterValueType
from qiskit.exceptions import QiskitError
from qiskit.providers import Backend
from qiskit.utils.deprecation import deprecate_func
from qiskit_experiments.framework import (
    BaseExperiment,
    BackendTiming,
    Options,
)
from qiskit_experiments.library.characterization.analysis import CrossResonanceHamiltonianAnalysis


class CrossResonanceHamiltonian(BaseExperiment):
    r"""Cross resonance Hamiltonian tomography experiment.

    # section: overview

        This experiment assumes the two qubit Hamiltonian in the form

        .. math::

            H = \frac{I \otimes A}{2} + \frac{Z \otimes B}{2}

        where :math:`A` and :math:`B` are linear combinations of
        the Pauli operators :math:`\in {X, Y, Z}`.
        The coefficient of each Pauli term in the Hamiltonian
        can be estimated with this experiment.

        This experiment is performed by stretching the pulse duration of a cross resonance pulse
        and measuring the target qubit by projecting onto the x, y, and z bases.
        The control qubit state dependent (controlled-) Rabi oscillation on the
        target qubit is observed by repeating the experiment with the control qubit
        both in the ground and excited states. The fit for the oscillations in the
        three bases with the two control qubit preparations tomographically
        reconstructs the Hamiltonian in the form shown above.
        See Ref. [1] for more details.

        More specifically, the following circuits are executed in this experiment.

        .. parsed-literal::

            (X measurement)

                 ┌───┐┌────────────────────┐
            q_0: ┤ P ├┤0                   ├────────────────────
                 └───┘│  cr_tone(duration) │┌─────────┐┌────┐┌─┐
            q_1: ─────┤1                   ├┤ Rz(π/2) ├┤ √X ├┤M├
                      └────────────────────┘└─────────┘└────┘└╥┘
            c: 1/═════════════════════════════════════════════╩═
                                                              0

            (Y measurement)

                 ┌───┐┌────────────────────┐
            q_0: ┤ P ├┤0                   ├─────────
                 └───┘│  cr_tone(duration) │┌────┐┌─┐
            q_1: ─────┤1                   ├┤ √X ├┤M├
                      └────────────────────┘└────┘└╥┘
            c: 1/══════════════════════════════════╩═
                                                   0

            (Z measurement)

                 ┌───┐┌────────────────────┐
            q_0: ┤ P ├┤0                   ├───
                 └───┘│  cr_tone(duration) │┌─┐
            q_1: ─────┤1                   ├┤M├
                      └────────────────────┘└╥┘
            c: 1/════════════════════════════╩═
                                             0

        The ``P`` gate on the control qubit (``q_0``) indicates the state preparation.
        Since this experiment requires two sets of sub experiments with the control qubit in the
        excited and ground state, ``P`` will become ``X`` gate or just be omitted, respectively.
        Here ``cr_tone`` is implemented by a single cross resonance tone
        driving the control qubit at the frequency of the target qubit.
        The pulse envelope might be a flat-topped Gaussian implemented by the parametric pulse
        :class:`~qiskit.pulse.library.parametric_pulses.GaussianSquare`.

        This experiment scans the total duration of the cross resonance pulse
        including the pulse ramps at both edges. The pulse shape is defined by the
        :class:`~qiskit.pulse.library.parametric_pulses.GaussianSquare`, and
        an effective length of these Gaussian ramps with :math:`\sigma` can be computed by

        .. math::

            \tau_{\rm edges}' = \sqrt{2 \pi} \sigma,

        which is usually shorter than the actual edge duration of

        .. math::

            \tau_{\rm edges} = 2 r \sigma,

        where the :math:`r` is the ratio of the actual edge duration to :math:`\sigma`.
        This effect must be considered in the following curve analysis to estimate
        interaction rates.

    # section: analysis_ref
        :class:`CrossResonanceHamiltonianAnalysis`

    # section: reference
        .. ref_arxiv:: 1 1603.04821

    # section: manual
        .. ref_website:: Qiskit Textbook 6.7,
            https://github.com/Qiskit/textbook/blob/main/notebooks/quantum-hardware-pulses/hamiltonian-tomography.ipynb
    """

    # Number of CR pulses. The flat top duration per pulse is divided by this number.
    num_pulses = 1

    class CRPulseGate(circuit.Gate):
        """A pulse gate of cross resonance. Definition should be provided via calibration."""

        def __init__(self, width: ParameterValueType):
            super().__init__("cr_gate", 2, [width])

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
        physical_qubits: Tuple[int, int],
        backend: Optional[Backend] = None,
        cr_gate: Optional[Type[circuit.Gate]] = None,
        durations: Optional[Sequence[int]] = None,
        **kwargs,
    ):
        """Create a new experiment.

        Args:
            physical_qubits: Two-value tuple of qubit indices on which to run tomography.
                The first index stands for the control qubit.
            backend: Optional, the backend to run the experiment on.
            cr_gate: Optional, circuit gate class representing the cross resonance pulse.
                Providing this object allows us to run this experiment with circuit simulator,
                and this object might be used for testing, development of analysis protocol,
                and educational purpose without needing to wait for hardware queueing.
                Note that this instance must provide matrix representation, such as
                unitary gate or Hamiltonian gate, and the class is expected to be instantiated
                with a single parameter ``width`` in units of sec.
            durations: Optional. The total duration of cross resonance pulse(s) including
                rising and falling edges. The minimum number should be larger than the
                total lengths of these ramps. If not provided, then ``num_durations`` evenly
                spaced durations between ``min_durations`` and ``max_durations`` are
                automatically generated from these experiment options. The default numbers
                are chosen to have a good sensitivity for the Hamiltonian coefficient
                of interest at the rate around 1 MHz.
                This argument should be provided in units of sec.
            kwargs: Pulse parameters. See :meth:`experiment_options` for details.

        Raises:
            QiskitError: When ``qubits`` length is not 2.
        """
        if len(physical_qubits) != 2:
            raise QiskitError(
                "Length of qubits is not 2. Please provide index for control and target qubit."
            )

        self._gate_cls = cr_gate or self.CRPulseGate
        self._backend_timing = None

        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="deprecation of Qiskit Pulse",
                module="qiskit_experiments",
                category=DeprecationWarning,
            )
            analysis = CrossResonanceHamiltonianAnalysis()
        super().__init__(physical_qubits, analysis=analysis, backend=backend)
        self.set_experiment_options(durations=durations, **kwargs)

    @classmethod
    def _default_experiment_options(cls) -> Options:
        """Default experiment options.

        Experiment Options:
            durations (np.ndarray): The total duration of the cross resonance pulse(s) to scan,
                in units of sec. Values should be longer than pulse ramps.
            min_durations (int): The minimum default pulse duration in samples.
            max_durations (int): The maximum default pulse duration in samples.
            num_durations (int): The number of measured durations. The experiment automatically
                creates durations of linear increment along with ``min_durations`` and
                ``max_durations`` when user doesn't explicitly provide ``durations``.
            amp (complex): Amplitude of the cross resonance tone.
            amp_t (complex): Amplitude of the cancellation or rotary drive on target qubit.
            sigma (float): Sigma of Gaussian rise and fall edges, in units of dt.
            risefall (float): Ratio of edge durations to sigma.
        """
        options = super()._default_experiment_options()
        options.durations = None
        options.min_durations = 60e-9
        options.max_durations = 1200e-9
        options.num_durations = 48
        options.amp = 0.5
        options.amp_t = 0.0
        options.sigma = 64
        options.risefall = 2

        return options

    def _set_backend(self, backend: Backend):
        """Set the backend for the experiment with timing analysis."""
        super()._set_backend(backend)
        self._backend_timing = BackendTiming(backend)

    def _get_dt(self) -> float:
        """A helper function to get finite dt.

        Returns:
            Backend dt value.
        """
        if not self._backend or self._backend_timing.dt is None:
            # When backend timing is not initialized or backend doesn't report dt.
            return 1.0
        return self._backend_timing.dt

    def _get_width(self, duration: ParameterValueType) -> ParameterValueType:
        """A helper function to get flat top width.

        Args:
            duration: Cross resonance pulse duration in units of sec.

        Returns:
            A flat top widths of cross resonance pulse in units of sec.
        """
        sigma_sec = self.experiment_options.sigma * self._get_dt()

        return duration - 2 * sigma_sec * self.experiment_options.risefall

    def _get_durations(self) -> np.ndarray:
        """Return cross resonance pulse durations in units of sec."""
        opt = self.experiment_options

        if opt.durations is None:
            return np.linspace(opt.min_durations, opt.max_durations, opt.num_durations)

        return np.asarray(opt.durations, dtype=float)

    def _build_cr_circuit(self, pulse_gate: circuit.Gate) -> QuantumCircuit:
        """Single tone cross resonance.

        Args:
            pulse_gate: A pulse gate to represent a single cross resonance pulse.

        Returns:
            A circuit definition for the cross resonance pulse to measure.
        """
        cr_circuit = QuantumCircuit(2)
        cr_circuit.append(pulse_gate, [0, 1])

        return cr_circuit

    def _build_default_schedule(self) -> pulse.ScheduleBlock:
        """GaussianSquared cross resonance pulse.

        Returns:
            A schedule definition for the cross resonance pulse to measure.
        """
        opt = self.experiment_options
        duration = circuit.Parameter("duration")

        cr_drive = self._backend_data.control_channel(self.physical_qubits)[0]
        c_drive = self._backend_data.drive_channel(self.physical_qubits[0])
        t_drive = self._backend_data.drive_channel(self.physical_qubits[1])

        with pulse.build(default_alignment="left", name="cr") as cross_resonance:
            # add cross resonance tone
            pulse.play(
                pulse.GaussianSquare(
                    duration=duration,
                    amp=opt.amp,
                    sigma=opt.sigma,
                    risefall_sigma_ratio=opt.risefall,
                ),
                cr_drive,
            )
            # add cancellation tone
            if not np.isclose(opt.amp_t, 0.0):
                pulse.play(
                    pulse.GaussianSquare(
                        duration=duration,
                        amp=opt.amp_t,
                        sigma=opt.sigma,
                        risefall_sigma_ratio=opt.risefall,
                    ),
                    t_drive,
                )
            else:
                pulse.delay(duration, t_drive)

            # place holder for empty drive channels. this is necessary due to known pulse gate bug.
            pulse.delay(duration, c_drive)

        return cross_resonance

    def circuits(self) -> List[QuantumCircuit]:
        """Return a list of experiment circuits.

        Returns:
            A list of :class:`QuantumCircuit`.

        Raises:
            QiskitError: When the backend is not set and cr gate is ``CRPulseGate`` type.
        """
        if self._gate_cls is self.CRPulseGate:
            if not self.backend:
                # Backend is not set, but trying to provide CR gate as a pulse gate.
                raise QiskitError(
                    "This experiment requires to have backend set to convert durations into samples "
                    "with backend reported dt value and also it requires the channel mapping from "
                    "the backend to build cross resonance pulse schedule. "
                    "Please provide valid backend object supporting 2Q pulse gate."
                )
            return self._pulse_gate_circuits()
        return self._unitary_circuits()

    def _pulse_gate_circuits(self):
        """Protocol to create circuits with pulse gate.

        Pulse gate has backend timing constraints and duration should be in units of dt.
        This method calls :meth:`_build_default_schedule` to generate actual schedule.
        We assume backend has been set in this method call.
        """
        schedule = self._build_default_schedule()

        # Assume this parameter is in units of dt, because this controls pulse samples.
        param_duration = next(iter(schedule.get_parameters("duration")))

        # Gate duration will be shown in sec, which is more intuitive.
        cr_gate = self._gate_cls(width=self._get_width(self._backend_timing.dt * param_duration))

        # Create parameterized circuits with calibration.
        tmp_circs = []
        for control_state in (0, 1):
            for meas_basis in ("x", "y", "z"):
                tmp_qc = QuantumCircuit(2, 1)
                if control_state:
                    tmp_qc.x(0)
                tmp_qc.compose(
                    other=self._build_cr_circuit(cr_gate),
                    qubits=[0, 1],
                    inplace=True,
                )
                if meas_basis == "x":
                    tmp_qc.rz(np.pi / 2, 1)
                if meas_basis in ("x", "y"):
                    tmp_qc.sx(1)
                tmp_qc.measure(1, 0)
                tmp_qc.metadata = {
                    "control_state": control_state,
                    "meas_basis": meas_basis,
                }
                tmp_qc.add_calibration(cr_gate, self.physical_qubits, schedule)
                tmp_circs.append(tmp_qc)

        circs = []
        for duration in self._get_durations():
            # Need to round pulse to satisfy hardware timing constraints.
            # Convert into samples for assignment and validation.
            valid_duration_dt = self._backend_timing.round_pulse(time=duration)

            # Convert into sec to pass xval to analysis.
            # Analysis expects xval of flat top widths in units of sec.
            flat_top_width_sec = self._get_width(self._backend_timing.dt * valid_duration_dt)
            if flat_top_width_sec < 0:
                raise ValueError(
                    f"Input duration={duration} is less than pulse ramps lengths, resulting in "
                    f"a negative flat top length of {flat_top_width_sec} sec. "
                    f"This cross resonance schedule is invalid."
                )

            for circ in tmp_circs:
                # Assign duration in dt to create pulse schedule.
                assigned_circ = circ.assign_parameters(
                    {param_duration: valid_duration_dt},
                    inplace=False,
                )
                assigned_circ.metadata["xval"] = self.num_pulses * flat_top_width_sec
                circs.append(assigned_circ)

        return circs

    def _unitary_circuits(self):
        """Protocol to create circuits with unitary gate.

        Unitary gate has no timing constraints and accepts duration in sec.
        Basically, this method doesn't require backend apart from conversion of
        sigma in samples into sec.
        """
        # Assume this parameter is in units of sec.
        param_duration = circuit.Parameter("duration")

        # Gate duration will be shown in sec, which is more intuitive.
        cr_gate = self._gate_cls(width=self._get_width(param_duration))

        # Create parameterized circuits without calibration.
        tmp_circs = []
        for control_state in (0, 1):
            for meas_basis in ("x", "y", "z"):
                tmp_qc = QuantumCircuit(2, 1)
                if control_state:
                    tmp_qc.x(0)
                tmp_qc.compose(
                    other=self._build_cr_circuit(cr_gate),
                    qubits=[0, 1],
                    inplace=True,
                )
                if meas_basis == "x":
                    tmp_qc.rz(np.pi / 2, 1)
                if meas_basis in ("x", "y"):
                    tmp_qc.sx(1)
                tmp_qc.measure(1, 0)
                tmp_qc.metadata = {
                    "control_state": control_state,
                    "meas_basis": meas_basis,
                }
                tmp_circs.append(tmp_qc)

        circs = []
        for duration in self._get_durations():
            flat_top_width_sec = self._get_width(duration)
            if flat_top_width_sec < 0:
                raise ValueError(
                    f"Input duration={duration} is less than pulse ramps lengths, resulting in "
                    f"a negative flat top length of {flat_top_width_sec} sec. "
                    f"This cross resonance schedule is invalid."
                )

            for circ in tmp_circs:
                # Assign duration in sec since this is unitary gate.
                assigned_circ = circ.assign_parameters(
                    {param_duration: duration},
                    inplace=False,
                )
                assigned_circ.metadata["xval"] = self.num_pulses * flat_top_width_sec
                circs.append(assigned_circ)

        return circs

    def _finalize(self):
        """Set analysis option for initial guess that depends on experiment option values."""
        edge_duration = np.sqrt(2 * np.pi) * self.experiment_options.sigma * self.num_pulses

        for analysis in self.analysis.analyses():
            init_guess = analysis.options.p0.copy()
            if "t_off" in init_guess:
                continue
            init_guess["t_off"] = self._get_dt() * edge_duration
            analysis.set_options(p0=init_guess)

    def _metadata(self):
        metadata = super()._metadata()
        # Store measurement level and meas return if they have been
        # set for the experiment
        for run_opt in ["meas_level", "meas_return"]:
            if hasattr(self.run_options, run_opt):
                metadata[run_opt] = getattr(self.run_options, run_opt)
        return metadata


class EchoedCrossResonanceHamiltonian(CrossResonanceHamiltonian):
    r"""Echoed cross resonance Hamiltonian tomography experiment.

    # section: overview

        This is a variant of :class:`CrossResonanceHamiltonian`
        for which the experiment framework is identical but the
        cross resonance operation is realized as an echoed sequence
        to remove unwanted single qubit rotations. The cross resonance
        circuit looks like:

        .. parsed-literal::

                 ┌────────────────────┐  ┌───┐  ┌────────────────────┐
            q_0: ┤0                   ├──┤ X ├──┤0                   ├──────────
                 │  cr_tone(duration) │┌─┴───┴─┐│  cr_tone(duration) │┌────────┐
            q_1: ┤1                   ├┤ Rz(π) ├┤1                   ├┤ Rz(-π) ├
                 └────────────────────┘└───────┘└────────────────────┘└────────┘

        Here two ``cr_tone`` are applied, where the latter one is with the
        control qubit state flipped and with a phase flip of the target qubit frame.
        This operation is equivalent to applying the ``cr_tone`` with a negative amplitude.
        The Hamiltonian for this decomposition has no IX and ZI interactions,
        and also a reduced IY interaction to some extent (not completely eliminated) [1].
        Note that the CR Hamiltonian tomography experiment cannot detect the ZI term.
        However, it is sensitive to the IX and IY terms.

    # section: reference
        .. ref_arxiv:: 1 2007.02925

    """

    num_pulses = 2

    def _build_cr_circuit(self, pulse_gate: circuit.Gate) -> QuantumCircuit:
        """Single tone cross resonance.

        Args:
            pulse_gate: A pulse gate to represent a single cross resonance pulse.

        Returns:
            A circuit definition for the cross resonance pulse to measure.
        """
        cr_circuit = QuantumCircuit(2)
        cr_circuit.append(pulse_gate, [0, 1])
        cr_circuit.x(0)
        cr_circuit.rz(np.pi, 1)
        cr_circuit.append(pulse_gate, [0, 1])
        cr_circuit.rz(-np.pi, 1)

        return cr_circuit
