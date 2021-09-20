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

from typing import List, Tuple, Optional, Iterable

import numpy as np
from qiskit import pulse, circuit, QuantumCircuit
from qiskit.exceptions import QiskitError
from qiskit.providers import Backend
from qiskit.utils import apply_prefix

from qiskit_experiments.framework import BaseExperiment, Options
from .cr_hamiltonian_analysis import CrossResonanceHamiltonianAnalysis


class CrossResonanceHamiltonian(BaseExperiment):
    r"""Cross resonance Hamiltonian tomography experiment.

    # section: overview

        This experiment assumes the two qubit Hamiltonian in the form

        .. math::

            H = \frac{I \otimes A}{2} + \frac{Z \otimes B}{2}

        where :math:`A` and :math:`B` are Pauli operator :math:`\in {X, Y, Z}`.
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
            q_0: ┤ X ├┤0                   ├────────
                 └───┘│  cr_tone(duration) │┌───┐┌─┐
            q_1: ─────┤1                   ├┤ H ├┤M├
                      └────────────────────┘└───┘└╥┘
            c: 1/═════════════════════════════════╩═
                                                  0

            (Y measurement)

                 ┌───┐┌────────────────────┐
            q_0: ┤ X ├┤0                   ├───────────────
                 └───┘│  cr_tone(duration) │┌─────┐┌───┐┌─┐
            q_1: ─────┤1                   ├┤ Sdg ├┤ H ├┤M├
                      └────────────────────┘└─────┘└───┘└╥┘
            c: 1/════════════════════════════════════════╩═
                                                         0

            (Z measurement)

                 ┌───┐┌────────────────────┐
            q_0: ┤ X ├┤0                   ├───
                 └───┘│  cr_tone(duration) │┌─┐
            q_1: ─────┤1                   ├┤M├
                      └────────────────────┘└╥┘
            c: 1/════════════════════════════╩═
                                             0

        The ``X`` gate on the control qubit (``q_0``) depends on the required control state.
        Here ``cr_tone`` is implemented by a single cross resonance tone
        driving the control qubit at the frequency of the target qubit.
        The pulse envelope is the flat-topped Gaussian implemented by the parametric pulse
        :py:class:`~qiskit.pulse.library.parametric_pulses.GaussianSquare`.
        The effect of pulse edges are also considered as a net cross resonance pulse duration.
        The net duration of pulse edges are approximated by

        .. math::

            \tau_{\rm edges} = \sqrt{2 \pi \sigma^2},

        where the :math:`\sigma` is the sigma of Gaussian rising and falling edges.
        Thus the net duration of the entire cross resonance pulse will become

        .. math::

            \tau = {\rm duration} - 2 r \sigma + \tau_{\rm edges},

        where :math:`r` is the ratio of the actual edge duration to the sigma.

    # section: reference
        .. ref_arxiv:: 1 1603.04821

    # section: tutorial
        .. ref_website:: Qiskit Textbook 6.7,
            https://qiskit.org/textbook/ch-quantum-hardware/hamiltonian-tomography.html
    """

    __analysis_class__ = CrossResonanceHamiltonianAnalysis

    def __init__(
        self, qubits: Tuple[int, int], durations: Iterable[float], unit: str = "dt", **kwargs
    ):
        """Create a new experiment.

        Args:
            qubits: Two-value tuple of qubit indices on which to run tomography.
                The first index stands for the control qubit.
            durations: The pulse durations to scan.
            unit: The time unit of durations.
            kwargs: Pulse parameters. See :meth:`experiment_options` for details.

        Raises:
            QiskitError: When ``qubits`` length is not 2.
        """
        super().__init__(qubits=qubits)

        if len(qubits) != 2:
            raise QiskitError(
                "Length of qubits is not 2. Please provide index for control and target qubit."
            )

        self.set_experiment_options(durations=durations, unit=unit, **kwargs)

    @classmethod
    def _default_experiment_options(cls) -> Options:
        """Default experiment options.

        Experiment Options:
            durations (np.ndarray): The length of the cross resonance tone.
            unit (str): Time unit of durations.
            amp (complex): Amplitude of the cross resonance tone.
            amp_t (complex): Amplitude of the cancellation or rotary drive on target qubit.
            sigma (float): Sigma of Gaussian rise and fall edges.
            risefall (float): Ratio of edge durations to sigma.
        """
        options = super()._default_experiment_options()
        options.durations = None
        options.unit = "dt"
        options.amp = 0.2
        options.amp_t = 0.0
        options.sigma = 64
        options.risefall = 2

        return options

    def _build_cr_circuit(
        self,
        backend: Backend,
        duration: circuit.Parameter,
        sigma: float,
    ) -> QuantumCircuit:
        """Single tone cross resonance.

        Args:
            backend: The target backend.
            duration: Parameter object representing a duration of cross resonance pulse.
            sigma: Sigma of Gaussian edges.

        Returns:
            A circuit decomposition for the cross resonance pulse to measure.
        """
        cr_gate = circuit.Gate("cr_tone", num_qubits=2, params=[duration])

        cr_circuit = QuantumCircuit(2)
        cr_circuit.append(cr_gate, [0, 1])

        opt = self.experiment_options
        with pulse.build(backend, default_alignment="left", name="cr") as cross_resonance:
            # add cross resonance tone
            pulse.play(
                pulse.GaussianSquare(
                    duration,
                    amp=opt.amp,
                    sigma=sigma,
                    risefall_sigma_ratio=opt.risefall,
                ),
                pulse.control_channels(*self.physical_qubits)[0],
            )
            # add cancellation tone
            if not np.isclose(opt.amp_t, 0.0):
                pulse.play(
                    pulse.GaussianSquare(
                        duration,
                        amp=opt.amp_t,
                        sigma=sigma,
                        risefall_sigma_ratio=opt.risefall,
                    ),
                    pulse.drive_channel(self.physical_qubits[1]),
                )
            else:
                pulse.delay(duration, pulse.drive_channel(self.physical_qubits[1]))

            # place holder for empty drive channels. this is necessary due to known pulse gate bug.
            pulse.delay(duration, pulse.drive_channel(self.physical_qubits[0]))

        cr_circuit.add_calibration(
            gate=cr_gate,
            qubits=self.physical_qubits,
            schedule=cross_resonance,
            params=[duration],
        )

        return cr_circuit

    def _net_duration(
        self,
        duration: float,
        sigma: float,
    ) -> float:
        """Calculate net cross resonance pulse duration considering the pulse edges.

        Args:
            duration: Parameter object representing a duration of cross resonance pulse.
            sigma: Sigma of Gaussian edges.

        Returns:
            Net duration in units of sec.
        """
        opt = self.experiment_options

        flat_top_width = duration - 2 * opt.risefall * sigma
        net_edge_width = np.sqrt(2 * np.pi) * sigma

        return flat_top_width + net_edge_width

    def circuits(self, backend: Optional[Backend] = None) -> List[QuantumCircuit]:

        opt = self.experiment_options
        prefactor = 1.0
        timing_constraints = getattr(backend.configuration(), "timing_constraints", dict())
        granularity = timing_constraints.get("granularity", 1)

        try:
            dt_factor = backend.configuration().dt
        except AttributeError as ex:
            raise AttributeError("Backend configuration does not provide time resolution.") from ex

        if opt.unit != "dt":
            if opt.unit != "s":
                prefactor *= apply_prefix(1.0, opt.unit)
            prefactor /= dt_factor

        # Define pulse gate of cross resonance tone
        duration = circuit.Parameter("duration")
        sigma_dt = prefactor * opt.sigma

        template_circuits = list()
        for control_state in (0, 1):
            for meas_basis in ("x", "y", "z"):
                tomo_circ = QuantumCircuit(2, 1)

                # state prep
                if control_state:
                    tomo_circ.x(0)

                # add cross resonance
                tomo_circ.compose(
                    other=self._build_cr_circuit(backend, duration, sigma_dt),
                    qubits=[0, 1],
                    inplace=True,
                )

                # measure
                if meas_basis == "x":
                    tomo_circ.h(1)
                elif meas_basis == "y":
                    tomo_circ.sdg(1)
                    tomo_circ.h(1)
                tomo_circ.measure(1, 0)

                # add metadata
                tomo_circ.metadata = {"control_state": control_state, "meas_basis": meas_basis}

                template_circuits.append(tomo_circ)

        expr_circs = list()
        for dur_value in opt.durations:
            # pulse duration should be an integer value which is multiple of granularity
            dur_value = granularity * int(prefactor * dur_value / granularity)

            for template_circuit in template_circuits:
                bind_circuit = template_circuit.assign_parameters(
                    {duration: int(dur_value)}, inplace=False
                )
                bind_circuit.metadata["experiment_type"] = self._type
                bind_circuit.metadata["qubits"] = self.physical_qubits
                bind_circuit.metadata["xval"] = self._net_duration(dur_value, sigma_dt) * dt_factor
                bind_circuit.metadata["dt"] = dt_factor

                expr_circs.append(bind_circuit)

        return expr_circs


class EchoedCrossResonanceHamiltonian(CrossResonanceHamiltonian):
    r"""Echoed cross resonance Hamiltonian tomography experiment.

    # section: overview

        This is a variant of :py:class:`CrossResonanceHamiltonian`
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

        Here two ``cr_tone``s are applied where the latter one is with the
        control qubit state flipped and with a phase flip of the target qubit frame.
        This operation is equivalent to applying the ``cr_tone`` with a negative amplitude.
        The Hamiltonian for this decomposition has no IX and ZI interactions,
        and also a reduced IY interaction to some extent (not completely eliminated) [1].
        Note that the CR Hamiltonian tomography experiment cannot detect the ZI term.
        However, it is sensitive to the IX and IY terms.

    # section: reference
        .. ref_arxiv:: 1 2007.02925

    """

    def _build_cr_circuit(
        self,
        backend: Backend,
        duration: circuit.Parameter,
        sigma: float,
    ) -> QuantumCircuit:
        """Echoed cross resonance.

        Args:
            backend: The target backend.
            duration: Parameter object representing a duration of cross resonance pulse.
            sigma: Sigma of Gaussian edges.

        Returns:
            A circuit decomposition for the cross resonance pulse to measure.
        """
        cr_gate = circuit.Gate("cr_tone", num_qubits=2, params=[duration])

        cr_circuit = QuantumCircuit(2)
        cr_circuit.append(cr_gate, [0, 1])
        cr_circuit.x(0)
        cr_circuit.rz(np.pi, 1)
        cr_circuit.append(cr_gate, [0, 1])
        cr_circuit.rz(-np.pi, 1)

        opt = self.experiment_options
        with pulse.build(backend, default_alignment="left", name="cr") as cross_resonance:
            # add cross resonance tone
            pulse.play(
                pulse.GaussianSquare(
                    duration,
                    amp=opt.amp,
                    sigma=sigma,
                    risefall_sigma_ratio=opt.risefall,
                ),
                pulse.control_channels(*self.physical_qubits)[0],
            )
            # add cancellation tone
            if not np.isclose(opt.amp_t, 0.0):
                pulse.play(
                    pulse.GaussianSquare(
                        duration,
                        amp=opt.amp_t,
                        sigma=sigma,
                        risefall_sigma_ratio=opt.risefall,
                    ),
                    pulse.drive_channel(self.physical_qubits[1]),
                )
            else:
                pulse.delay(duration, pulse.drive_channel(self.physical_qubits[1]))

            # place holder for empty drive channels. this is necessary due to known pulse gate bug.
            pulse.delay(duration, pulse.drive_channel(self.physical_qubits[0]))

        cr_circuit.add_calibration(
            gate=cr_gate,
            qubits=self.physical_qubits,
            schedule=cross_resonance,
            params=[duration],
        )

        return cr_circuit

    def _net_duration(
        self,
        duration: float,
        sigma: float,
    ) -> float:
        """Calculate net cross resonance pulse duration considering the pulse edges.

        Args:
            duration: Parameter object representing a duration of cross resonance pulse.
            sigma: Sigma of Gaussian edges.

        Returns:
            Net duration in units of sec.
        """
        opt = self.experiment_options

        flat_top_width = duration - 2 * opt.risefall * sigma
        net_edge_width = np.sqrt(2 * np.pi * sigma ** 2)

        return 2 * (flat_top_width + net_edge_width)
