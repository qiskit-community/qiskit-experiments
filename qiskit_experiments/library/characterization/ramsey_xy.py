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

from typing import List, Optional
import numpy as np

from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.providers.backend import Backend

from qiskit_experiments.framework import BaseExperiment
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

    def _metadata(self):
        metadata = super()._metadata()
        # Store measurement level and meas return if they have been
        # set for the experiment
        for run_opt in ["meas_level", "meas_return"]:
            if hasattr(self.run_options, run_opt):
                metadata[run_opt] = getattr(self.run_options, run_opt)
        return metadata
