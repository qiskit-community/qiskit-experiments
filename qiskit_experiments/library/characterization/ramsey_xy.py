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

from qiskit_experiments.framework import BaseExperiment, fix_class_docs
from qiskit_experiments.library.calibration.analysis.remsey_xy_analysis import RamseyXYAnalysis


@fix_class_docs
class RamseyXY(BaseExperiment):
    r"""Ramsey XY experiment to measure the frequency of a qubit.

    # section: overview

        This experiment differs from the :class:`~qiskit_experiments.characterization.\
        t2ramsey.T2Ramsey` since it is sensitive to the sign of frequency offset from the main
        transition. This experiment consists of following two circuits:

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

        The first and second circuits measure the expectation value along the X and Y axis,
        respectively. This experiment therefore draws the dynamics of the Bloch vector as a
        Lissajous figure. Since the control electronics tracks the frame of qubit at the
        reference frequency, which differs from the true qubit frequency by :math:`\Delta\omega`,
        we can describe the dynamics of two circuits as follows. The Hamiltonian during the
        ``Delay`` instruction is :math:`H^R = - \frac{1}{2} \Delta\omega` in the rotating frame,
        and the propagator will be :math:`U(\tau) = \exp(-iH^R\tau)` where :math:`\tau` is the
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
        i.e. :math:`\cos(-\Delta\omega\tau) = \cos(\Delta\omega\tau)`.

        The experiment also allows users to add a small frequency offset to better resolve
        any oscillations. This is implemented by a virtual Z rotation in the circuits. In the
        circuit above it appears as the delay-dependent angle θ(τ).
    """

    __analysis_class__ = RamseyXYAnalysis

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
        super().__init__([qubit], backend=backend)

        delays = delays or self.experiment_options.delays
        self.set_experiment_options(delays=delays, osc_freq=osc_freq)

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
        # Compute the rz rotation angle to add a modulation.
        p_delay = Parameter("delay")

        rotation_angle = 2 * np.pi * self.experiment_options.osc_freq * p_delay

        # Create the X and Y circuits.
        metadata = {
            "experiment_type": self._type,
            "qubits": self.physical_qubits,
            "osc_freq": self.experiment_options.osc_freq,
            "unit": "s",
        }

        ram_x = self._pre_circuit()
        ram_x.sx(0)
        ram_x.delay(p_delay, 0, "s")
        ram_x.rz(rotation_angle, 0)
        ram_x.sx(0)
        ram_x.measure_active()
        ram_x.metadata = metadata.copy()

        ram_y = self._pre_circuit()
        ram_y.sx(0)
        ram_y.delay(p_delay, 0, "s")
        ram_y.rz(rotation_angle - np.pi / 2, 0)
        ram_y.sx(0)
        ram_y.measure_active()
        ram_y.metadata = metadata.copy()

        circs = []
        for delay in self.experiment_options.delays:

            # create ramsey x
            assigned_x = ram_x.assign_parameters({p_delay: delay}, inplace=False)
            assigned_x.metadata["series"] = "X"
            assigned_x.metadata["xval"] = delay

            # create ramsey y
            assigned_y = ram_y.assign_parameters({p_delay: delay}, inplace=False)
            assigned_y.metadata["series"] = "Y"
            assigned_y.metadata["xval"] = delay

            circs.extend([assigned_x, assigned_y])

        return circs
