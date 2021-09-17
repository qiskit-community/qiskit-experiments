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

"""Fine DRAG calibration experiment."""

from typing import Optional, List
import numpy as np

from qiskit import QuantumCircuit
from qiskit.circuit import Gate
from qiskit.qobj.utils import MeasLevel
from qiskit.providers import Backend
from qiskit.pulse import ScheduleBlock
import qiskit.pulse as pulse

from qiskit_experiments.framework import BaseExperiment, Options
from qiskit_experiments.library.calibration.analysis.fine_amplitude_analysis import (
    FineAmplitudeAnalysis,
)


class FineDrag(BaseExperiment):
    r"""Fine DRAG Calibration experiment.

    # section: overview

        The class :class:`FineDrag` runs fine DRAG calibration experiments (see :class:`DragCal`
        for the definition of DRAG pulses). Fine DRAG calibration proceeds by iterating the
        gate sequence Rp - Rm where Rp is a rotation around an axis and Rm is the same rotation
        but in the opposite direction. The circuits that are executed are of the form

        .. parsed-literal::

                    ┌─────┐┌────┐┌────┐     ┌────┐┌────┐┌──────┐ ░ ┌─┐
               q_0: ┤ Pre ├┤ Rp ├┤ Rm ├ ... ┤ Rp ├┤ Rm ├┤ Post ├─░─┤M├
                    └─────┘└────┘└────┘     └────┘└────┘└──────┘ ░ └╥┘
            meas: 1/═══════════════════ ... ════════════════════════╩═
                                                                    0

        Here, Pre and Post designate gates that may be pre-appended and and post-appended,
        respectively, to the repeated sequence of Rp and Rm gates. When calibrating a pulse
        with a target rotation angle of π the Pre and Post gates are Id and RYGate(π/2),
        respectively. When calibrating a pulse with a target rotation angle of π/2 the Pre and
        Post gates are RXGate(π/2) and RYGate(π/2), respectively.

        We now describe what this experiment corrects by following Ref. [1]. We follow equations
        4.30 and onwards of Ref. [1] which state that the first-order corrections to the control
        fields are

        .. math::

            \begin{align}
                \bar{\Omega}_x^{(1)}(t) = &\, 2\dot{s}^{(1)}_{x,0,1}(t) \\
                \bar{\Omega}_y^{(1)}(t) = &\, 2\dot{s}^{(1)}_{y,0,1}(t)
                - s_{z,1}^{(1)}(t)t_g\Omega_x(t) \\
                \bar{\delta}^{(1)}(t) = &\, \dot{s}_{z,1}^{(1)}(t) + 2s^{(1)}_{y,0,1}(t)t_g\Omega_x(t)
                 + \frac{\lambda_1^2 t_g^2 \Omega_x^2(t)}{4}
            \end{align}


        Here, the :math:`s` terms are coefficients of the expansion of an operator :math:`S(t)`
        that generates a transformation that keeps the qubit sub-space isolated from the
        higher-order states. :math:`t_g` is the gate time, :math:`\Omega_x(t)` is the pulse envelope
        on the in-phase component of the drive and :math:`\lambda_1` is a parmeter of the Hamiltonian.
        For additional details please see Ref. [1].
        As in Ref. [1] we now set :math:`s^{(1)}_{x,0,1}` and :math:`s^{(1)}_{z,1}` to zero
        and set :math:`s^{(1)}_{y,0,1}` to :math:`-\lambda_1^2 t_g\Omega_x(t)/8`. This
        results in a Z angle rotation rate of :math:`\bar{\delta}^{(1)}(t)=0` in the equations
        above and defines the value for the ideal :math:`\beta` parameter.
        With these choices, the Y quadrature of the first-order DRAG pulse is

        .. math::

            \Omega_y(t)=-\frac{\lambda_1^2\dot{\Omega}_x(t)}{4\Delta}

        In Qiskit pulse, the definition of the DRAG pulse is

        .. math::

            \Omega(t) = \Omega_x(t) + i\beta\,\dot{\Omega}_x(t)\quad\Longrightarrow\quad
            \Omega_y(t)= \beta\,\dot{\Omega}_x(t)

        From which we identify the ideal value of :math:`\beta` as :math:`-\lambda^2_1/(4\Delta)`.
        We now assume that there is a small error :math:`{\rm d}\beta` in :math:`\beta` such
        that the instantaneous Z-angle error is

        .. math::

            \bar\delta(t) = 2\,{\rm d}\beta\, \Omega^2_x(t)


        We can integrate :math:`\bar{\delta}(t)`, i.e. the instantaneous Z-angle rotation error,
        to obtain the total rotation angle error per pulse :math:`{\rm d}\theta`.

        .. math::

            \int\bar\delta(t){\rm d}t = 2{\rm d}\beta \int\Omega^2_x(t){\rm d}t

        If we assume a Gaussian pulse, i.e. :math:`\Omega_x(t)=A\exp[-t^2/(2\sigma^2)]`
        then the integral of :math:`\Omega_x^2(t)` in the equation above results in
        :math:`A^2\sigma\sqrt{\pi}`. Furthermore, the integral of :math:`\Omega_x(t)` is
        :math:`A\sigma\sqrt{\pi/2}=\theta_\text{target}`, where :math:`\theta_\text{target}`
        is the target rotation angle, i.e. the area under the pulse. This last point allows
        us to rewrite :math:`A^2\sigma\sqrt{\pi}` as
        :math:`\theta^2_\text{target}/(2\sigma\sqrt{\pi})`. The total Z angle error per pulse
        is therefore

        .. math::

            \int\bar\delta(t){\rm d}t=2\,{\rm d}\beta\,\frac{\theta^2_\text{target}}{2\sigma\sqrt{\pi}}
            ={\rm d}\theta

        Here, :math:`{\rm d}\theta` is the Z angle error which the gate sequence shown above
        can measure. Inverting the relation above yields the error in :math:`\beta` that
        produced the rotation error :math:`{\rm d}\theta`.

        .. math::

            {\rm d}\beta=\frac{\sqrt{\pi}\,{\rm d}\theta\sigma}{ \theta_\text{target}^2}

    # section: reference
        .. ref_arxiv:: 1 1011.1949
    """

    __analysis_class__ = FineAmplitudeAnalysis

    @classmethod
    def _default_run_options(cls) -> Options:
        """Default option values for the experiment :meth:`run` method."""
        options = super()._default_run_options()
        options.meas_level = MeasLevel.CLASSIFIED
        options.meas_return = "avg"

        return options

    @classmethod
    def _default_experiment_options(cls) -> Options:
        r"""Default values for the fine amplitude experiment.

        Experiment Options:
            repetitions (List[int]): A list of the number of times that Rp - Rm gate sequence
                is repeated.
            schedule (ScheduleBlock): The schedule for the plus rotation.
            normalization (bool): If set to True the DataProcessor will normalized the
                measured signal to the interval [0, 1]. Defaults to True.
            sx_schedule (ScheduleBlock): The schedule to attache to the SX gate.
        """
        options = super()._default_experiment_options()
        options.repetitions = list(range(20))
        options.schedule = None
        options.normalization = True
        options.sx_schedule = None

        return options

    @classmethod
    def _default_analysis_options(cls) -> Options:
        """Default analysis options."""
        options = super()._default_analysis_options()
        options.angle_per_gate = 0.0
        options.phase_offset = np.pi / 2

        return options

    def __init__(self, qubit: int):
        """Setup a fine amplitude experiment on the given qubit.

        Args:
            qubit: The qubit on which to run the fine amplitude calibration experiment.
        """
        super().__init__([qubit])

    @staticmethod
    def _pre_circuit() -> QuantumCircuit:
        """Return the quantum circuit to apply before repeating the Rp and Rm gates."""
        return QuantumCircuit(1)

    @staticmethod
    def _post_circuit() -> QuantumCircuit:
        """Return the quantum circuit to apply after repeating the Rp and Rm gates."""
        circ = QuantumCircuit(1)
        circ.ry(np.pi / 2, 0)
        return circ

    # TODO Remove once #251 gets merged.
    def _set_anti_schedule(self, schedule) -> ScheduleBlock:
        """A DRAG specific method that sets the rm schedule based on rp.
        The rm schedule, i.e. the anti-schedule, is the rp schedule sandwiched
        between two virtual phase gates with angle pi.
        """
        with pulse.build(name="rm") as minus_sched:
            pulse.shift_phase(np.pi, pulse.DriveChannel(self._physical_qubits[0]))
            pulse.call(schedule)
            pulse.shift_phase(-np.pi, pulse.DriveChannel(self._physical_qubits[0]))

        return minus_sched

    def circuits(self, backend: Optional[Backend] = None) -> List[QuantumCircuit]:
        """Create the circuits for the fine DRAG calibration experiment.

        Args:
            backend: A backend object.

        Returns:
            A list of circuits with a variable number of gates. Each gate has the same
            pulse schedule.
        """

        schedule, qubits = self.experiment_options.schedule, self._physical_qubits

        # Prepare the circuits
        repetitions = self.experiment_options.get("repetitions")

        circuits = []

        for repetition in repetitions:
            circuit = self._pre_circuit()

            for _ in range(repetition):
                circuit.append(Gate(name="Rp", num_qubits=1, params=[]), (0,))
                circuit.append(Gate(name="Rm", num_qubits=1, params=[]), (0,))

            circuit.compose(self._post_circuit(), inplace=True)

            circuit.measure_all()
            circuit.add_calibration("Rp", qubits, schedule, params=[])
            circuit.add_calibration("Rm", qubits, self._set_anti_schedule(schedule), params=[])

            # TODO update with metadata function after #251
            circuit.metadata = {
                "experiment_type": self._type,
                "qubits": (self.physical_qubits[0],),
                "xval": repetition,
                "unit": "gate number",
            }

            circuits.append(circuit)

        return circuits
