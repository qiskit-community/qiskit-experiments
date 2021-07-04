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

"""Rough drag pulse calibration experiment."""

from typing import List, Optional
import numpy as np

from qiskit import QuantumCircuit
from qiskit.circuit import Gate, Parameter
from qiskit.qobj.utils import MeasLevel
from qiskit.providers import Backend
import qiskit.pulse as pulse
from qiskit.providers.options import Options

from qiskit_experiments.base_experiment import BaseExperiment
from qiskit_experiments.data_processing.processor_library import get_to_signal_processor
from qiskit_experiments.calibration.exceptions import CalibrationError
from qiskit_experiments.calibration.experiments.drag_analysis import DragCalAnalysis


class DragCal(BaseExperiment):
    r"""An experiment that scans the DRAG parameter to find the optimal value.

    A Derivative Removal by Adiabatic Gate (DRAG) pulse is designed to minimize leakage
    to a neighbouring transition. It is a standard pulse with an additional derivative
    component. It is designed to reduce the frequency spectrum of a normal pulse near
    the :math:`|1\rangle` - :math:`|2\rangle` transition, reducing the chance of leakage
    to the :math:`|2\rangle` state. The optimal value of the DRAG parameter is chosen to
    minimize both leakage and phase errors resulting from the AC Stark shift.

    .. math::

        f(t) = \Omega(t) + 1j \beta d/dt \Omega(t)

    Here, :math:`\Omega` is the envelop of the in-phase component of the pulse and
    :math:`\beta` is the strength of the quadrature which we refer to as the DRAG
    parameter and seek to calibrate in this experiment. The DRAG calibration will run
    several series of circuits. In a given circuit a Rp(β) - Rm(β) block is repeated
    :math:`N` times. Here, Rp is a rotation with a positive angle and Rm is the same rotation
    with a native angle. As example the circuit of a single repetition, i.e. :math:`N=1`, is
    shown below.

    .. parsed-literal::

                   ┌───────┐ ┌───────┐ ░ ┌─┐
              q_0: ┤ Rp(β) ├─┤ Rm(β) ├─░─┤M├
                   └───────┘ └───────┘ ░ └╥┘
        measure: 1/═══════════════════════╩═
                                          0

    Here, the Rp gate and the Rm gate are can be pi and -pi rotations about the
    x-axis of the Bloch sphere. The parameter β is scanned to find the value that minimizes
    the leakage to the second excited state. Note that the analysis class requires this
    experiment to run with three repetition numbers.

    References:
        1. |citation1|_

        .. _citation1: https://link.aps.org/doi/10.1103/PhysRevA.83.012308

        .. |citation1| replace:: *Gambetta, J. M., Motzoi, F., Merkel, S. T. & Wilhelm, F. K.
           Analytic control methods for high-fidelity unitary operations
           in a weakly nonlinear oscillator. Phys. Rev. A 83, 012308 (2011).*

        2. |citation2|_

        .. _citation2: https://link.aps.org/doi/10.1103/PhysRevLett.103.110501

        .. |citation2| replace:: *F. Motzoi, J. M. Gambetta, P. Rebentrost, and F. K. Wilhelm
           Phys. Rev. Lett. 103, 110501 – Published 8 September 2009.*

        3. |citation3|_

        .. _citation3: https://link.aps.org/doi/10.1103/PhysRevLett.116.020501

        .. |citation3| replace:: *Z. Chen, et al.
           Measuring and Suppressing Quantum State Leakage in a Superconducting Qubit
           Phys. Rev. Lett. 116, 020501 – Published 13 January 2016.*
    """

    __analysis_class__ = DragCalAnalysis

    @classmethod
    def _default_run_options(cls) -> Options:
        """Default option values for the experiment :meth:`run` method."""
        return Options(
            meas_level=MeasLevel.CLASSIFIED,
            meas_return="avg",
        )

    @classmethod
    def _default_experiment_options(cls) -> Options:
        """Default values for the pulse if no schedule is given.
        Users can set the positive and negative rotation schedules with

        .. code-block::

            drag.set_experiment_options(rp=xp_schedule, rm=xm_schedule)
        """
        options = super()._default_experiment_options()

        options.rp = None
        options.rm = None
        options.amp = 0.2
        options.duration = 160
        options.sigma = 40
        options.reps = [1, 3, 5]
        options.betas = np.linspace(-5, 5, 51)

        return options

    # pylint: disable=arguments-differ
    def set_experiment_options(self, reps: Optional[List] = None, **fields):
        """Raise if reps has a length different from three.

        Raises:
            CalibrationError: if the number of repetitions is different from three.
        """

        if reps is None:
            reps = [1, 3, 5]
        else:
            reps = sorted(reps)  # ensure reps 1 is the lowest frequency.

        if len(reps) != 3:
            raise CalibrationError(
                f"{self.__class__.__name__} must use exactly three repetition numbers. "
                f"Received {reps} with length {len(reps)} != 3."
            )

        super().set_experiment_options(reps=reps, **fields)

    def __init__(self, qubit: int):
        """
        Args:
            qubit: The qubit for which to run the Drag calibration.
        """

        super().__init__([qubit])

    def circuits(self, backend: Optional[Backend] = None) -> List[QuantumCircuit]:
        """Create the circuits for the Drag calibration.

        Args:
            backend: A backend object.

        Returns:
            circuits: The circuits that will run the Drag calibration.

        Raises:
            CalibrationError:
                - If the beta parameters in the xp and xm pulses are not the same.
                - If either the xp or xm pulse do not have at least one Drag pulse.
                - If the number of different repetition series is not three.
        """
        # TODO this is temporary logic.
        self.set_analysis_options(
            data_processor=get_to_signal_processor(
                meas_level=self.run_options.meas_level,
                meas_return=self.run_options.meas_return,
            ),
        )

        plus_sched = self.experiment_options.rp
        minus_sched = self.experiment_options.rm

        if plus_sched is None:
            beta = Parameter("β")
            with pulse.build(backend=backend, name="xp") as plus_sched:
                pulse.play(
                    pulse.Drag(
                        duration=self.experiment_options.duration,
                        amp=self.experiment_options.amp,
                        sigma=self.experiment_options.sigma,
                        beta=beta,
                    ),
                    pulse.DriveChannel(self._physical_qubits[0]),
                )

            with pulse.build(backend=backend, name="xm") as minus_sched:
                pulse.play(
                    pulse.Drag(
                        duration=self.experiment_options.duration,
                        amp=-self.experiment_options.amp,
                        sigma=self.experiment_options.sigma,
                        beta=beta,
                    ),
                    pulse.DriveChannel(self._physical_qubits[0]),
                )

        if len(plus_sched.parameters) != 1 or len(minus_sched.parameters) != 1:
            raise CalibrationError(
                "The schedules for Drag calibration must both have one free parameter."
                f"Found {len(plus_sched.parameters)} and {len(minus_sched.parameters)} "
                "for Rp and Rm, respectively."
            )

        beta_xp = next(iter(plus_sched.parameters))
        beta_xm = next(iter(minus_sched.parameters))

        if beta_xp != beta_xm:
            raise CalibrationError(
                f"Beta for xp and xm in {self.__class__.__name__} calibration are not identical."
            )

        xp_gate = Gate(name="Rp", num_qubits=1, params=[beta_xp])
        xm_gate = Gate(name="Rm", num_qubits=1, params=[beta_xp])

        reps = self.experiment_options.reps
        if len(reps) != 3:
            raise CalibrationError(
                f"{self.__class__.__name__} must use exactly three repetition numbers. "
                f"Received {reps} with length {len(reps)} != 3."
            )

        circuits = []
        for beta in self.experiment_options.betas:

            beta = np.round(beta, decimals=6)

            for idx, rep in enumerate(reps):
                circuit = QuantumCircuit(1)
                for _ in range(rep):
                    circuit.append(xp_gate, (0,))
                    circuit.append(xm_gate, (0,))

                circuit.measure_active()
                circuit.assign_parameters({beta_xp: beta}, inplace=True)

                xm_ = minus_sched.assign_parameters({beta_xp: beta}, inplace=False)
                xp_ = plus_sched.assign_parameters({beta_xp: beta}, inplace=False)

                circuit.add_calibration("Rp", (self.physical_qubits[0],), xp_, params=[beta])
                circuit.add_calibration("Rm", (self.physical_qubits[0],), xm_, params=[beta])

                circuit.metadata = {
                    "experiment_type": self._type,
                    "qubits": (self.physical_qubits[0],),
                    "xval": beta,
                    "series": idx,
                }

                circuits.append(circuit)

        return circuits
