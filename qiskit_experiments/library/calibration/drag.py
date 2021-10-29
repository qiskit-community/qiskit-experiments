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
from qiskit.providers.backend import Backend
import qiskit.pulse as pulse

from qiskit_experiments.framework import BaseExperiment, Options, fix_class_docs
from qiskit_experiments.exceptions import CalibrationError
from qiskit_experiments.library.calibration.analysis.drag_analysis import DragCalAnalysis


@fix_class_docs
class DragCal(BaseExperiment):
    r"""An experiment that scans the DRAG parameter to find the optimal value.

    # section: overview

        A Derivative Removal by Adiabatic Gate (DRAG) pulse is designed to minimize phase
        errors and leakage resulting from the presence of a neighbouring transition. DRAG
        is a standard pulse with an additional derivative component. The optimal value of
        the DRAG parameter, :math:`\beta`, is chosen to primarily minimize phase errors
        resulting from the AC Stark shift and potentially leakage errors. The DRAG pulse is

        .. math::

            f(t) = \Omega(t) + 1j \beta d/dt \Omega(t)

        Here, :math:`\Omega` is the envelop of the in-phase component of the pulse and
        :math:`\beta` is the strength of the quadrature which we refer to as the DRAG
        parameter and seek to calibrate in this experiment. The DRAG calibration will run
        several series of circuits. In a given circuit a Rp(β) - Rm(β) block is repeated
        :math:`N` times. Here, Rp is a rotation with a positive angle and Rm is the same rotation
        with a native angle and is implemented by the gate sequence Rz(π) - Rp(β) - Rz(π) where
        the Z rotations are virtual. As example the circuit of a single repetition, i.e.
        :math:`N=1`, is shown below.

        .. parsed-literal::

                       ┌───────┐┌───────┐┌───────┐┌───────┐ ░ ┌─┐
                  q_0: ┤ Rp(β) ├┤ Rz(π) ├┤ Rp(β) ├┤ Rz(π) ├─░─┤M├
                       └───────┘└───────┘└───────┘└───────┘ ░ └╥┘
            measure: 1/════════════════════════════════════════╩═
                                                               0

        The parameter β is scanned to find the value that minimizes the unwanted Z-rotation.
        Note that the analysis class requires this experiment to run with three repetition numbers.

    # section: reference
        .. ref_arxiv:: 1 1011.1949
        .. ref_arxiv:: 2 0901.0534
        .. ref_arxiv:: 3 1509.05470

    # section: tutorial
        :doc:`/tutorials/calibrating_armonk`

    """

    __analysis_class__ = DragCalAnalysis

    @classmethod
    def _default_experiment_options(cls) -> Options:
        r"""Default values for the pulse if no schedule is given.
        Users can set the positive and negative rotation schedules with

        .. code-block::

            drag.set_experiment_options(schedule=xp_schedule)

        Experiment Options:
            schedule (ScheduleBlock): The schedule of the rotation.
            amp (complex): The amplitude for the default Drag pulse. Must have a magnitude
                smaller than one.
            duration (int): The duration of the default pulse in samples.
            sigma (float): The standard deviation of the default pulse.
            reps (List[int]): The number of times the Rp - Rm gate sequence is repeated in
                each series. Note that this list must always have a length of three as
                otherwise the analysis class will not run.
            betas (Iterable): the values of the DRAG parameter to scan.
        """
        options = super()._default_experiment_options()

        options.schedule = None
        options.amp = 0.2
        options.duration = 160
        options.sigma = 40
        options.reps = [1, 3, 5]
        options.betas = np.linspace(-5, 5, 51)

        return options

    @classmethod
    def _default_analysis_options(cls) -> Options:
        """Default analysis options."""
        options = super()._default_analysis_options()
        options.normalization = True

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

    def __init__(self, qubit: int, backend: Optional[Backend] = None):
        """
        Args:
            qubit: The qubit for which to run the Drag calibration.
            backend: Optional, the backend to run the experiment on.
        """

        super().__init__([qubit], backend=backend)

    def circuits(self) -> List[QuantumCircuit]:
        """Create the circuits for the Drag calibration.

        Returns:
            circuits: The circuits that will run the Drag calibration.

        Raises:
            CalibrationError:
                - If the beta parameters in the xp and xm pulses are not the same.
                - If either the xp or xm pulse do not have at least one Drag pulse.
                - If the number of different repetition series is not three.
        """
        schedule = self.experiment_options.schedule

        if schedule is None:
            beta = Parameter("β")
            with pulse.build(backend=self.backend, name="drag") as schedule:
                pulse.play(
                    pulse.Drag(
                        duration=self.experiment_options.duration,
                        amp=self.experiment_options.amp,
                        sigma=self.experiment_options.sigma,
                        beta=beta,
                    ),
                    pulse.DriveChannel(self.physical_qubits[0]),
                )

        if len(schedule.parameters) != 1:
            raise CalibrationError(
                "The schedule for Drag calibration must have one free parameter."
                f"Found {len(schedule.parameters)}."
            )

        beta = next(iter(schedule.parameters))

        drag_gate = Gate(name=schedule.name, num_qubits=1, params=[beta])

        reps = self.experiment_options.reps
        if len(reps) != 3:
            raise CalibrationError(
                f"{self.__class__.__name__} must use exactly three repetition numbers. "
                f"Received {reps} with length {len(reps)} != 3."
            )

        circuits = []

        for idx, rep in enumerate(reps):
            circuit = QuantumCircuit(1)
            for _ in range(rep):
                circuit.append(drag_gate, (0,))
                circuit.rz(np.pi, 0)
                circuit.append(drag_gate, (0,))
                circuit.rz(np.pi, 0)

            circuit.measure_active()

            circuit.add_calibration(schedule.name, self.physical_qubits, schedule, params=[beta])

            for beta_val in self.experiment_options.betas:
                beta_val = np.round(beta_val, decimals=6)

                assigned_circuit = circuit.assign_parameters({beta: beta_val}, inplace=False)

                assigned_circuit.metadata = {
                    "experiment_type": self._type,
                    "qubits": self.physical_qubits,
                    "xval": beta_val,
                    "series": idx,
                }

                circuits.append(assigned_circuit)

        return circuits
