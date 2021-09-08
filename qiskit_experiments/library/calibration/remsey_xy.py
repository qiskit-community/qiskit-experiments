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

"""Ramsey XY frequency calibration experiment."""

from typing import List, Optional
import numpy as np

from qiskit import QuantumCircuit
from qiskit.circuit import Parameter, Gate
from qiskit.providers import Backend
from qiskit.utils import apply_prefix
import qiskit.pulse as pulse

from qiskit_experiments.framework import BaseExperiment
from qiskit_experiments.exceptions import CalibrationError
from qiskit_experiments.library.calibration.analysis.remsey_xy_analysis import RamseyXYAnalysis


class RamseyXY(BaseExperiment):
    r"""Ramsey XY experiment to measure the frequency of a qubit.

    # section: overview

        This experiment differs from the :class:`~qiskit_experiments.characterization.\
        t2ramsey.T2Ramsey` since it is sensitive to the sign of frequency offset from the main
        transition. This experiment consists of following two circuits:

        .. parsed-literal::

            (Ramsey X) The second pulse rotates by pi-half around the X axis

                       ┌────┐┌─────────────┐┌────┐ ░ ┌─┐
                  q_0: ┤ √X ├┤ Delay(τ[s]) ├┤ √X ├─░─┤M├
                       └────┘└─────────────┘└────┘ ░ └╥┘
            measure: 1/═══════════════════════════════╩═
                                                      0

            (Ramsey Y) The second pulse rotates by pi-half around the Y axis

                       ┌────┐┌─────────────┐┌──────────┐┌────┐ ░ ┌─┐
                  q_0: ┤ √X ├┤ Delay(τ[s]) ├┤ Rz(-π/2) ├┤ √X ├─░─┤M├
                       └────┘└─────────────┘└──────────┘└────┘ ░ └╥┘
            measure: 1/═══════════════════════════════════════════╩═
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

        Note: The experiment allows users to add a small frequency offset to better resolve
        any oscillations. This is implemented by embedding the circuits shown above in a
        single gate (:code:`RamseyX` or :code:`RamseyY`) since :code:`pulse.set_frequency`
        on IBM backends has unexpected behaviour. The circuits that are run are thus

        .. parsed-literal::

                       ┌───────────────┐ ░ ┌─┐
                  q_0: ┤ RamseyX(τ[s]) ├─░─┤M├
                       └───────────────┘ ░ └╥┘
            measure: 1/═════════════════════╩═
                                            0

                       ┌───────────────┐ ░ ┌─┐
                  q_0: ┤ RamseyY(τ[s]) ├─░─┤M├
                       └───────────────┘ ░ └╥┘
            measure: 1/═════════════════════╩═
                                            0

    """

    __analysis_class__ = RamseyXYAnalysis

    @classmethod
    def _default_experiment_options(cls):
        """Default values for the Ramsey XY experiment.

        Experiment Options:
            schedule (ScheduleBlock): The schedule for the sx gate.
            delays (list): The list of delays that will be scanned in the experiment.
            unit (str): The unit of the delays. Accepted values are dt, i.e. the
                duration of a single sample on the backend, seconds, and sub-units,
                e.g. ms, us, ns.
            frequency_offset (float): A frequency shift in Hz that will be applied to the
                schedule to increase the frequency of the measured oscillation.
        """
        options = super()._default_experiment_options()
        options.schedule = None
        options.delays = np.linspace(0, 1.0e-6, 51)
        options.unit = "s"
        options.frequency_offset = 2e6
        options.sample_multiple = 16

        return options

    def __init__(
        self,
        qubit: int,
        delays: Optional[List] = None,
        unit: Optional[str] = None,
        freq_offset: Optional[float] = None,
    ):
        """Create new experiment.
        
        Args:
            qubit: The qubit on which to run the Ramsey XY experiment.
            delays: The delays to scan.
            unit: The unit of the delays.
            freq_offset: A frequency offset from the qubit's frequency.
        """
        super().__init__([qubit])

        if delays is not None:
            self.experiment_options.delays = delays

        if unit is not None:
            self.experiment_options.unit = unit

        if freq_offset is not None:
            self.experiment_options.frequency_offset = freq_offset

    def _pre_circuit(self) -> QuantumCircuit:
        """Return a preparation circuit.

        This method can be overridden by subclasses e.g. to calibrate schedules on
        transitions other than the 0 <-> 1 transition.
        """
        return QuantumCircuit(1)

    def circuits(self, backend: Optional[Backend] = None) -> List[QuantumCircuit]:
        """Create the circuits for the Ramsey XY calibration experiment.

        Args:
            backend: A backend object.

        Returns:
            A list of circuits with a variable delay.
        """
        p_delay = Parameter("delay")
        unit = self._experiment_options.unit

        # Get the schedule and add a potential frequency shift.
        schedule = self.experiment_options.schedule

        if schedule is None:
            raise CalibrationError(f"The schedule was not specified for self.__class__.__name__.")

        d_freq = self.experiment_options.frequency_offset

        with pulse.build(backend=backend, name="RamseyXY_x") as sched_x:
            with pulse.frequency_offset(d_freq, pulse.drive_channel(self.physical_qubits[0])):
                pulse.call(schedule)
                pulse.delay(p_delay, pulse.drive_channel(self.physical_qubits[0]))
                pulse.call(schedule)

        with pulse.build(backend=backend, name="RamseyXY_y") as sched_y:
            with pulse.frequency_offset(d_freq, pulse.DriveChannel(self.physical_qubits[0])):
                pulse.call(schedule)
                pulse.delay(p_delay, pulse.drive_channel(self.physical_qubits[0]))
                pulse.shift_phase(-np.pi / 2, pulse.drive_channel(self.physical_qubits[0]))
                pulse.call(schedule)

        # Create the X and Y circuits.
        ram_x = self._pre_circuit()
        ram_x.append(Gate("RamseyX", num_qubits=1, params=[p_delay]), (0,))
        ram_x.measure_active()
        ram_x.add_calibration("RamseyX", self.physical_qubits, sched_x, params=[p_delay])

        ram_y = self._pre_circuit()
        ram_y.append(Gate("RamseyY", num_qubits=1, params=[p_delay]), (0,))
        ram_y.measure_active()
        ram_y.add_calibration("RamseyY", self.physical_qubits, sched_y, params=[p_delay])

        circs = []
        for delay in self.experiment_options.delays:

            # format delay to SI unit for analysis
            if unit == "dt":
                xval = delay * backend.configuration().dt
                samples = delay
            else:
                if unit == "s":
                    xval = delay
                else:
                    xval = apply_prefix(delay, unit)

                samples = int(xval / backend.configuration().dt)

            if self.experiment_options.sample_multiple:
                mult = self.experiment_options.sample_multiple
                samples = int(samples / mult) * mult

            metadata = {
                "experiment_type": self._type,
                "qubits": (self.physical_qubits[0],),
                "delay": delay,
                "unit": unit,
                "xval": xval,
            }

            # create ramsey x
            assigned_x = ram_x.assign_parameters({p_delay: samples}, inplace=False)
            assigned_x.metadata = metadata.copy()
            assigned_x.metadata["series"] = "X"

            # create ramsey y
            assigned_y = ram_y.assign_parameters({p_delay: samples}, inplace=False)
            assigned_y.metadata = metadata.copy()
            assigned_y.metadata["series"] = "Y"

            circs.extend([assigned_x, assigned_y])

        return circs
