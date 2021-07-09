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

"""Rabi amplitude experiment."""

from typing import List, Optional
import numpy as np

from qiskit import QiskitError, QuantumCircuit
from qiskit.circuit import Gate, Parameter
from qiskit.qobj.utils import MeasLevel
from qiskit.providers import Backend
import qiskit.pulse as pulse
from qiskit.providers.options import Options

from qiskit_experiments.base_experiment import BaseExperiment
from qiskit_experiments.data_processing.processor_library import get_to_signal_processor
from qiskit_experiments.calibration.analysis.oscillation_analysis import OscillationAnalysis


class Rabi(BaseExperiment):
    """An experiment that scans the amplitude of a pulse to calibrate rotations between 0 and 1.

    The circuits that are run have a custom rabi gate with the pulse schedule attached to it
    through the calibrations. The circuits are of the form:

    .. parsed-literal::

                   ┌───────────┐ ░ ┌─┐
              q_0: ┤ Rabi(amp) ├─░─┤M├
                   └───────────┘ ░ └╥┘
        measure: 1/═════════════════╩═
                                    0

    If the user provides his own schedule for the Rabi then it must have one free parameter, i.e.
    the amplitude that will be scanned, and a drive channel which matches the qubit.
    """

    __analysis_class__ = OscillationAnalysis

    @classmethod
    def _default_run_options(cls) -> Options:
        """Default option values for the experiment :meth:`run` method."""
        return Options(
            meas_level=MeasLevel.KERNELED,
            meas_return="single",
        )

    @classmethod
    def _default_experiment_options(cls) -> Options:
        """Default values for the pulse if no schedule is given.

        Users can set a schedule by doing

        .. code-block::

            rabi.set_experiment_options(schedule=rabi_schedule)

        """
        return Options(
            duration=160,
            sigma=40,
            amplitudes=np.linspace(-0.95, 0.95, 51),
            schedule=None,
            normalization=True,
        )

    def __init__(self, qubit: int):
        """Initialize a Rabi experiment on the given qubit.

        Args:
            qubit: The qubit on which to run the Rabi experiment.
        """
        super().__init__([qubit])

    def circuits(self, backend: Optional[Backend] = None) -> List[QuantumCircuit]:
        """Create the circuits for the Rabi experiment.

        Args:
            backend: A backend object.

        Returns:
            A list of circuits with a rabi gate with an attached schedule. Each schedule
            will have a different value of the scanned amplitude.

        Raises:
            QiskitError:
                - If the user-provided schedule does not contain a channel with an index
                  that matches the qubit on which to run the Rabi experiment.
                - If the user provided schedule has more than one free parameter.
        """
        # TODO this is temporary logic. Need update of circuit data and processor logic.
        self.set_analysis_options(
            data_processor=get_to_signal_processor(
                meas_level=self.run_options.meas_level,
                meas_return=self.run_options.meas_return,
                normalize=self.experiment_options.normalization,
            )
        )

        schedule = self.experiment_options.get("schedule", None)

        if schedule is None:
            amp = Parameter("amp")
            with pulse.build() as default_schedule:
                pulse.play(
                    pulse.Gaussian(
                        duration=self.experiment_options.duration,
                        amp=amp,
                        sigma=self.experiment_options.sigma,
                    ),
                    pulse.DriveChannel(self.physical_qubits[0]),
                )

            schedule = default_schedule
        else:
            if self.physical_qubits[0] not in set(ch.index for ch in schedule.channels):
                raise QiskitError(
                    f"User provided schedule {schedule.name} does not contain a channel "
                    "for the qubit on which to run Rabi."
                )

        if len(schedule.parameters) != 1:
            raise QiskitError("Schedule in Rabi must have exactly one free parameter.")

        param = next(iter(schedule.parameters))

        gate = Gate(name="Rabi", num_qubits=1, params=[param])

        circuit = QuantumCircuit(1)
        circuit.append(gate, (0,))
        circuit.measure_active()
        circuit.add_calibration(gate, (self.physical_qubits[0],), schedule, params=[param])

        circs = []
        for amp in self.experiment_options.amplitudes:
            amp = np.round(amp, decimals=6)
            assigned_circ = circuit.assign_parameters({param: amp}, inplace=False)
            assigned_circ.metadata = {
                "experiment_type": self._type,
                "qubits": (self.physical_qubits[0],),
                "xval": amp,
                "unit": "arb. unit",
                "amplitude": amp,
                "schedule": str(schedule),
            }

            if backend:
                assigned_circ.metadata["dt"] = getattr(backend.configuration(), "dt", "n.a.")

            circs.append(assigned_circ)

        return circs
