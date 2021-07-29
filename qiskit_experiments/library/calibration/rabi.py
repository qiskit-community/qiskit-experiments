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

from qiskit import QuantumCircuit
from qiskit.circuit import Gate, Parameter
from qiskit.qobj.utils import MeasLevel
from qiskit.providers import Backend
import qiskit.pulse as pulse
from qiskit.providers.options import Options

from qiskit_experiments.framework import BaseExperiment
from qiskit_experiments.library.calibration.analysis.oscillation_analysis import OscillationAnalysis
from qiskit_experiments.exceptions import CalibrationError


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
    __rabi_gate_name__ = "Rabi"

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
        )

    @classmethod
    def _default_analysis_options(cls) -> Options:
        """Default analysis options."""
        options = super()._default_analysis_options()
        options.db_parameters = {"freq": ("Rabi rate", None)}
        options.normalization = True

        return options

    def __init__(self, qubit: int):
        """Initialize a Rabi experiment on the given qubit.

        The parameters of the Gaussian Rabi pulse can be specified at run-time.
        The rabi pulse has the following parameters:
        - duration: The duration of the rabi pulse in samples, the default is 160 samples.
        - sigma: The standard deviation of the pulse, the default is duration 40.
        - amplitudes: The amplitude that are scanned in the experiment, default  is
        np.linspace(-0.95, 0.95, 51)

        Args:
            qubit: The qubit on which to run the Rabi experiment.
        """
        super().__init__([qubit])

    def _template_circuit(self, amp_param) -> QuantumCircuit:
        """Return the template quantum circuit."""
        gate = Gate(name=self.__rabi_gate_name__, num_qubits=1, params=[amp_param])

        circuit = QuantumCircuit(1)
        circuit.append(gate, (0,))
        circuit.measure_active()

        return circuit

    def _default_gate_schedule(self, backend: Optional[Backend] = None):
        """Create the default schedule for the Rabi gate."""
        amp = Parameter("amp")
        with pulse.build(backend=backend, name="rabi") as default_schedule:
            pulse.play(
                pulse.Gaussian(
                    duration=self.experiment_options.duration,
                    amp=amp,
                    sigma=self.experiment_options.sigma,
                ),
                pulse.DriveChannel(self.physical_qubits[0]),
            )

        return default_schedule

    def circuits(self, backend: Optional[Backend] = None) -> List[QuantumCircuit]:
        """Create the circuits for the Rabi experiment.

        Args:
            backend: A backend object.

        Returns:
            A list of circuits with a rabi gate with an attached schedule. Each schedule
            will have a different value of the scanned amplitude.

        Raises:
            CalibrationError:
                - If the user-provided schedule does not contain a channel with an index
                  that matches the qubit on which to run the Rabi experiment.
                - If the user provided schedule has more than one free parameter.
        """
        schedule = self.experiment_options.get("schedule", None)

        if schedule is None:
            schedule = self._default_gate_schedule(backend=backend)
        else:
            if self.physical_qubits[0] not in set(ch.index for ch in schedule.channels):
                raise CalibrationError(
                    f"User provided schedule {schedule.name} does not contain a channel "
                    "for the qubit on which to run Rabi."
                )

        if len(schedule.parameters) != 1:
            raise CalibrationError("Schedule in Rabi must have exactly one free parameter.")

        param = next(iter(schedule.parameters))

        # Create template circuit
        circuit = self._template_circuit(param)
        circuit.add_calibration(
            self.__rabi_gate_name__, (self.physical_qubits[0],), schedule, params=[param]
        )

        # Create the circuits to run
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


class EFRabi(Rabi):
    """An experiment that scans the amplitude of a pulse to calibrate rotations between 1 and 2.

    This experiment is similar to the Rabi experiment between the ground and first excited state.
    The difference is in the initial X gate used to populate the first excited state. The Rabi pulse
    is then applied on the 1 <-> 2 transition (sometimes also labeled the e <-> f transition) which
    implies that frequency shift instructions are used. The necessary frequency shift (typically the
    qubit anharmonicity) should be specified through the experiment options.

    The circuits are of the form:

    .. parsed-literal::

                   ┌───┐┌───────────┐ ░ ┌─┐
              q_0: ┤ X ├┤ Rabi(amp) ├─░─┤M├
                   └───┘└───────────┘ ░ └╥┘
        measure: 1/══════════════════════╩═
                                         0
    """

    @classmethod
    def _default_experiment_options(cls) -> Options:
        """Default values for the pulse if no schedule is given.

        Users can set a schedule by doing

        .. code-block::

            ef_rabi.set_experiment_options(schedule=rabi_schedule)

        """
        return Options(
            duration=160,
            sigma=40,
            amplitudes=np.linspace(-0.95, 0.95, 51),
            schedule=None,
            normalization=True,
            frequency_shift=None,
        )

    @classmethod
    def _default_analysis_options(cls) -> Options:
        """Default analysis options."""
        options = super()._default_analysis_options()
        options.db_parameters = {"freq": ("EF Rabi rate", None)}

        return options

    def _default_gate_schedule(self, backend: Optional[Backend] = None):
        """Create the default schedule for the EFRabi gate with a frequency shift to the 1-2
        transition."""

        if self.experiment_options.frequency_shift is None:
            try:
                anharm, _ = backend.properties().qubit_property(self.physical_qubits[0])[
                    "anharmonicity"
                ]
                self.set_experiment_options(frequency_shift=anharm)
            except KeyError as key_err:
                raise CalibrationError(
                    f"The backend {backend} does not provide an anharmonicity for qubit "
                    f"{self.physical_qubits[0]}. Use EFRabi.set_experiment_options(frequency_shift="
                    f"anharmonicity) to manually set the correct frequency for the 1-2 transition."
                ) from key_err
            except AttributeError as att_err:
                raise CalibrationError(
                    "When creating the default schedule without passing a backend, the frequency needs "
                    "to be set manually through EFRabi.set_experiment_options(frequency_shift=..)."
                ) from att_err

        amp = Parameter("amp")
        with pulse.build(backend=backend, name=self.__rabi_gate_name__) as default_schedule:
            with pulse.frequency_offset(
                self.experiment_options.frequency_shift,
                pulse.DriveChannel(self.physical_qubits[0]),
            ):
                pulse.play(
                    pulse.Gaussian(
                        duration=self.experiment_options.duration,
                        amp=amp,
                        sigma=self.experiment_options.sigma,
                    ),
                    pulse.DriveChannel(self.physical_qubits[0]),
                )

        return default_schedule

    def _template_circuit(self, amp_param) -> QuantumCircuit:
        """Return the template quantum circuit."""
        circuit = QuantumCircuit(1)
        circuit.x(0)
        circuit.append(Gate(name=self.__rabi_gate_name__, num_qubits=1, params=[amp_param]), (0,))
        circuit.measure_active()

        return circuit
