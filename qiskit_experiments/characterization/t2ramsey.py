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
T2Ramsey Experiment class.
"""

from typing import List, Optional, Union, Iterable
import numpy as np

import qiskit
from qiskit.providers import Backend, Options
from qiskit.circuit import QuantumCircuit, Parameter
from qiskit_experiments.base_experiment import BaseExperiment
from .t2ramsey_analysis import T2RamseyAnalysis, RamseyXYAnalysis
from qiskit.utils import apply_prefix
from qiskit.exceptions import QiskitError


class T2Ramsey(BaseExperiment):
    """T2Ramsey class"""

    __analysis_class__ = T2RamseyAnalysis

    def __init__(
        self,
        qubit: int,
        delays: Union[List[float], np.array],
        unit: str = "s",
        osc_freq: float = 0.0,
        experiment_type: Optional[str] = None,
    ):
        """Initialize the T2Ramsey class.

        Args:
            qubit: the qubit under test
            delays: delay times of the experiments
            unit: Optional, time unit of `delays`.
            Supported units: 's', 'ms', 'us', 'ns', 'ps', 'dt'.
            The unit is used for both T2Ramsey and the frequency
            osc_freq: the oscillation frequency induced using by the user
            experiment_type: String indicating the experiment type.
        """

        self._qubit = qubit
        self._delays = delays
        self._unit = unit
        self._osc_freq = osc_freq
        super().__init__([qubit], experiment_type)

    def circuits(self, backend: Optional[Backend] = None) -> List[QuantumCircuit]:
        """Return a list of experiment circuits.

        Each circuit consists of a Hadamard gate, followed by a fixed delay,
        a phase gate (with a linear phase), and an additional Hadamard gate.

        Args:
            backend: Optional, a backend object

        Returns:
            The experiment circuits

        Raises:
            AttributeError: if unit is dt but dt parameter is missing in the backend configuration
        """
        if self._unit == "dt":
            try:
                dt_factor = getattr(backend._configuration, "dt")
            except AttributeError as no_dt:
                raise AttributeError("Dt parameter is missing in backend configuration") from no_dt

        circuits = []
        for delay in self._delays:
            circ = qiskit.QuantumCircuit(1, 1)
            circ.h(0)
            circ.delay(delay, 0, self._unit)
            circ.p(2 * np.pi * self._osc_freq, 0)
            circ.barrier(0)
            circ.h(0)
            circ.barrier(0)
            circ.measure(0, 0)

            circ.metadata = {
                "experiment_type": self._type,
                "qubit": self._qubit,
                "osc_freq": self._osc_freq,
                "xval": delay,
                "unit": self._unit,
            }
            if self._unit == "dt":
                circ.metadata["dt_factor"] = dt_factor

            circuits.append(circ)

        return circuits


class RamseyXY(BaseExperiment):
    """Ramsey experiment for frequency calibration.

    This experiment differs from the standard Ramsey experiment by the sensitivity to the
    sign of frequency error. This experiment consists of following two circuits.

    TODO more documentation
    """

    __analysis_class__ = RamseyXYAnalysis

    def __init__(
            self,
            qubit: int,
            delays: Union[Iterable[float], np.ndarray],
            unit: str = "s",
            offset_frequency: float = 0.,
    ):
        """Initialize new experiment.

        Args:
            qubit: The qubit under test.
            delays: Delay times of the experiments.
            unit: Time unit of `delays`. One of 's', 'ms', 'us', 'ns', 'ps', 'dt'.
            offset_frequency: Frequency offset for this experiment.

        .. warning::

            ``offset_frequency`` doesn't really change the qubit frequency for now,
            because of the missing oscillator frequency upgrade pass in the circuit execution.
        """
        super().__init__([qubit])

        default_options = self.experiment_options
        default_options.delays = delays
        default_options.unit = unit
        default_options.offset_frequency = offset_frequency

        self.set_experiment_options(**default_options.__dict__)

    @classmethod
    def _default_experiment_options(cls) -> Options:
        return Options(
            delays=None,
            unit="s",
            offset_frequency=0.,
            calibrations=dict(),
        )

    def circuits(self, backend: Optional[Backend] = None) -> List[QuantumCircuit]:
        """Return a list of experiment circuits.

        Args:
            backend: Optional, a backend object.

        Returns:
            A list of :class:`QuantumCircuit`.
        """
        options = self.experiment_options

        p_delay = Parameter("delay")

        ram_x = QuantumCircuit(1)
        ram_x.sx(0)
        ram_x.delay(p_delay, 0, unit=options.unit)
        ram_x.sx(0)
        ram_x.measure_active()

        ram_y = QuantumCircuit(1)
        ram_y.sx(0)
        ram_y.delay(p_delay, 0, unit=options.unit)
        ram_y.rz(-np.pi/2, 0)
        ram_y.sx(0)
        ram_y.measure_active()

        circs = []
        for delay in options.delays:

            # format delay to SI unit for analysis
            if options.unit == "dt":
                xval = delay * backend.configuration().dt
            else:
                if options.unit == "s":
                    xval = delay
                else:
                    xval = apply_prefix(delay, options.unit)

            metadata = {
                "experiment_type": self._type,
                "qubits": (self.physical_qubits[0], ),
                "delay": delay,
                "unit": options.unit,
                "xval": xval,
            }

            # create ramsey x
            assigned_x = ram_x.assign_parameters({p_delay: delay}, inplace=False)
            assigned_x.metadata = metadata.copy()
            assigned_x.metadata["post_pulse"] = "x"

            # create ramsey y
            assigned_y = ram_y.assign_parameters({p_delay: delay}, inplace=False)
            assigned_y.metadata = metadata.copy()
            assigned_y.metadata["post_pulse"] = "y"

            circs.extend([assigned_x, assigned_y])

        return circs

    def _postprocess_transpiled_circuits(self, circuits, backend, **run_options):
        """Attach gate calibration if available.

        TODO perhaps this go to base class if we can make consensus of how we add calibration
        i.e. currently it is user configurable field in experiment option.
        all experiments can update the gate definition in this way.
        """
        calibrations = self.experiment_options.calibrations

        if not any(calibrations.values()):
            return

        for circ in circuits:
            for gate_qubits, schedule in calibrations.items():
                try:
                    gate, qubits = gate_qubits
                except ValueError:
                    raise QiskitError(
                        f"Calibration has invalid gate definition {repr(gate_qubits)}. "
                        "This should be a tuple of gate name and qubit index."
                    )
                circ.add_calibration(gate=gate, qubits=qubits, schedule=schedule)
            circ.metadata["calibrations"] = calibrations
