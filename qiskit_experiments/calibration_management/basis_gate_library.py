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
A collections of libraries to setup Calibrations.

Note that the set of available libraries will be extended in future releases.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple
from warnings import warn

from qiskit.circuit import Parameter
import qiskit.pulse as pulse
from qiskit.pulse import ScheduleBlock

from qiskit_experiments.calibration_management.calibration_key_types import ParameterValueType
from qiskit_experiments.exceptions import CalibrationError
from qiskit_experiments.database_service.json import serialize_object, deserialize_object


class BasisGateLibrary(ABC):
    """A base class for libraries of basis gates to make it easier to setup Calibrations."""

    # Location where default parameter values are stored. These may be updated at construction.
    __default_values__ = dict()

    # The basis gates that this library generates. This is a dict with the gate name as key
    # and the number of qubits that gate acts on as values.
    __supported_gates__ = None

    def __init__(
        self, basis_gates: Optional[List[str]] = None, default_values: Optional[Dict] = None
    ):
        """Setup the library.

        Args:
            basis_gates: The basis gates to generate.
            default_values: A dictionary to override library default parameter values.

        Raises:
            CalibrationError: If on of the given basis gates is not supported by the library.
        """
        self._schedules = dict()

        # Update the default values.
        self._default_values = dict(self.__default_values__)
        if default_values is not None:
            self._default_values.update(default_values)

        if basis_gates is None:
            self._basis_gates = self.__supported_gates__

        else:
            self._basis_gates = dict()

            for gate in basis_gates:
                if gate not in self.__supported_gates__:
                    raise CalibrationError(
                        f"Gate {gate} is not supported by {self.__class__.__name__}. "
                        f"Supported gates are: {self.__supported_gates__}."
                    )

                self._basis_gates[gate] = self.__supported_gates__[gate]

    def __getitem__(self, name: str) -> ScheduleBlock:
        """Return the schedule."""
        if name not in self._schedules:
            raise CalibrationError(f"Gate {name} is not contained in {self.__class__.__name__}.")

        return self._schedules[name]

    def __contains__(self, name: str) -> bool:
        """Check if the basis gate is in the library."""
        return name in self._schedules

    def __hash__(self) -> int:
        """Return the hash of the library by computing the hash pf the schedule strings."""
        return hash(tuple(str(self[gate]) for gate in self.basis_gates))

    def num_qubits(self, schedule_name: str) -> int:
        """Return the number of qubits that the schedule with the given name acts on."""
        return self._basis_gates[schedule_name]

    @property
    def basis_gates(self) -> List[str]:
        """Return the basis gates supported by the library."""
        return list(name for name in self._schedules)

    @property
    def init_default_values(self) -> Dict:
        """The default values used in the initialization of the library."""
        return self._default_values

    @abstractmethod
    def default_values(self) -> List[Tuple[ParameterValueType, Parameter, Tuple, ScheduleBlock]]:
        """Return the default values for the parameters.

        Returns
            A list of tuples is returned. These tuples are structured so that instances of
            :class:`Calibrations` can call :meth:`add_parameter_value` on the tuples.
        """

    @abstractmethod
    def serialize(self) -> Dict:
        """Serialize the library."""

    @classmethod
    def deserialize(cls, serialized_dict: Dict) -> "BasisGateLibrary":
        """Deserialize the library given the input dictionary"""
        library = deserialize_object(
            serialized_dict["__value__"]["__module__"],
            serialized_dict["__value__"]["__name__"],
            tuple(),
            serialized_dict["__value__"]["__kwargs__"],
        )

        if hash(library) != serialized_dict["__value__"]["__schedule_hash__"]:
            warn(
                "Deserialized basis gate library's hash does not "
                "match the hash of the serialized library."
            )

        return library


class FixedFrequencyTransmon(BasisGateLibrary):
    """A library of gates for fixed-frequency superconducting qubit architectures.

    Note that for now this library supports single-qubit gates and will be extended
    in the future.
    """

    __default_values__ = {"duration": 160, "amp": 0.5, "β": 0.0}

    __supported_gates__ = {"x": 1, "y": 1, "sx": 1, "sy": 1}

    def __init__(
        self,
        basis_gates: Optional[List[str]] = None,
        default_values: Optional[Dict] = None,
        use_drag: bool = True,
        link_parameters: bool = True,
    ):
        """Setup the schedules.

        Args:
            basis_gates: The basis gates to generate.
            default_values: Default values for the parameters this dictionary can contain
                the following keys: "duration", "amp", "β", and "σ". If "σ" is not provided
                this library will take one fourth of the pulse duration as default value.
            use_drag: If set to False then Gaussian pulses will be used instead of DRAG
                pulses.
            link_parameters: if set to True then the amplitude and DRAG parameters of the
                X and Y gates will be linked as well as those of the SX and SY gates.
        """
        super().__init__(basis_gates, default_values)
        self._link_parameters = link_parameters
        self._use_drag = use_drag

        dur = Parameter("duration")
        sigma = Parameter("σ")

        # Generate the pulse parameters
        def _beta(use_drag):
            return Parameter("β") if use_drag else None

        x_amp, x_beta = Parameter("amp"), _beta(use_drag)

        if self._link_parameters:
            y_amp, y_beta = 1.0j * x_amp, x_beta
        else:
            y_amp, y_beta = Parameter("amp"), _beta(use_drag)

        sx_amp, sx_beta = Parameter("amp"), _beta(use_drag)

        if self._link_parameters:
            sy_amp, sy_beta = 1.0j * sx_amp, sx_beta
        else:
            sy_amp, sy_beta = Parameter("amp"), _beta(use_drag)

        # Create the schedules for the gates
        sched_x = self._single_qubit_schedule("x", dur, x_amp, sigma, x_beta)
        sched_y = self._single_qubit_schedule("y", dur, y_amp, sigma, y_beta)
        sched_sx = self._single_qubit_schedule("sx", dur, sx_amp, sigma, sx_beta)
        sched_sy = self._single_qubit_schedule("sy", dur, sy_amp, sigma, sy_beta)

        for sched in [sched_x, sched_y, sched_sx, sched_sy]:
            if sched.name in self._basis_gates:
                self._schedules[sched.name] = sched

    @staticmethod
    def _single_qubit_schedule(
        name: str,
        dur: Parameter,
        amp: Parameter,
        sigma: Parameter,
        beta: Optional[Parameter] = None,
    ) -> ScheduleBlock:
        """Build a single qubit pulse."""

        chan = pulse.DriveChannel(Parameter("ch0"))

        if beta is not None:
            with pulse.build(name=name) as sched:
                pulse.play(pulse.Drag(duration=dur, amp=amp, sigma=sigma, beta=beta), chan)
        else:
            with pulse.build(name=name) as sched:
                pulse.play(pulse.Gaussian(duration=dur, amp=amp, sigma=sigma), chan)

        return sched

    def default_values(self) -> List[Tuple[ParameterValueType, Parameter, Tuple, ScheduleBlock]]:
        """Return the default values for the parameters.

        Returns
            A list of tuples is returned. These tuples are structured so that instances of
            :class:`Calibrations` can call :meth:`add_parameter_value` on the tuples.
        """
        defaults = []
        for name in self.basis_gates:
            schedule = self._schedules[name]
            for param in schedule.parameters:
                if "ch" not in param.name:
                    if "y" in name and self._link_parameters:
                        continue

                    if param.name == "σ" and "σ" not in self._default_values:
                        value = self._default_values["duration"] / 4
                    else:
                        value = self._default_values[param.name]

                    if name in {"sx", "sy"} and param.name == "amp":
                        value /= 2.0

                    if "y" in name and param.name == "amp":
                        value *= 1.0j

                    defaults.append((value, param.name, tuple(), name))

        return defaults

    def serialize(self) -> Dict:
        """Serialize the object."""

        kwargs = {
            "basis_gates": self.basis_gates,
            "default_values": self._default_values,
            "use_drag": self._use_drag,
            "link_parameters": self._link_parameters,
        }

        serialized_library = serialize_object(type(self), kwargs=kwargs)
        serialized_library["__value__"]["__schedule_hash__"] = hash(self)

        return serialized_library
