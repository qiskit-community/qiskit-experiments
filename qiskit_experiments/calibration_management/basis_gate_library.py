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
from collections.abc import Mapping
from typing import Any, Dict, List, Optional, Set
from warnings import warn

from qiskit.circuit import Parameter
import qiskit.pulse as pulse
from qiskit.pulse import ScheduleBlock

from qiskit_experiments.calibration_management.calibration_key_types import DefaultCalValue
from qiskit_experiments.exceptions import CalibrationError


class BasisGateLibrary(ABC, Mapping):
    """A base class for libraries of basis gates to make it easier to setup Calibrations."""

    # Location where default parameter values are stored. These may be updated at construction.
    __default_values__ = dict()

    def __init__(
        self,
        basis_gates: Optional[List[str]] = None,
        default_values: Optional[Dict] = None,
        dependencies: Optional[List["BasisGateLibrary"]] = None,
        **extra_kwargs,
    ):
        """Setup the library.

        Args:
            basis_gates: The basis gates to generate.
            default_values: A dictionary to override library default parameter values.
            extra_kwargs: Extra key-word arguments of the subclasses that are saved to be able
                to reconstruct the library using the :meth:`__init__` method.
            dependencies: A list of libraries that ``self`` depends on, e.g., to call
                other schedules that it needs when building its own schedules.

        Raises:
            CalibrationError: If on of the given basis gates is not supported by the library.
        """
        # Update the default values.
        self._extra_kwargs = extra_kwargs
        self._default_values = self.__default_values__.copy()
        self._dependencies = dependencies
        if default_values is not None:
            self._default_values.update(default_values)

        if basis_gates is None:
            basis_gates = list(self.__supported_gates__)

        for gate in basis_gates:
            if gate not in self.__supported_gates__:
                raise CalibrationError(
                    f"Gate {gate} is not supported by {self.__class__.__name__}. "
                    f"Supported gates are: {self.__supported_gates__}."
                )

        self._schedules = self._build_schedules(set(basis_gates), dependencies)

    @property
    @abstractmethod
    def __supported_gates__(self) -> Dict[str, int]:
        """Return the supported gates of the library.

        The key is the name of the gate and the value is the number of qubits it applies to.
        """
        raise NotImplementedError

    def __getitem__(self, name: str) -> ScheduleBlock:
        """Return the schedule."""
        if name not in self._schedules:
            raise CalibrationError(f"Gate {name} is not contained in {self.__class__.__name__}.")

        return self._schedules[name]

    def __contains__(self, name: str) -> bool:
        """Check if the basis gate is in the library."""
        return name in self._schedules

    def __hash__(self) -> int:
        """Return the hash of the library by computing the hash of the schedule strings."""
        data_to_hash = []
        for name, schedule in sorted(self._schedules.items()):
            data_to_hash.append((name, str(schedule), self.__supported_gates__[name]))

        return hash(tuple(data_to_hash))

    def __len__(self):
        """The length of the library defined as the number of basis gates."""
        return len(self._schedules)

    def __iter__(self):
        """Return an iterator over the basis gate library."""
        return iter(self._schedules)

    def num_qubits(self, name: str) -> int:
        """Return the number of qubits that the schedule with the given name acts on."""
        return self.__supported_gates__[name]

    @property
    def basis_gates(self) -> List[str]:
        """Return the basis gates supported by the library."""
        return list(self._schedules)

    def default_values(self) -> List[DefaultCalValue]:
        """Return the default values for the parameters.

        Returns
            A list of tuples is returned. These tuples are structured so that instances of
            :class:`Calibrations` can call :meth:`add_parameter_value` on the tuples.
        """
        defaults = []
        for name, schedule in self.items():
            for param in schedule.parameters:
                if "ch" not in param.name:
                    # When the library calls schedules from another library then the default
                    # values should come from that other library to avoid adding them twice
                    # in the calibrations.
                    value = self._default_values.get(param.name, None)
                    if value is not None:
                        defaults.append(DefaultCalValue(value, param.name, tuple(), name))

        return defaults

    @abstractmethod
    def _build_schedules(
        self,
        basis_gates: Set[str],
        other_libraries: Optional[List["BasisGateLibrary"]] = None,
    ) -> Dict[str, ScheduleBlock]:
        """Build the schedules stored in the library.

        This method is called as the last step in the :meth:`__init__`. Subclasses must implement
        :meth:`_build_schedules` to build the schedules of the library based on the inputs given
        to the :meth:`__init__` method.

        Args:
            basis_gates: The set of basis gates to build.
            other_libraries: A list of other libraries that ``self`` can use to, e.g., call
                other schedules that it needs when building its own schedules.

        Returns:
            A dictionary where the keys are the names of the schedules/basis gates and the values
            are the corresponding schedules.
        """

    def config(self) -> Dict[str, Any]:
        """Return the settings used to initialize the library."""

        kwargs = {"basis_gates": self.basis_gates, "default_values": self._default_values}

        if self._dependencies is not None:
            kwargs["dependencies"] = self._dependencies

        kwargs.update(self._extra_kwargs)

        return {
            "class": self.__class__.__name__,
            "kwargs": kwargs,
            "hash": self.__hash__(),
        }

    @classmethod
    def from_config(cls, config: Dict) -> "BasisGateLibrary":
        """Deserialize the library given the input dictionary"""
        library = cls(**config["kwargs"])

        if hash(library) != config["hash"]:
            warn(
                "Deserialized basis gate library's hash does not match the hash of the serialized "
                "library. Typically, the hash changes when the internal structure of the template "
                "schedules has been changed."
            )

        return library

    def __json_encode__(self):
        """Convert to format that can be JSON serialized."""
        return self.config()

    @classmethod
    def __json_decode__(cls, value: Dict[str, Any]) -> "BasisGateLibrary":
        """Load from JSON compatible format."""
        return cls.from_config(value)


class FixedFrequencyTransmon(BasisGateLibrary):
    """A library of gates for fixed-frequency superconducting qubit architectures.

    Note that for now this library supports single-qubit gates and will be extended
    in the future.
    """

    __default_values__ = {"duration": 160, "amp": 0.5, "β": 0.0}

    def __init__(
        self,
        basis_gates: Optional[List[str]] = None,
        default_values: Optional[Dict] = None,
        link_parameters: bool = True,
    ):
        """Setup the schedules.

        Args:
            basis_gates: The basis gates to generate.
            default_values: Default values for the parameters this dictionary can contain
                the following keys: "duration", "amp", "β", and "σ". If "σ" is not provided
                this library will take one fourth of the pulse duration as default value.
            link_parameters: if set to True then the amplitude and DRAG parameters of the
                X and Y gates will be linked as well as those of the SX and SY gates.
        """
        self._link_parameters = link_parameters

        extra_kwargs = {"link_parameters": link_parameters}

        super().__init__(basis_gates, default_values, **extra_kwargs)

    @property
    def __supported_gates__(self) -> Dict[str, int]:
        """The gates that this library supports."""
        return {"x": 1, "y": 1, "sx": 1, "sy": 1}

    def _build_schedules(
        self,
        basis_gates: Set[str],
        other_libraries: Optional[List["BasisGateLibrary"]] = None,
    ) -> Dict[str, ScheduleBlock]:
        """Build the schedule of the class."""
        dur = Parameter("duration")
        sigma = Parameter("σ")

        x_amp, x_beta = Parameter("amp"), Parameter("β")

        if self._link_parameters:
            y_amp, y_beta = 1.0j * x_amp, x_beta
        else:
            y_amp, y_beta = Parameter("amp"), Parameter("β")

        sx_amp, sx_beta = Parameter("amp"), Parameter("β")

        if self._link_parameters:
            sy_amp, sy_beta = 1.0j * sx_amp, sx_beta
        else:
            sy_amp, sy_beta = Parameter("amp"), Parameter("β")

        # Create the schedules for the gates
        sched_x = self._single_qubit_schedule("x", dur, x_amp, sigma, x_beta)
        sched_y = self._single_qubit_schedule("y", dur, y_amp, sigma, y_beta)
        sched_sx = self._single_qubit_schedule("sx", dur, sx_amp, sigma, sx_beta)
        sched_sy = self._single_qubit_schedule("sy", dur, sy_amp, sigma, sy_beta)

        schedules = dict()
        for sched in [sched_x, sched_y, sched_sx, sched_sy]:
            if sched.name in basis_gates:
                schedules[sched.name] = sched

        return schedules

    @staticmethod
    def _single_qubit_schedule(
        name: str,
        dur: Parameter,
        amp: Parameter,
        sigma: Parameter,
        beta: Parameter,
    ) -> ScheduleBlock:
        """Build a single qubit pulse."""

        chan = pulse.DriveChannel(Parameter("ch0"))

        with pulse.build(name=name) as sched:
            pulse.play(pulse.Drag(duration=dur, amp=amp, sigma=sigma, beta=beta), chan)

        return sched

    def default_values(self) -> List[DefaultCalValue]:
        """Return the default values for the parameters.

        This overrides the method of the base class to account for parameter linkages and
        standard sigma to duration relations.

        Returns
            A list of tuples is returned. These tuples are structured so that instances of
            :class:`Calibrations` can call :meth:`add_parameter_value` on the tuples.
        """
        defaults = []
        for name, schedule in self.items():
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

                    defaults.append(DefaultCalValue(value, param.name, tuple(), name))

        return defaults


class EchoedCrossResonance(BasisGateLibrary):
    """A fixed-frequency transmon gate library with echoed cross-resonance gates.

    This library extends the FixedFrequencyTransmon library by adding support for
    cross-resonance gates.
    """

    __default_values__ = {
        "cr_duration": 640,
        "cr_width": 384,
        "cr_σ": 64,
        "cr_amp": 0.5,
        "rot_amp": 0.0,
    }

    def __init__(
        self,
        single_qubit_library: BasisGateLibrary,
        rotary: bool = True,
        basis_gates: Optional[List[str]] = None,
        default_values: Optional[Dict] = None,
    ):
        """Setup the schedules.

        Args:
            single_qubit_library: The library with the single-qubit gates from which
                the x-gate schedule will be called to build the echo.
            rotary: A boolean to indicate if a rotary tones are used in the CR gate.
            basis_gates: The basis gates to generate.
            default_values: Default values for the parameters this dictionary can contain
                the following keys: "duration", "amp", "β", and "σ". If "σ" is not provided
                this library will take one fourth of the pulse duration as default value.
        """
        self._rotary = rotary
        super().__init__(basis_gates, default_values, [single_qubit_library])

    @property
    def __supported_gates__(self) -> Dict[str, int]:
        """The gates that this library supports."""
        return {"cr": 2, "cr90p": 2, "cr90m": 2}

    def _build_schedules(
        self,
        basis_gates: Set[str],
        other_libraries: Optional[List["BasisGateLibrary"]] = None,
    ) -> Dict[str, ScheduleBlock]:
        """Build the schedules of the library."""
        if "x" not in other_libraries[0]:
            raise CalibrationError("x gate is required to build cross-resonance schedules.")

        target = pulse.DriveChannel(Parameter("ch1"))
        cr_chan = pulse.ControlChannel(Parameter("ch0.1"))

        cr_amp = Parameter("cr_amp")
        rot_amp = Parameter("rot_amp")
        cr_sigma = Parameter("cr_σ")
        cr_duration = Parameter("cr_duration")
        cr_width = Parameter("cr_width")

        cr90p = pulse.GaussianSquare(
            duration=cr_duration, amp=cr_amp, sigma=cr_sigma, width=cr_width, name="cr90p"
        )

        cr90m = pulse.GaussianSquare(
            duration=cr_duration, amp=-cr_amp, sigma=cr_sigma, width=cr_width, name="cr90m"
        )

        rot90p = pulse.GaussianSquare(
            duration=cr_duration, amp=rot_amp, sigma=cr_sigma, width=cr_width, name="rot90p"
        )

        rot90m = pulse.GaussianSquare(
            duration=cr_duration, amp=-rot_amp, sigma=cr_sigma, width=cr_width, name="rot90m"
        )

        with pulse.build(name="cr") as cr_sched:
            with pulse.align_sequential():
                with pulse.align_left():
                    pulse.play(cr90p, cr_chan)
                    if self._rotary:
                        pulse.play(rot90p, target)

                pulse.call(other_libraries[0]["x"])

                with pulse.align_left():
                    pulse.play(cr90m, cr_chan)
                    if self._rotary:
                        pulse.play(rot90m, target)

                pulse.call(other_libraries[0]["x"])

        with pulse.build(name="cr90p") as cr90p_sched:
            pulse.play(cr90p, cr_chan)
            if self._rotary:
                pulse.play(rot90p, target)

        with pulse.build(name="cr90m") as cr90m_sched:
            pulse.play(cr90m, cr_chan)
            if self._rotary:
                pulse.play(rot90m, target)

        return {"cr": cr_sched, "cr90m": cr90m_sched, "cr90p": cr90p_sched}

    def config(self) -> Dict[str, Any]:
        """Return the settings used to initialize the library."""
        conf = super().config()
        conf["kwargs"]["single_qubit_library"] = conf["kwargs"]["dependencies"][0]
        del conf["kwargs"]["dependencies"]
        return conf
