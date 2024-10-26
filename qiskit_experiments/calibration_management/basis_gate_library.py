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
import numpy as np

from qiskit.circuit import Parameter
from qiskit import pulse
from qiskit.pulse import ScheduleBlock
from qiskit.utils.deprecation import deprecate_func

from qiskit_experiments.calibration_management.calibration_key_types import DefaultCalValue
from qiskit_experiments.exceptions import CalibrationError


class BasisGateLibrary(ABC, Mapping):
    """A base class for libraries of basis gates to make it easier to setup Calibrations."""

    # Location where default parameter values are stored. These may be updated at construction.
    __default_values__ = {}

    # Parameters that do not belong to a schedule, a set of names
    __parameters_without_schedule__ = set()

    @deprecate_func(
        since="0.8",
        package_name="qiskit-experiments",
        additional_msg=(
            "Due to the deprecation of Qiskit Pulse, support for pulse "
            "gate calibrations has been deprecated."
        ),
    )
    def __init__(
        self,
        basis_gates: Optional[List[str]] = None,
        default_values: Optional[Dict] = None,
        **extra_kwargs,
    ):
        """Setup the library.

        Args:
            basis_gates: The basis gates to generate.
            default_values: A dictionary to override library default parameter values.
            extra_kwargs: Extra key-word arguments of the subclasses that are saved to be able
                to reconstruct the library using the :meth:`__init__` method.

        Raises:
            CalibrationError: If on of the given basis gates is not supported by the library.
        """
        # Update the default values.
        self._extra_kwargs = extra_kwargs
        self._default_values = self.__default_values__.copy()
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

        self._schedules = self._build_schedules(set(basis_gates))

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

    @abstractmethod
    def default_values(self) -> List[DefaultCalValue]:
        """Return the default values for the parameters.

        Returns
            A list of tuples is returned. These tuples are structured so that instances
            of :class:`.Calibrations` can call :meth:`.Calibrations.add_parameter_value`
            on the tuples.
        """

    @abstractmethod
    def _build_schedules(self, basis_gates: Set[str]) -> Dict[str, ScheduleBlock]:
        """Build the schedules stored in the library.

        This method is called as the last step in the :meth:`__init__`. Subclasses must implement
        :meth:`_build_schedules` to build the schedules of the library based on the inputs given
        to the :meth:`__init__` method.

        Args:
            basis_gates: The set of basis gates to build. These will be the supported gates or
                a subset thereof.

        Returns:
            A dictionary where the keys are the names of the schedules/basis gates and the values
            are the corresponding schedules.
        """

    def config(self) -> Dict[str, Any]:
        """Return the settings used to initialize the library."""

        kwargs = {"basis_gates": self.basis_gates, "default_values": self._default_values}
        kwargs.update(self._extra_kwargs)

        return {
            "class": self.__class__.__name__,
            "kwargs": kwargs,
            "hash": hash(self),
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
    r"""A library of gates for fixed-frequency superconducting qubit architectures.

    Note that for now this library supports single-qubit gates and will be extended
    in the future.

    Provided gates:
        - x: :math:`\pi` pulse around the x-axis.
        - sx: :math:`\pi/2` pulse around the x-axis.
        - y: :math:`\pi` pulse around the y-axis.
        - sy: :math:`\pi/2` pulse around the y-axis.

    Pulse parameters:
        - duration: Duration of the pulses Default value: 160 samples.
        - σ: Standard deviation of the pulses Default value: ``duration / 4``.
        - β: DRAG parameter of the pulses Default value: 0.
        - amp: Magnitude of the complex amplitude of the pulses. If the parameters are
          linked then ``x`` and ``y``
          share the same parameter and ``sx`` and ``sy`` share the same parameter.
          Default value: 50% of the maximum output for ``x`` and ``y`` and 25% of the
          maximum output for ``sx`` and ``sy``. Note that the user provided default amplitude
          in the ``__init__`` method sets the default amplitude of the ``x`` and ``y`` pulses.
          The amplitude of the ``sx`` and ``sy`` pulses is half the provided value.
        - angle: The phase of the complex amplitude of the pulses.

    Parameters without schedule:
        - meas_freq: frequency of the measurement drives.
        - drive_freq: frequency of the qubit drives.

    Note that the β and amp parameters may be linked between the x and y as well as between
    the sx and sy pulses. All pulses share the same duration and σ parameters.
    """

    __default_values__ = {"duration": 160, "amp": 0.5, "β": 0.0, "angle": 0.0}

    __parameters_without_schedule__ = {"meas_freq", "drive_freq"}

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
            link_parameters: If set to ``True``, then the amplitude and DRAG parameters of the
                :math:`X` and :math:`Y` gates will be linked as well as those of
                the :math:`SX` and :math:`SY` gates.
        """
        self._link_parameters = link_parameters

        extra_kwargs = {"link_parameters": link_parameters}

        super().__init__(basis_gates, default_values, **extra_kwargs)

    @property
    def __supported_gates__(self) -> Dict[str, int]:
        """The gates that this library supports."""
        return {"x": 1, "y": 1, "sx": 1, "sy": 1}

    def _build_schedules(self, basis_gates: Set[str]) -> Dict[str, ScheduleBlock]:
        """Build the schedule of the class."""
        dur = Parameter("duration")
        sigma = Parameter("σ")

        x_amp, x_beta, x_angle = Parameter("amp"), Parameter("β"), Parameter("angle")

        if self._link_parameters:
            y_amp, y_beta, y_angle = x_amp, x_beta, x_angle + np.pi / 2
        else:
            y_amp, y_beta, y_angle = Parameter("amp"), Parameter("β"), Parameter("angle")

        sx_amp, sx_beta, sx_angle = Parameter("amp"), Parameter("β"), Parameter("angle")

        if self._link_parameters:
            sy_amp, sy_beta, sy_angle = sx_amp, sx_beta, sx_angle + np.pi / 2
        else:
            sy_amp, sy_beta, sy_angle = Parameter("amp"), Parameter("β"), Parameter("angle")

        # Create the schedules for the gates
        sched_x = self._single_qubit_schedule("x", dur, x_amp, sigma, x_beta, x_angle)
        sched_y = self._single_qubit_schedule("y", dur, y_amp, sigma, y_beta, y_angle)
        sched_sx = self._single_qubit_schedule("sx", dur, sx_amp, sigma, sx_beta, sx_angle)
        sched_sy = self._single_qubit_schedule("sy", dur, sy_amp, sigma, sy_beta, sy_angle)

        schedules = {}
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
        angle: Parameter,
    ) -> ScheduleBlock:
        """Build a single qubit pulse."""

        chan = pulse.DriveChannel(Parameter("ch0"))

        with pulse.build(name=name) as sched:
            pulse.play(pulse.Drag(duration=dur, amp=amp, sigma=sigma, beta=beta, angle=angle), chan)

        return sched

    def default_values(self) -> List[DefaultCalValue]:
        """Return the default values for the parameters.

        Returns
            A list of tuples is returned. These tuples are structured so that instances
            of :class:`.Calibrations` can call :meth:`.Calibrations.add_parameter_value`
            on the tuples.
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

                    if "y" in name and param.name == "angle":
                        value += np.pi / 2

                    defaults.append(DefaultCalValue(value, param.name, tuple(), name))

        return defaults


class EchoedCrossResonance(BasisGateLibrary):
    r"""A library for echoed cross-resonance gates.

    The ``cr45p`` and ``cr45m`` include a pulse on the control qubit and optionally a pulse
    on the target qubit.

    Provided gates:
        - cr45p: GaussianSquare cross-resonance gate for a :math:`+\pi/4` rotation.
        - cr45m: GaussianSquare cross-resonance gate for a :math:`-\pi/4` rotation.
        - ecr: Echoed cross-resonance gate defined as ``cr45p - x - cr45m``.
        - rzx: RZXGate built from the ecr as ``cr45p - x - cr45m - x``.

    Required gates:
        - x: the x gate is defined outside of this library, see :class:`.FixedFrequencyTransmon`.

    Pulse parameters:
        - tgt_amp: The amplitude of the pulse applied to the target qubit. Default value: 0.
        - σ: The standard deviation of the flanks. Default value: 64 samples.
        - amp: The amplitude of the pulses applied to the control qubit. Default value: 50%.
        - duration: The duration of the cr45p and cr45m pulses. Default value: 1168 samples.
        - risefall: The number of σ's in the flanks of the pulses. Default value: 2.
    """

    __default_values__ = {
        "tgt_amp": 0.0,
        "tgt_angle": 0.0,
        "amp": 0.5,
        "angle": 0.0,
        "σ": 64,
        "risefall": 2,
        "duration": 1168,
    }

    def __init__(
        self,
        basis_gates: Optional[List[str]] = None,
        default_values: Optional[Dict] = None,
        target_pulses: bool = True,
    ):
        """Setup the library.

        Args:
            basis_gates: The basis gates to generate.
            default_values: A dictionary to override library default parameter values.
            target_pulses: If True (the default) then drives will be added to the target qubit
                during the CR tones on the control qubit.
        """
        self._target_pulses = target_pulses
        super().__init__(basis_gates, default_values)

    @property
    def __supported_gates__(self) -> Dict[str, int]:
        """The supported gates of the library are two-qubit pulses for the ecr gate."""
        return {"cr45p": 2, "cr45m": 2, "ecr": 2, "rzx": 2}

    def default_values(self) -> List[DefaultCalValue]:
        """The default values of the CR library."""
        defaults = []
        for name, schedule in self.items():
            for param in schedule.parameters:
                if "ch" not in param.name:
                    value = self._default_values[param.name]
                    defaults.append(DefaultCalValue(value, param.name, tuple(), name))

        return defaults

    def _build_schedules(self, basis_gates: Set[str]) -> Dict[str, ScheduleBlock]:
        """Build the schedules of the CR library."""

        schedules = {}

        tgt_amp = Parameter("tgt_amp")
        tgt_angle = Parameter("tgt_angle")
        sigma = Parameter("σ")
        cr_amp = Parameter("amp")
        cr_angle = Parameter("angle")
        cr_dur = Parameter("duration")
        cr_rf = Parameter("risefall")
        t_chan_idx = Parameter("ch1")
        u_chan_idx = Parameter("ch0.1")
        t_chan = pulse.DriveChannel(t_chan_idx)
        u_chan = pulse.ControlChannel(u_chan_idx)

        if "cr45p" in basis_gates:
            with pulse.build(name="cr45p") as cr45p:
                pulse.play(
                    pulse.GaussianSquare(
                        cr_dur, cr_amp, angle=cr_angle, risefall_sigma_ratio=cr_rf, sigma=sigma
                    ),
                    u_chan,
                )

                if self._target_pulses:
                    pulse.play(
                        pulse.GaussianSquare(
                            cr_dur,
                            tgt_amp,
                            angle=tgt_angle,
                            risefall_sigma_ratio=cr_rf,
                            sigma=sigma,
                        ),
                        t_chan,
                    )

            schedules["cr45p"] = cr45p

        if "cr45m" in basis_gates:
            with pulse.build(name="cr45m") as cr45m:
                pulse.play(
                    pulse.GaussianSquare(
                        cr_dur,
                        cr_amp,
                        angle=cr_angle + np.pi,
                        risefall_sigma_ratio=cr_rf,
                        sigma=sigma,
                    ),
                    u_chan,
                )

                if self._target_pulses:
                    pulse.play(
                        pulse.GaussianSquare(
                            cr_dur,
                            tgt_amp,
                            angle=tgt_angle + np.pi,
                            risefall_sigma_ratio=cr_rf,
                            sigma=sigma,
                        ),
                        t_chan,
                    )

            schedules["cr45m"] = cr45m

        # Echoed Cross-Resonance gate
        if "ecr" in basis_gates:
            with pulse.build(name="ecr") as ecr:
                with pulse.align_sequential():
                    pulse.reference("cr45p", "q0", "q1")
                    pulse.reference("x", "q0")
                    pulse.reference("cr45m", "q0", "q1")

            schedules["ecr"] = ecr

        # RZXGate built from Echoed Cross-Resonance gate
        if "rzx" in basis_gates:
            with pulse.build(name="rzx") as rzx:
                with pulse.align_sequential():
                    pulse.reference("cr45p", "q0", "q1")
                    pulse.reference("x", "q0")
                    pulse.reference("cr45m", "q0", "q1")
                    pulse.reference("x", "q0")

            schedules["rzx"] = rzx

        return schedules
