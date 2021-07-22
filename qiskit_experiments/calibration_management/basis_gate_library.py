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

Note that the set of available library will be extended in future releases.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple

from qiskit.circuit import Parameter
import qiskit.pulse as pulse
from qiskit.pulse import ScheduleBlock

from qiskit_experiments.calibration_management.calibration_key_types import ParameterValueType
from qiskit_experiments.exceptions import CalibrationError


class BasisGateLibrary(ABC):
    """A base class for libraries of basis gates to make it easier to setup Calibrations.

    Calibration instances can be setup using subclasses of BasisGateLibrary. For example,

    .. code-block:: python

        cals = BackendCalibrations(
                FakeArmonk(),
                basis_gates=["x", "sx"],
                library=FixedFrequencyTransmon,
                default_values={"duration": 320},
            )

    creates an instance of :class:`BackendCalibrations` with the X and square-root X gates.
    These gates in this example are obtained from the :class:`FixedFrequencyTransmon` gate
    library. Here, we also override the default schedule duration of 160 samples and set
    it to 320 samples.
    """

    # Location where default parameter values are stored. These may be updated at construction.
    __default_values__ = dict()

    def __init__(self, default_values: Optional[Dict] = None):
        """Setup the library.

        Args:
            default_values: A dictionary to override library default parameter values.
        """
        self._schedules = dict()

        if default_values is not None:
            self.__default_values__.update(default_values)

    def __getitem__(self, name: str) -> ScheduleBlock:
        """Return the schedule."""
        if name not in self._schedules:
            raise CalibrationError(f"Gate {name} is not contained in {self.__class__.__name__}.")

        return self._schedules[name]

    @property
    def basis_gates(self) -> List[str]:
        """Return the basis gates supported by the library."""
        return list(name for name in self._schedules)

    @abstractmethod
    def default_values(self) -> List[Tuple[ParameterValueType, Parameter, Tuple, ScheduleBlock]]:
        """Return the default values for the parameters.

        Returns
            A list of tuples is returned. These tuples are structured so that instances of
            :class:`Calibrations` can call :meth:`add_parameter_value` on the tuples.
        """


class FixedFrequencyTransmon(BasisGateLibrary):
    """A library of gates for fixed-frequency superconducting qubit architectures.

    Note that for now this library supports single-qubit gates and will be extended
    in the future.
    """

    __default_values__ = {"duration": 160, "amp": 0.5, "β": 0.0}

    def __init__(self, default_values: Optional[Dict] = None, use_drag: bool = True):
        """Setup the schedules.

        Args:
            default_values: Default values for the parameters this dictionary can contain
                the following keys: "duration", "amp", "β", and "σ". If "σ" is not provided
                this library will take one fourth of the pulse duration as default value.
            use_drag: If set to False then Gaussian pulses will be used instead of DRAG
                pulses.
        """
        super().__init__(default_values)

        dur = Parameter("duration")
        sigma = Parameter("σ")

        def _beta(use_drag):
            return Parameter("β") if use_drag else None

        # X gates
        sched_x = self._single_qubit_schedule("x", dur, Parameter("amp"), sigma, _beta(use_drag))

        # Y gates
        sched_y = self._single_qubit_schedule("y", dur, Parameter("amp"), sigma, _beta(use_drag))

        # square-root X gates
        sched_sx = self._single_qubit_schedule("sx", dur, Parameter("amp"), sigma, _beta(use_drag))

        # square-root Y gates
        sched_sy = self._single_qubit_schedule("sy", dur, Parameter("amp"), sigma, _beta(use_drag))

        for sched in [sched_x, sched_y, sched_sx, sched_sy]:
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

                    if param.name == "σ" and "σ" not in self.__default_values__:
                        value = self.__default_values__["duration"] / 4
                    else:
                        value = self.__default_values__[param.name]

                    if name in {"sx", "sy"} and param.name == "amp":
                        value /= 2.0

                    if "y" in name and param.name == "amp":
                        value *= 1.0j

                    defaults.append((value, param.name, tuple(), name))

        return defaults
