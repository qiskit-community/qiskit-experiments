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

"""Store and manage the results of calibration experiments in the context of a backend."""

from collections import defaultdict
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union
import copy
from warnings import warn

from qiskit.providers.backend import BackendV1 as Backend
from qiskit.circuit import Parameter
from qiskit.pulse import InstructionScheduleMap, ScheduleBlock

from qiskit_experiments.framework.json import _serialize_type, _deserialize_object_legacy
from qiskit_experiments.calibration_management.parameter_value import ParameterValue
from qiskit_experiments.calibration_management.calibrations import (
    Calibrations,
    ParameterKey,
    ParameterValueType,
)
from qiskit_experiments.exceptions import CalibrationError
from qiskit_experiments.calibration_management.basis_gate_library import (
    BasisGateLibrary,
    deserialize_library,
)


class FrequencyElement(Enum):
    """An extendable enum for components that have a frequency."""

    QUBIT = "Qubit"
    READOUT = "Readout"


class BackendCalibrations(Calibrations):
    """
    A Calibrations class to enable a seamless interplay with backend objects.
    This class enables users to export their calibrations into a backend object.
    Additionally, it creates frequency parameters for qubits and readout resonators.
    The parameters are named `qubit_lo_freq` and `meas_lo_freq` to be consistent
    with the naming in backend.defaults(). These two parameters are not attached to
    any schedule.
    """

    __qubit_freq_parameter__ = "qubit_lo_freq"
    __readout_freq_parameter__ = "meas_lo_freq"

    def __init__(
        self,
        backend: Backend,
        library: BasisGateLibrary = None,
        add_parameter_defaults: bool = True,
    ):
        """Setup an instance to manage the calibrations of a backend.

        Args:
            backend: A backend instance from which to extract the qubit and readout frequencies
                (which will be added as first guesses for the corresponding parameters) as well
                as the coupling map.
            library: A library class that will be instantiated with the library options to then
                get template schedules to register as well as default parameter values.
            add_parameter_defaults: A boolean to indicate whether the default parameter values of
                the given library should be used to populate the calibrations. By default this
                value is True but can be set to false when deserializing a calibrations object.

        Raises:
            CalibrationError: If the backend configuration does not have num_qubits.
        """
        self._update_inst_map = False
        super().__init__(
            getattr(backend.configuration(), "control_channels", None),
            library,
            add_parameter_defaults,
        )

        # Instruction schedule map variables and support variables.
        self._inst_map = InstructionScheduleMap()
        self._operated_qubits = defaultdict(list)
        self._update_inst_map = False  # When True add_parameter_value triggers an inst. map update

        # Use the same naming convention as in backend.defaults()
        self.qubit_freq = Parameter(self.__qubit_freq_parameter__)
        self.meas_freq = Parameter(self.__readout_freq_parameter__)
        self._register_parameter(self.qubit_freq, ())
        self._register_parameter(self.meas_freq, ())

        self._qubits = list(range(backend.configuration().num_qubits))
        self._backend = backend

        if add_parameter_defaults:
            for qubit, freq in enumerate(backend.defaults().qubit_freq_est):
                self.add_parameter_value(freq, self.qubit_freq, qubit)

            for meas, freq in enumerate(backend.defaults().meas_freq_est):
                self.add_parameter_value(freq, self.meas_freq, meas)

        self._update_inst_map = True

        # Push the schedules to the instruction schedule map.
        self.update_inst_map()

    @property
    def default_inst_map(self) -> InstructionScheduleMap:
        """Return the default and up to date instruction schedule map."""
        return self._inst_map

    @property
    def backend(self) -> Backend:
        """Return the backend of the default cals."""
        return self._backend

    def get_inst_map(
        self,
        group: str = "default",
        cutoff_date: datetime = None,
    ) -> InstructionScheduleMap:
        """Get a new instance of an Instruction schedule map.

        Args:
            group: The calibration group from which to draw the parameters.
                If not specified this defaults to the 'default' group.
            cutoff_date: Retrieve the most recent parameter up until the cutoff date. Parameters
                generated after the cutoff date will be ignored. If the cutoff_date is None then
                all parameters are considered. This allows users to discard more recent values that
                may be erroneous.

        Returns:
            An instruction schedule map with parameters updated up to the desired cutoff date
            and from the desired calibration group.
        """
        inst_map = InstructionScheduleMap()

        self.update_inst_map(group=group, cutoff_date=cutoff_date, inst_map=inst_map)

        return inst_map

    def _get_frequencies(
        self,
        element: FrequencyElement,
        group: str = "default",
        cutoff_date: datetime = None,
    ) -> List[float]:
        """Internal helper method."""

        if element == FrequencyElement.READOUT:
            param = self.meas_freq.name
        elif element == FrequencyElement.QUBIT:
            param = self.qubit_freq.name
        else:
            raise CalibrationError(f"Frequency element {element} is not supported.")

        freqs = []
        for qubit in self._qubits:
            schedule = None  # A qubit frequency is not attached to a schedule.
            if ParameterKey(param, (qubit,), schedule) in self._params:
                freq = self.get_parameter_value(param, (qubit,), schedule, True, group, cutoff_date)
            else:
                if element == FrequencyElement.READOUT:
                    freq = self._backend.defaults().meas_freq_est[qubit]
                elif element == FrequencyElement.QUBIT:
                    freq = self._backend.defaults().qubit_freq_est[qubit]
                else:
                    raise CalibrationError(f"Frequency element {element} is not supported.")

            freqs.append(freq)

        return freqs

    def get_qubit_frequencies(
        self,
        group: str = "default",
        cutoff_date: datetime = None,
    ) -> List[float]:
        """
        Get the most recent qubit frequencies. They can be passed to the run-time
        options of :class:`BaseExperiment`. If no calibrated frequency value of a
        qubit is found then the default value from the backend defaults is used.
        Only valid parameter values are returned.

        Args:
            group: The calibration group from which to draw the
                parameters. If not specified, this defaults to the 'default' group.
            cutoff_date: Retrieve the most recent parameter up until the cutoff date. Parameters
                generated after the cutoff date will be ignored. If the cutoff_date is None then
                all parameters are considered. This allows users to discard more recent values
                that may be erroneous.

        Returns:
            A List of qubit frequencies for all qubits of the backend.
        """
        return self._get_frequencies(FrequencyElement.QUBIT, group, cutoff_date)

    def get_meas_frequencies(
        self,
        group: str = "default",
        cutoff_date: datetime = None,
    ) -> List[float]:
        """
        Get the most recent measurement frequencies. They can be passed to the run-time
        options of :class:`BaseExperiment`. If no calibrated frequency value of a
        measurement is found then the default value from the backend defaults is used.
        Only valid parameter values are returned.

        Args:
            group: The calibration group from which to draw the
                parameters. If not specified, this defaults to the 'default' group.
            cutoff_date: Retrieve the most recent parameter up until the cutoff date. Parameters
                generated after the cutoff date will be ignored. If the cutoff_date is None then
                all parameters are considered. This allows users to discard more recent values
                that may be erroneous.

        Returns:
            A List of measurement frequencies for all qubits of the backend.
        """
        return self._get_frequencies(FrequencyElement.READOUT, group, cutoff_date)

    def export_backend(self) -> Backend:
        """
        Exports the calibrations to a backend object that can be used.

        Returns:
            calibrated backend: A backend with the calibrations in it.
        """
        backend = copy.deepcopy(self._backend)

        backend.defaults().qubit_freq_est = self.get_qubit_frequencies()
        backend.defaults().meas_freq_est = self.get_meas_frequencies()
        backend.default().instruction_schedule_map = self._inst_map

        return backend

    def inst_map_add(
        self,
        instruction_name: str,
        qubits: Tuple[int],
        schedule_name: Optional[str] = None,
        assign_params: Optional[Dict[Union[str, ParameterKey], ParameterValueType]] = None,
    ):
        """Update a single instruction in the instruction schedule map.

        This method can be used to update a single instruction for the given qubits but
        it can also be used by experiments that define custom gates with parameters
        such as the :class:`Rabi` experiment. In a Rabi experiment there is a gate named
        "Rabi" that scans a pulse with a custom amplitude. Therefore we would do

        .. code-block:: python

            cals.inst_map_add("Rabi", (0, ), "xp", assign_params={"amp": Parameter("amp")})

        to temporarily add a pulse for the Rabi gate in the instruction schedule map. This
        then allows calling :code:`transpile(circ, inst_map=cals.instruction_schedule_map)`.

        Args:
            instruction_name: The name of the instruction to add to the instruction schedule map.
            qubits: The qubits to which the instruction will apply.
            schedule_name: The name of the schedule. If None is given then we assume that the
                schedule and the instruction have the same name.
            assign_params: An optional dict of parameter mappings to apply. See for instance
                :meth:`get_schedule` of :class:`Calibrations`.
        """
        schedule_name = schedule_name or instruction_name

        inst_map_args = None
        if assign_params is not None:
            inst_map_args = assign_params.keys()

        self._inst_map.add(
            instruction=instruction_name,
            qubits=qubits,
            schedule=self.get_schedule(schedule_name, qubits, assign_params),
            arguments=inst_map_args,
        )

    def update_inst_map(
        self,
        schedules: Optional[set] = None,
        qubits: Optional[Tuple[int]] = None,
        group: Optional[str] = "default",
        cutoff_date: datetime = None,
        inst_map: Optional[InstructionScheduleMap] = None,
    ):
        """Push all schedules from the Calibrations to the inst map.

        This will create instructions with the same name as the schedules.

        Args:
            schedules: The name of the schedules to update. If None is given then
                all schedules will be pushed to instructions.
            qubits: The qubits for which to update the instruction schedule map.
                If qubits is None then all possible schedules defined by the coupling
                map will be updated.
            group: The calibration group from which to draw the parameters. If not specified
                this defaults to the 'default' group.
            cutoff_date: Retrieve the most recent parameter up until the cutoff date. Parameters
                generated after the cutoff date will be ignored. If the cutoff_date is None then
                all parameters are considered. This allows users to discard more recent values that
                may be erroneous.
            inst_map: The instruction schedule map to update. If None is given then the default
                instruction schedule map (i.e. self._inst_map) will be updated.
        """
        inst_map = inst_map or self._inst_map

        for key in self._schedules:
            sched_name = key.schedule

            if schedules is not None and sched_name not in schedules:
                continue

            if qubits is not None:
                inst_map.add(
                    instruction=sched_name,
                    qubits=qubits,
                    schedule=self.get_schedule(
                        sched_name, qubits, group=group, cutoff_date=cutoff_date
                    ),
                )

            else:
                for qubits_ in self.operated_qubits[self._schedules_qubits[key]]:
                    try:
                        inst_map.add(
                            instruction=sched_name,
                            qubits=qubits_,
                            schedule=self.get_schedule(
                                sched_name, qubits_, group=group, cutoff_date=cutoff_date
                            ),
                        )
                    except CalibrationError:
                        # get_schedule may raise an error if not all parameters have values or
                        # default values. In this case we ignore and continue updating inst_map.
                        pass

    def _parameter_inst_map_update(self, param: Parameter):
        """Update all instructions in the inst map that contain the given parameter."""

        schedules = set(key.schedule for key in self._parameter_map_r[param])

        self.update_inst_map(schedules)

    def add_parameter_value(
        self,
        value: Union[int, float, complex, ParameterValue],
        param: Union[Parameter, str],
        qubits: Union[int, Tuple[int, ...]] = None,
        schedule: Union[ScheduleBlock, str] = None,
    ):
        """Add a parameter value to the stored parameters.

        This parameter value may be applied to several channels, for instance, all
        DRAG pulses may have the same standard deviation. Additionally, this function
        will update any instructions in the instruction schedule map that contain this
        parameter.

        Args:
            value: The value of the parameter to add. If an int, float, or complex is given
                then the timestamp of the parameter value will automatically be generated
                and set to the current local time of the user.
            param: The parameter or its name for which to add the measured value.
            qubits: The qubits to which this parameter applies.
            schedule: The schedule or its name for which to add the measured parameter value.
        """
        super().add_parameter_value(value, param, qubits, schedule)

        if self._update_inst_map:
            if schedule is not None:
                schedule = schedule.name if isinstance(schedule, ScheduleBlock) else schedule
                param_obj = self.calibration_parameter(param, qubits, schedule)
                self._parameter_inst_map_update(param_obj)

    @property
    def operated_qubits(self) -> Dict[int, List[int]]:
        """Get a dict describing qubit couplings.

        This is an extension of the coupling map and used as a convenience to help populate
        the instruction schedule map.

        Returns:
            A dict where the key is the number of qubits coupled and the value is a list of
            lists where the sublist shows which qubits are coupled. For example, a three qubit
            system with a three qubit gate and three two-qubit gates would be represented as

            .. parsed-literal::

                {
                    1: [[0], [1], [2]],
                    2: [[0, 1], [1, 2], [2, 1]],
                    3: [[0, 1, 2]]
                }
        """

        # Use the cached map if there is one.
        if len(self._operated_qubits) != 0:
            return self._operated_qubits

        # Single qubits
        for qubit in self._qubits:
            self._operated_qubits[1].append([qubit])

        # Multi-qubit couplings
        if self._backend.configuration().coupling_map is not None:
            for coupling in self._backend.configuration().coupling_map:
                self._operated_qubits[len(coupling)].append(coupling)

        return self._operated_qubits

    def config(self, save_parameters: bool = True) -> Dict:
        """Serializes the class to a Dictionary.

        Args:
            save_parameters: If set to True, the default value, then all the values of the
                calibrations will also be serialized.

        Returns:
            A dict object that represents the calibrations and can be used to rebuild the
            calibrations. See :meth:`deserialize`.

        Raises:
            CalibrationError: if the calibrations were not built from a library.
        """

        if self._library is None:
            raise CalibrationError(
                "Cannot serialize calibrations that are not constructed from a library."
            )

        serialized_cals = _serialize_type(type(self))
        serialized_cals["__value__"].update(
            {"library": self._library.config,
             "backend_name": self._backend.name(),
             "backend_version": self._backend.version,
             }
        )

        if save_parameters:
            serialized_cals["__value__"]["parameter_values"] = self.parameters_table()["data"]

        return serialized_cals

    # pylint: disable=arguments-differ
    @classmethod
    def from_config(cls, config: Dict, backend: Backend) -> "BackendCalibrations":
        """Deserialize from a dictionary.

        Args:
            config: The dictionary from which to create the calibrations instance.
            backend: The backend instance from which to construct the calibrations.

        Returns:
            An instance of Calibrations.

        Raises:
            CalibrationError: if the backend name does not match the name in the serialized data.
        """

        # Deserialize the library.
        library = deserialize_library(config["__value__"].pop("library"))

        expected_backend = config["__value__"]["backend_name"]
        expected_version = config["__value__"]["backend_version"]
        if backend.name() != expected_backend:
            raise CalibrationError(
                f"Wrong backend in deserialization: {backend.name} != {expected_backend}."
            )

        if backend.version != expected_version:
            warn(
                f"Deserialization Backend version mismatch {backend.version} != {expected_version}."
            )

        params = config["__value__"].get("parameter_values", [])

        cals = BackendCalibrations(
            backend=backend, library=library, add_parameter_defaults=len(params) == 0
        )

        # Add the parameter values if any
        params = config["__value__"].get("parameter_values", [])
        for param_conf in params:
            cals.add_parameter_value_from_conf(**param_conf)

        return cals
