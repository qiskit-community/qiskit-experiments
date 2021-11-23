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
from typing import Dict, List, Optional, Tuple, Union

from qiskit.providers.backend import BackendV1 as Backend
from qiskit.circuit import Parameter
from qiskit.pulse import InstructionScheduleMap, ScheduleBlock, ControlChannel

from qiskit_experiments.calibration_management.parameter_value import ParameterValue
from qiskit_experiments.calibration_management.calibrations import (
    Calibrations,
    ParameterKey,
    ParameterValueType,
)
from qiskit_experiments.exceptions import CalibrationError
from qiskit_experiments.calibration_management.basis_gate_library import BasisGateLibrary


class BackendCalibrations(Calibrations):
    """
    A Calibrations class to enable a seamless interplay with backend objects.
    This class creates frequency parameters for qubits and readout resonators.
    The parameters are named `qubit_lo_freq` and `meas_lo_freq` to be consistent
    with the naming in backend.defaults(). These two parameters are not attached to
    any schedule.
    """

    __qubit_freq_parameter__ = "qubit_lo_freq"
    __readout_freq_parameter__ = "meas_lo_freq"

    def __init__(
        self,
        backend: Optional[Backend] = None,
        library: Optional[BasisGateLibrary] = None,
        add_parameter_defaults: bool = True,
        control_config: Optional[Dict[Tuple[int, ...], List[ControlChannel]]] = None,
        coupling_map: Optional[List[List[int]]] = None,
        num_qubits: Optional[int] = None,
    ):
        """Setup an instance to manage the calibrations of a backend.

        Args:
            backend: A backend instance from which to extract the qubit and readout frequencies
                (which will be added as first guesses for the corresponding parameters) as well
                as the coupling map. If the backend is not given then the calibrations will be
                initialized with the following key-word arguments: control_config, coupling_map,
                num_qubits.
            library: A library class that will be instantiated with the library options to then
                get template schedules to register as well as default parameter values.
            add_parameter_defaults: A boolean to indicate whether the default parameter values of
                the given library should be used to populate the calibrations. By default this
                value is True but can be set to false when deserializing a calibrations object.
            control_config: A configuration dictionary of any control channels. The
                keys are tuples of qubits and the values are a list of ControlChannels
                that correspond to the qubits in the keys. This argument is optional and only
                needed if the backend is not provided.
            coupling_map: The coupling map of the device. This option is not needed if the backend
                is provided.
            num_qubits: The number of qubits of the system. This parameter is only needed if the
                backend is not given.

        Raises:
            CalibrationError: if backend is None and any of control_config, coupling_map, num_qubits
                are None.
        """
        if backend is None:
            if any(var is None for var in [control_config, coupling_map, num_qubits]):
                raise CalibrationError(
                    "If backend is None then all of control_config, "
                    "coupling_map, and num_qubits must be given."
                )

        self._update_inst_map = False

        config = backend.configuration() if backend is not None else {}

        self._qubits = list(range(getattr(config, "num_qubits", num_qubits)))
        self._coupling_map = getattr(config, "coupling_map", coupling_map)

        super().__init__(
            getattr(config, "control_channels", control_config),
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

        if add_parameter_defaults and backend is not None:
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
        return [
            self.get_parameter_value(self.qubit_freq, qubit, group=group, cutoff_date=cutoff_date)
            for qubit in self._qubits
        ]

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
        return [
            self.get_parameter_value(self.meas_freq, qubit, group=group, cutoff_date=cutoff_date)
            for qubit in self._qubits
        ]

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
        if self._coupling_map is not None:
            for coupling in self._coupling_map:
                self._operated_qubits[len(coupling)].append(coupling)

        return self._operated_qubits
