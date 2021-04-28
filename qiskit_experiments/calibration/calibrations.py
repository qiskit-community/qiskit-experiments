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

"""Class to store and manage the results of a calibration experiments."""

from collections import namedtuple, defaultdict
from datetime import datetime
from typing import Any, Dict, Set, Tuple, Union, List, Optional
import csv
import dataclasses
import regex as re

from qiskit.circuit import Gate
from qiskit import QuantumCircuit
from qiskit.pulse import (
    Schedule,
    DriveChannel,
    ControlChannel,
    MeasureChannel,
    Call,
    Play,
    ShiftPhase,
    SetPhase,
    ShiftFrequency,
    SetFrequency,
)
from qiskit.pulse.channels import PulseChannel
from qiskit.pulse.transforms import inline_subroutines
from qiskit.circuit import Parameter, ParameterExpression
from qiskit_experiments.calibration.exceptions import CalibrationError
from qiskit_experiments.calibration.parameter_value import ParameterValue

ParameterKey = namedtuple("ParameterKey", ["schedule", "parameter", "qubits"])
ScheduleKey = namedtuple("ScheduleKey", ["schedule", "qubits"])


class Calibrations:
    r"""
    A class to manage schedules with calibrated parameter values. Schedules are
    intended to be fully parameterized, including the index of the channels.
    This class:
    - supports having different schedules share parameters
    - allows default schedules for qubits that can be overridden for specific qubits.

    Parametric channel naming convention.
    Channels must be name according to a predefined pattern so that self can resolve
    the channels and control channels when assigning values to the parametric channel
    indices. This pattern is "^ch\d[.\d]*\${0,1}[\d]*$", examples of which include "ch0",
    "ch1", "ch0.1", "ch0$", "ch2$3", and "ch1.0.3$2". The "." delimiter is used to
    specify the different qubits when looking for control channels.
    The optional $ delimiter is used to specify which control channel to use
    if several control channels work together on the same qubits. For example, if the
    control channel configuration is {(3,2): [ControlChannel(3), ControlChannel(12)]}
    then given qubits (2, 3) the name "ch1.0$1" will resolve to ControlChannel(12) while
    "ch1.0$0" will resolve to ControlChannel(3).

    Parameter naming restriction.
    Each parameter must have a unique name within each schedule. For example, it is
    acceptable to have a parameter named 'amp' in the schedule 'xp' and a different
    parameter instance named 'amp' in the schedule named 'xm'. It is not acceptable
    to have two parameters named 'amp' in the same schedule.
    """

    # The channel indices need to be parameterized following this regex.
    __channel_pattern__ = r"^ch\d[\.\d]*\${0,1}[\d]*$"

    def __init__(self, control_config: Dict[Tuple[int, ...], List[ControlChannel]] = None):
        """
        Initialize the calibrations.

        Args:
            control_config: A configuration dictionary of any control channels. The
                keys are tuples of qubits and the values are a list of ControlChannels
                that correspond to the qubits in the keys.
        """

        self._controls_config = control_config if control_config else {}

        # Dict of the form: (schedule.name, parameter.name, qubits): Parameter
        self._parameter_map = {}

        # Reverse mapping of _parameter_map
        self._parameter_map_r = defaultdict(set)

        # Default dict of the form: (schedule.name, parameter.name, qubits): [ParameterValue, ...]
        self._params = defaultdict(list)

        self._schedules = {}

    def add_schedule(self, schedule: Schedule, qubits: Tuple = None):
        """
        Add a schedule and register its parameters.

        Args:
            schedule: The schedule to add.
            qubits: The qubits for which to add the schedules. If None is given then this
                schedule is the default schedule for all qubits.

        Raises:
            CalibrationError: If the parameterized channel index is not formatted
                following index1.index2... or if several parameters in the same schedule
                have the same name.
        """
        # check that channels, if parameterized, have the proper name format.
        param_indices = set()
        for ch in schedule.channels:
            if isinstance(ch.index, Parameter):
                param_indices.add(ch.index)
                if re.compile(self.__channel_pattern__).match(ch.index.name) is None:
                    raise CalibrationError(
                        f"Parameterized channel must correspond to {self.__channel_pattern__}"
                    )

        self._schedules[ScheduleKey(schedule.name, qubits)] = schedule

        param_names = [param.name for param in schedule.parameters]

        if len(param_names) != len(set(param_names)):
            raise CalibrationError(f"Parameter names in {schedule.name} must be unique.")

        # Register parameters that are not indices.
        for param in schedule.parameters:
            if param not in param_indices:
                self._register_parameter(param, schedule, qubits)

    def _register_parameter(
        self, parameter: Parameter, schedule: Schedule = None, qubits: Tuple = None
    ):
        """
        Registers a parameter for the given schedule. This allows self to determine the
        parameter instance that corresponds to the given schedule name, parameter name
        and qubits.

        Args:
            parameter: The parameter to register.
            schedule: The Schedule to which this parameter belongs. The schedule can
                be None which allows the calibration to accommodate, e.g. qubit frequencies.
            qubits: The qubits for which to register the parameter.
        """
        sched_name = schedule.name if schedule else None
        key = ParameterKey(sched_name, parameter.name, qubits)
        self._parameter_map[key] = parameter
        self._parameter_map_r[parameter].add(key)

    @property
    def parameters(self) -> Dict[Parameter, Set[ParameterKey]]:
        """
        Returns a dictionary mapping parameters managed by the calibrations to the schedules and
        qubits and parameter names using the parameters. The values of the dict are sets containing
        the parameter keys. Parameters that are not attached to a schedule will have None in place
        of a schedule name.
        """
        parameters = {}
        for param, key in self._parameter_map_r.items():
            parameters[(param, hash(param))] = key

        return parameters

    def calibration_parameter(
        self, parameter_name: str, qubits: Tuple[int, ...] = None, schedule_name: str = None
    ) -> Parameter:
        """
        Returns a Parameter object given the triplet parameter_name, qubits and schedule_name
        which uniquely determine the context of a parameter.

        Args:
            parameter_name: Name of the parameter to get.
            qubits: The qubits to which this parameter belongs. If qubits is None then
                the default scope is assumed.
            schedule_name: The name of the schedule to which this parameter belongs. A
                parameter may not belong to a schedule in which case None is accepted.

        Returns:
             calibration parameter: The parameter that corresponds to the given arguments.

        Raises:
            CalibrationError: If the desired parameter is not found.
        """
        # 1) Check for qubit specific parameters.
        if (schedule_name, parameter_name, qubits) in self._parameter_map:
            return self._parameter_map[(schedule_name, parameter_name, qubits)]

        # 2) Check for default parameters.
        elif (schedule_name, parameter_name, None) in self._parameter_map:
            return self._parameter_map[(schedule_name, parameter_name, None)]
        else:
            raise CalibrationError(
                f"No parameter for {parameter_name} and schedule {schedule_name} "
                f"and qubits {qubits}. No default value exists."
            )

    def add_parameter_value(
        self,
        value: Union[int, float, complex, ParameterValue],
        param: Union[Parameter, str],
        qubits: Tuple[int, ...] = None,
        schedule: Union[Schedule, str] = None,
    ):
        """
        Add a parameter value to the stored parameters. This parameter value may be
        applied to several channels, for instance, all DRAG pulses may have the same
        standard deviation.

        Args:
            value: The value of the parameter to add. If an int, float, or complex is given
                then the timestamp of the parameter values will automatically be generated to
                correspond to the current time.
            param: The parameter or its name for which to add the measured value.
            qubits: The qubits to which this parameter applies.
            schedule: The schedule or its name for which to add the measured parameter value.

        Raises:
            CalibrationError: If the schedule name is given but no schedule with that name
                exists.
        """
        if isinstance(value, (int, float, complex)):
            value = ParameterValue(value, datetime.now())

        param_name = param.name if isinstance(param, Parameter) else param
        sched_name = schedule.name if isinstance(schedule, Schedule) else schedule

        registered_schedules = set(key.schedule for key in self._schedules)

        if sched_name and sched_name not in registered_schedules:
            raise CalibrationError(f"Schedule named {sched_name} was never registered.")

        self._params[ParameterKey(sched_name, param_name, qubits)].append(value)

    def _get_channel_index(self, qubits: Tuple, chan: PulseChannel) -> int:
        """
        Get the index of the parameterized channel based on the given qubits
        and the name of the parameter in the channel index. The name of this
        parameter for control channels must be written as qubit_index1.qubit_index2... .
        For example, the following parameter names are valid: 'ch1', 'ch1.0', 'ch30.12'.

        Args:
            qubits: The qubits for which we want to obtain the channel index.
            chan: The channel with a parameterized name.
            pair of qubits has more than one control channel.

        Returns:
            index: The index of the channel. For example, if qubits=(10, 32) and
                chan is a control channel with parameterized index name 'ch1.0'
                the method returns the control channel corresponding to
                qubits (qubits[1], qubits[0]) which is here the control channel of
                qubits (32, 10).

        Raises:
            CalibrationError: if the number of qubits is incorrect, if the
                number of inferred ControlChannels is not correct, or if ch is not
                a DriveChannel, MeasureChannel, or ControlChannel.
        """
        if isinstance(chan.index, Parameter):
            if isinstance(chan, (DriveChannel, MeasureChannel)):
                index = int(chan.index.name[2:].replace("ch", "").split("$")[0])

                if len(qubits) < index:
                    raise CalibrationError(f"Not enough qubits given for channel {chan}.")

                return qubits[index]

            # Control channels name example ch1.0$1
            if isinstance(chan, ControlChannel):

                channel_index_parts = chan.index.name[2:].split("$")
                qubit_channels = channel_index_parts[0]

                indices = [int(sub_channel) for sub_channel in qubit_channels.split(".")]
                ch_qubits = tuple(qubits[index] for index in indices)
                chs_ = self._controls_config[ch_qubits]

                control_index = 0
                if len(channel_index_parts) == 2:
                    control_index = int(channel_index_parts[1])

                if len(chs_) < control_index:
                    raise CalibrationError(
                        f"Control channel index {control_index} not found for qubits {qubits}."
                    )

                return chs_[control_index].index

            raise CalibrationError(f"{chan} must be a sub-type of {PulseChannel}.")

        return chan.index

    def get_parameter_value(
        self,
        param: Union[Parameter, str],
        qubits: Tuple[int, ...],
        schedule: Union[Schedule, str, None] = None,
        valid_only: bool = True,
        group: str = "default",
        cutoff_date: datetime = None,
    ) -> Union[int, float, complex]:
        """
        Retrieves the value of a parameter. Parameters may be linked. get_parameter_value does the
        following steps:
        1) Retrieve the parameter object corresponding to (param, qubits, schedule)
        2) The values of this parameter may be stored under another schedule since
           schedules can share parameters. To deal we this a list of candidate keys
           is created internally based on the current configuration.
        3) Look for candidate parameter values under the candidate keys.
        4) Filter the candidate parameter values according to their date (up until the
           cutoff_date), validity and calibration group.
        5) Return the most recent parameter.

        Args:
            param: The parameter or the name of the parameter for which to get the parameter value.
            qubits: The qubits for which to get the value of the parameter.
            schedule: The schedule or its name for which to get the parameter value.
            valid_only: Use only parameters marked as valid.
            group: The calibration group from which to draw the
                parameters. If not specifies this defaults to the 'default' group.
            cutoff_date: Retrieve the most recent parameter up until the cutoff date. Parameters
                generated after the cutoff date will be ignored. If the cutoff_date is None then
                all parameters are considered. This allows users to discard more recent values that
                may be erroneous.

        Returns:
            value: The value of the parameter.

        Raises:
            CalibrationError: if there is no parameter value for the given parameter name
                and pulse channel or if there is an inconsistency in the stored parameters.
        """

        # 1) Identify the parameter object.
        param_name = param.name if isinstance(param, Parameter) else param
        sched_name = schedule.name if isinstance(schedule, Schedule) else schedule

        param = self.calibration_parameter(param_name, qubits, sched_name)

        # 2) Get a list of candidate keys restricted to the qubits of interest.
        candidate_keys = []
        for key in self._parameter_map_r[param]:
            candidate_keys.append(ParameterKey(key.schedule, key.parameter, qubits))

        # 3) Loop though the candidate keys to candidate values
        candidates = []
        parameter_not_found = True
        for key in candidate_keys:
            if key in self._params:
                if parameter_not_found:
                    candidates = self._params[key]
                    parameter_not_found = False
                else:
                    raise CalibrationError("Duplicate parameters.")

        # If no candidate parameter values were found look for default parameters
        # i.e. parameters that do not specify a qubit.
        if len(candidates) == 0:
            candidate_default_keys = []

            for key in candidate_keys:
                candidate_default_keys.append(ParameterKey(key.schedule, key.parameter, None))

            candidate_default_keys = set(candidate_default_keys)

            for key in set(candidate_default_keys):
                if key in self._params:
                    if parameter_not_found:
                        candidates = self._params[key]
                        parameter_not_found = False
                    else:
                        raise CalibrationError("Duplicate parameters.")

        # 4) Filter candidate parameter values.
        if valid_only:
            candidates = [val for val in candidates if val.valid]

        candidates = [val for val in candidates if val.group == group]

        if cutoff_date:
            candidates = [val for val in candidates if val.date_time <= cutoff_date]

        if len(candidates) == 0:
            msg = f"No candidate parameter values for {param_name} in calibration group {group} "

            if qubits:
                msg += f"on qubits {qubits} "

            msg += f"in schedule {sched_name}"

            if cutoff_date:
                msg += f" Cutoff date: {cutoff_date}"

            raise CalibrationError(msg)

        # 5) Return the most recent parameter.
        candidates.sort(key=lambda x: x.date_time)

        return candidates[-1].value

    def get_schedule(
        self,
        name: str,
        qubits: Tuple[int, ...],
        free_params: List[str] = None,
        group: Optional[str] = "default",
    ) -> Schedule:
        """
        Get the schedule with the non-free parameters assigned to their values.

        Args:
            name: The name of the schedule to get.
            qubits: The qubits for which to get the schedule.
            free_params: The parameters that should remain unassigned.
            group: The calibration group from which to draw the
                parameters. If not specifies this defaults to the 'default' group.

        Returns:
            schedule: A copy of the template schedule with all parameters assigned
                except for those specified by free_params.

        Raises:
            CalibrationError: if the name of the schedule is not known or if a parameter could
                not be found.
        """
        if (name, qubits) in self._schedules:
            schedule = self._schedules[ScheduleKey(name, qubits)]
        elif (name, None) in self._schedules:
            schedule = self._schedules[ScheduleKey(name, None)]
        else:
            raise CalibrationError(f"Schedule {name} is not defined for qubits {qubits}.")

        # Retrieve the channel indices based on the qubits and bind them.
        binding_dict = {}
        for ch in schedule.channels:
            if ch.is_parameterized():
                binding_dict[ch.index] = self._get_channel_index(qubits, ch)

        # Binding the channel indices makes it easier to deal with parameters later on
        schedule = schedule.assign_parameters(binding_dict, inplace=False)

        # The following code allows us to get the keys when the schedule has call instructions.
        # We cannot inline the subroutines yet because we would lose the name of the subroutines.
        parameter_keys = Calibrations.get_parameter_keys(schedule, set(), binding_dict, qubits)

        # Now that we have the parameter keys we must inline all call subroutines.
        schedule = inline_subroutines(schedule)

        # Build the parameter binding dictionary.
        free_params = free_params if free_params else []

        for key in parameter_keys:
            if key.parameter not in free_params:
                # Get the parameter object. Since we are dealing with a schedule the name of
                # the schedule is always defined. However, the parameter may be a default
                # parameter for all qubits, i.e. qubits may be None.
                if key in self._parameter_map:
                    param = self._parameter_map[key]
                elif (key.schedule, key.parameter, None) in self._parameter_map:
                    param = self._parameter_map[(key.schedule, key.parameter, None)]
                else:
                    raise CalibrationError(
                        f"Ill configured calibrations {key} is not present and has not default value."
                    )

                if param not in binding_dict:
                    binding_dict[param] = self.get_parameter_value(
                        key.parameter, key.qubits, key.schedule, group=group
                    )

        return schedule.assign_parameters(binding_dict, inplace=False)

    @staticmethod
    def get_parameter_keys(
        schedule: Schedule, keys: Set, binding_dict: Dict[Parameter, int], qubits: Tuple[int, ...]
    ):
        """
        Recursive function to extract parameter keys from a schedule. The recursive
        behaviour is needed to handle Call instructions. Each time a Call is found
        get_parameter_keys is call on the subroutine of the Call instruction and the
        qubits that are in the subroutine. This also implies carefully extracting the
        qubits from the subroutine and in the appropriate order.

        Args:
            schedule: A schedule from which to extract parameters.
            keys: A set of keys that will be populated.
            binding_dict: A binding dictionary intended only for channels. This is needed
                because calling assign_parameters on a schedule with a Call instruction will
                not assign the parameters in the subroutine of the Call instruction.
            qubits: The qubits for which we want to have the schedule.

        Returns:
            keys: The set of keys populated with schedule name, parameter name, qubits.
        """

        # schedule.channels may give the qubits in any order. This order matters. For example,
        # the parameter ('cr', 'amp', (2, 3)) is not the same as ('cr', 'amp', (3, 2)).
        # Furthermore, as we call subroutines the list of qubits involved might shrink. For
        # example, part of a cross-resonance schedule might involve.
        #
        # pulse.call(xp)
        # ...
        # pulse.play(GaussianSquare(...), ControlChannel(X))
        #
        # Here, the call instruction might, e.g., only involve qubit 2 while the play instruction
        # will apply to qubits (2, 3).

        qubit_set = set()
        for chan in schedule.channels:
            if isinstance(chan, DriveChannel):
                qubit_set.add(chan.index)

        qubits_ = tuple(qubit for qubit in qubits if qubit in qubit_set)

        for _, inst in schedule.instructions:

            if isinstance(inst, Play):
                for params in inst.pulse.parameters.values():
                    if isinstance(params, ParameterExpression):
                        for param in params.parameters:
                            keys.add(ParameterKey(schedule.name, param.name, qubits_))

            if isinstance(inst, (ShiftPhase, SetPhase)):
                if isinstance(inst.phase, ParameterExpression):
                    for param in inst.phase.parameters:
                        keys.add(ParameterKey(schedule.name, param.name, (inst.channel.index,)))

            if isinstance(inst, (ShiftFrequency, SetFrequency)):
                if isinstance(inst.frequency, ParameterExpression):
                    for param in inst.frequency.parameters:
                        keys.add(ParameterKey(schedule.name, param.name, (inst.channel.index,)))

            if isinstance(inst, Call):
                sched_ = inst.subroutine.assign_parameters(binding_dict, inplace=False)
                keys = Calibrations.get_parameter_keys(sched_, keys, binding_dict, qubits_)

        return keys

    def _get_circuit(
        self,
        schedule_name: str,
        qubits: Tuple,
        free_params: List[str] = None,
        group: Optional[str] = "default",
        schedule: Schedule = None,
    ) -> QuantumCircuit:
        """
        Queries a schedule by name for the given set of qubits. The parameters given
        under the list free_params are left unassigned. The queried schedule is then
        embedded in a gate with a calibration and returned as a quantum circuit.

        Args:
            schedule_name: The name of the schedule to retrieve.
            qubits: The qubits for which to generate the gate with the schedule in it.
            free_params: Names of the parameters that will remain unassigned.
            group: The calibration group from which to retrieve the calibrated values.
                If unspecified this defaults to 'default'.
            schedule: The schedule to add to the gate if the internally stored one is
                not used.

        Returns:
            A quantum circuit in which the parameter values have been assigned aside from
            those explicitly specified in free_params.
        """
        if schedule is None:
            schedule = self.get_schedule(schedule_name, qubits, free_params, group)

        gate = Gate(name=schedule_name, num_qubits=len(qubits), params=list(schedule.parameters))
        circ = QuantumCircuit(len(qubits), len(qubits))
        circ.append(gate, list(range(len(qubits))))
        circ.add_calibration(gate, qubits, schedule, params=schedule.parameters)

        return circ

    def schedules(self) -> List[Dict[str, Any]]:
        """
        Return the schedules in self in a list of dictionaries to help
        users manage their schedules.

        Returns:
            data: A list of dictionaries with all the schedules in it.
        """
        data = []
        for context, sched in self._schedules.items():
            data.append({"context": context, "schedule": sched, "parameters": sched.parameters})

        return data

    def parameters_table(
        self,
        parameters: List[str] = None,
        schedules: List[Union[Schedule, str]] = None,
        qubit_list: List[Tuple[int, ...]] = None,
    ) -> List[Dict[str, Any]]:
        """
        A convenience function to help users visualize the values of their parameter.

        Args:
            parameters: The parameter names that should be included in the returned
                table. If None is given then all names are included.
            schedules: The schedules to which to restrict the output.
            qubit_list: The qubits that should be included in the returned table.
                If None is given then all channels are returned.

        Returns:
            data: A dictionary of parameter values which can easily be converted to a
                data frame.
        """

        data = []

        # Convert inputs to lists of strings
        if parameters is not None:
            parameters = {prm.name if isinstance(prm, Parameter) else prm for prm in parameters}

        if schedules is not None:
            schedules = {sdl.name if isinstance(sdl, Schedule) else sdl for sdl in schedules}

        # Look for exact matches. Default values will be ignored.
        keys = set()
        for key in self._params.keys():
            if parameters and key.parameter not in parameters:
                continue
            if schedules and key.schedule not in schedules:
                continue
            if qubit_list and key.qubits not in qubit_list:
                continue

            keys.add(key)

        for key in keys:
            for value in self._params[key]:
                value_dict = dataclasses.asdict(value)
                value_dict["qubits"] = key.qubits
                value_dict["parameter"] = key.parameter
                value_dict["schedule"] = key.schedule

                data.append(value_dict)

        return data

    def to_csv(self):
        """
        Serializes the parameterized schedules and parameter values so
        that they can be stored in csv file.
        """

        # Write the parameter configuration.
        header_keys = ["parameter.name", "hash(parameter)", "schedule", "qubits"]
        body = []

        for parameter, keys in self.parameters.items():
            for key in keys:
                body.append(
                    {
                        "parameter.name": parameter[0].name,
                        "hash(parameter)": parameter[1],
                        "schedule": key.schedule,
                        "qubits": key.qubits,
                    }
                )

        with open("parameter_config.csv", "w", newline="") as output_file:
            dict_writer = csv.DictWriter(output_file, header_keys)
            dict_writer.writeheader()
            dict_writer.writerows(body)

        # Write the values of the parameters.
        values = self.parameters_table()
        if len(values) > 0:
            header_keys = values[0].keys()

            with open("parameter_values.csv", "w", newline="") as output_file:
                dict_writer = csv.DictWriter(output_file, header_keys)
                dict_writer.writeheader()
                dict_writer.writerows(values)

        # Serialize the schedules. For now we just print them.
        schedules = []
        header_keys = ["name", "qubits", "schedule"]
        for key, sched in self._schedules.items():
            schedules.append({"name": key.schedule, "qubits": key.qubits, "schedule": str(sched)})

        with open("schedules.csv", "w", newline="") as output_file:
            dict_writer = csv.DictWriter(output_file, header_keys)
            dict_writer.writeheader()
            dict_writer.writerows(schedules)

    @classmethod
    def from_csv(cls):
        """
        Retrieves the parameterized schedules and pulse parameters from an
        external DB.
        """
        raise NotImplementedError
