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

"""Class to store and manage the results of calibration experiments."""

import os
from collections import namedtuple, defaultdict
from datetime import datetime
from typing import Any, Dict, Set, Tuple, Union, List, Optional
import csv
import dataclasses
import regex as re

from qiskit.pulse import (
    ScheduleBlock,
    DriveChannel,
    ControlChannel,
    MeasureChannel,
    Call,
    Instruction,
    AcquireChannel,
    RegisterSlot,
    MemorySlot,
)
from qiskit.pulse.channels import PulseChannel
from qiskit.circuit import Parameter, ParameterExpression
from qiskit_experiments.calibration.exceptions import CalibrationError
from qiskit_experiments.calibration.parameter_value import ParameterValue

ParameterKey = namedtuple("ParameterKey", ["parameter", "qubits", "schedule"])
ScheduleKey = namedtuple("ScheduleKey", ["schedule", "qubits"])


class Calibrations:
    """
    A class to manage schedules with calibrated parameter values. Schedules are
    intended to be fully parameterized, including the index of the channels. See
    the module-level documentation for extra details.
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

        # Mapping between qubits and their control channels.
        self._controls_config = control_config if control_config else {}

        # Store the reverse mapping between control channels and qubits for ease of look-up.
        self._controls_config_r = {}
        for qubits, channels in self._controls_config.items():
            for channel in channels:
                self._controls_config_r[channel] = qubits

        # Dict of the form: (schedule.name, parameter.name, qubits): Parameter
        self._parameter_map = {}

        # Reverse mapping of _parameter_map
        self._parameter_map_r = defaultdict(set)

        # Default dict of the form: (schedule.name, parameter.name, qubits): [ParameterValue, ...]
        self._params = defaultdict(list)

        self._schedules = {}

        # A variable to store all parameter hashes encountered and present them as ordered
        # indices to the user.
        self._hash_to_counter_map = {}
        self._parameter_counter = 0

    def add_schedule(self, schedule: ScheduleBlock, qubits: Union[int, Tuple[int, ...]] = None):
        """
        Add a schedule and register its parameters.

        Schedules that use Call instructions must register the called schedules separately.

        Args:
            schedule: The schedule to add.
            qubits: The qubits for which to add the schedules. If None or an empty tuple is
                given then this schedule is the default schedule for all qubits.

        Raises:
            CalibrationError:
                - If schedule is not a ScheduleBlock.
                - If the parameterized channel index is not formatted properly.
                - If several parameters in the same schedule have the same name.
                - If a channel is parameterized by more than one parameter.
                - If the schedule name starts with the prefix of ScheduleBlock.
        """
        qubits = self._to_tuple(qubits)

        if not isinstance(schedule, ScheduleBlock):
            raise CalibrationError(f"{schedule.name} is not a ScheduleBlock.")

        # check that channels, if parameterized, have the proper name format.
        if schedule.name.startswith(ScheduleBlock.prefix):
            raise CalibrationError(
                f"A registered schedule name cannot start with {ScheduleBlock.prefix}, "
                f"received {schedule.name}. "
                f"Use a name that does not start with {ScheduleBlock.prefix}."
            )

        param_indices = set()
        for ch in schedule.channels:
            if isinstance(ch.index, Parameter):
                if len(ch.index.parameters) != 1:
                    raise CalibrationError(f"Channel {ch} can only have one parameter.")

                param_indices.add(ch.index)
                if re.compile(self.__channel_pattern__).match(ch.index.name) is None:
                    raise CalibrationError(
                        f"Parameterized channel must correspond to {self.__channel_pattern__}"
                    )

        # Clean the parameter to schedule mapping. This is needed if we overwrite a schedule.
        self._clean_parameter_map(schedule.name, qubits)

        # Add the schedule.
        self._schedules[ScheduleKey(schedule.name, qubits)] = schedule

        # Register parameters that are not indices.
        # Do not register parameters that are in call instructions.
        params_to_register = set()
        for inst in self._exclude_calls(schedule, []):
            for param in inst.parameters:
                if param not in param_indices:
                    params_to_register.add(param)

        if len(params_to_register) != len(set(param.name for param in params_to_register)):
            raise CalibrationError(f"Parameter names in {schedule.name} must be unique.")

        for param in params_to_register:
            self._register_parameter(param, qubits, schedule)

    def _exclude_calls(
        self, schedule: ScheduleBlock, instructions: List[Instruction]
    ) -> List[Instruction]:
        """
        Recursive function to get all non-Call instructions. This will flatten all blocks
        in a ScheduleBlock and return the instructions of the ScheduleBlock leaving out
        any Call instructions.

        Args:
            schedule: A ScheduleBlock from which to extract the instructions.
            instructions: The list of instructions that is recursively populated.

        Returns:
            The list of instructions to which all non-Call instructions have been added.
        """
        for block in schedule.blocks:
            if isinstance(block, ScheduleBlock):
                instructions = self._exclude_calls(block, instructions)
            else:
                if not isinstance(block, Call):
                    instructions.append(block)

        return instructions

    def remove_schedule(self, schedule: ScheduleBlock, qubits: Union[int, Tuple[int, ...]] = None):
        """
        Allows users to remove a schedule from the calibrations. The history of the parameters
        will remain in the calibrations.

        Args:
            schedule: The schedule to remove.
            qubits: The qubits for which to remove the schedules. If None is given then this
                schedule is the default schedule for all qubits.
        """
        qubits = self._to_tuple(qubits)

        if ScheduleKey(schedule.name, qubits) in self._schedules:
            del self._schedules[ScheduleKey(schedule.name, qubits)]

        # Clean the parameter to schedule mapping.
        self._clean_parameter_map(schedule.name, qubits)

    def _clean_parameter_map(self, schedule_name: str, qubits: Tuple[int, ...]):
        """Clean the parameter to schedule mapping for the given schedule, parameter and qubits.

        Args:
            schedule_name: The name of the schedule.
            qubits: The qubits to which this schedule applies.

        """
        keys_to_remove = []  # of the form (schedule.name, parameter.name, qubits)
        for key in self._parameter_map:
            if key.schedule == schedule_name and key.qubits == qubits:
                keys_to_remove.append(key)

        for key in keys_to_remove:
            del self._parameter_map[key]

            # Key set is a set of tuples (schedule.name, parameter.name, qubits)
            for param, key_set in self._parameter_map_r.items():
                if key in key_set:
                    key_set.remove(key)

        # Remove entries that do not point to at least one (schedule.name, parameter.name, qubits)
        keys_to_delete = []
        for param, key_set in self._parameter_map_r.items():
            if not key_set:
                keys_to_delete.append(param)

        for key in keys_to_delete:
            del self._parameter_map_r[key]

    def _register_parameter(
        self,
        parameter: Parameter,
        qubits: Tuple[int, ...],
        schedule: ScheduleBlock = None,
    ):
        """
        Registers a parameter for the given schedule. This allows self to determine the
        parameter instance that corresponds to the given schedule name, parameter name
        and qubits.

        Args:
            parameter: The parameter to register.
            qubits: The qubits for which to register the parameter.
            schedule: The schedule to which this parameter belongs. The schedule can
                be None which allows the calibration to accommodate, e.g. qubit frequencies.
        """
        if parameter not in self._hash_to_counter_map:
            self._hash_to_counter_map[parameter] = self._parameter_counter
            self._parameter_counter += 1

        sched_name = schedule.name if schedule else None
        key = ParameterKey(parameter.name, qubits, sched_name)
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
        return self._parameter_map_r

    def calibration_parameter(
        self,
        parameter_name: str,
        qubits: Union[int, Tuple[int, ...]] = None,
        schedule_name: str = None,
    ) -> Parameter:
        """
        Returns a Parameter object given the triplet parameter_name, qubits and schedule_name
        which uniquely determine the context of a parameter.

        Args:
            parameter_name: Name of the parameter to get.
            qubits: The qubits to which this parameter belongs. If qubits is None then
                the default scope is assumed and the key will be an empty tuple.
            schedule_name: The name of the schedule to which this parameter belongs. A
                parameter may not belong to a schedule in which case None is accepted.

        Returns:
             calibration parameter: The parameter that corresponds to the given arguments.

        Raises:
            CalibrationError: If the desired parameter is not found.
        """
        qubits = self._to_tuple(qubits)

        # 1) Check for qubit specific parameters.
        if ParameterKey(parameter_name, qubits, schedule_name) in self._parameter_map:
            return self._parameter_map[ParameterKey(parameter_name, qubits, schedule_name)]

        # 2) Check for default parameters.
        elif ParameterKey(parameter_name, (), schedule_name) in self._parameter_map:
            return self._parameter_map[ParameterKey(parameter_name, (), schedule_name)]
        else:
            raise CalibrationError(
                f"No parameter for {parameter_name} and schedule {schedule_name} "
                f"and qubits {qubits}. No default value exists."
            )

    def add_parameter_value(
        self,
        value: Union[int, float, complex, ParameterValue],
        param: Union[Parameter, str],
        qubits: Union[int, Tuple[int, ...]] = None,
        schedule: Union[ScheduleBlock, str] = None,
    ):
        """
        Add a parameter value to the stored parameters.

        This parameter value may be applied to several channels, for instance, all
        DRAG pulses may have the same standard deviation.

        Args:
            value: The value of the parameter to add. If an int, float, or complex is given
                then the timestamp of the parameter value will automatically be generated
                and set to the current time.
            param: The parameter or its name for which to add the measured value.
            qubits: The qubits to which this parameter applies.
            schedule: The schedule or its name for which to add the measured parameter value.

        Raises:
            CalibrationError: If the schedule name is given but no schedule with that name
                exists.
        """
        qubits = self._to_tuple(qubits)

        if isinstance(value, (int, float, complex)):
            value = ParameterValue(value, datetime.now())

        param_name = param.name if isinstance(param, Parameter) else param
        sched_name = schedule.name if isinstance(schedule, ScheduleBlock) else schedule

        registered_schedules = set(key.schedule for key in self._schedules)

        if sched_name and sched_name not in registered_schedules:
            raise CalibrationError(f"Schedule named {sched_name} was never registered.")

        self._params[ParameterKey(param_name, qubits, sched_name)].append(value)

    def _get_channel_index(self, qubits: Tuple[int, ...], chan: PulseChannel) -> int:
        """
        Get the index of the parameterized channel based on the given qubits
        and the name of the parameter in the channel index. The name of this
        parameter for control channels must be written as chqubit_index1.qubit_index2...
        followed by an optional $index.
        For example, the following parameter names are valid: 'ch1', 'ch1.0', 'ch30.12',
        and 'ch1.0$1'.

        Args:
            qubits: The qubits for which we want to obtain the channel index.
            chan: The channel with a parameterized name.

        Returns:
            index: The index of the channel. For example, if qubits=(10, 32) and
                chan is a control channel with parameterized index name 'ch1.0'
                the method returns the control channel corresponding to
                qubits (qubits[1], qubits[0]) which is here the control channel of
                qubits (32, 10).

        Raises:
            CalibrationError:
                - If the number of qubits is incorrect.
                - If the number of inferred ControlChannels is not correct.
                - If ch is not a DriveChannel, MeasureChannel, or ControlChannel.
        """
        if isinstance(chan.index, Parameter):
            if isinstance(
                chan, (DriveChannel, MeasureChannel, AcquireChannel, RegisterSlot, MemorySlot)
            ):
                index = int(chan.index.name[2:].split("$")[0])

                if len(qubits) <= index:
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

                if len(chs_) <= control_index:
                    raise CalibrationError(
                        f"Control channel index {control_index} not found for qubits {qubits}."
                    )

                return chs_[control_index].index

            raise CalibrationError(
                f"{chan} must be a sub-type of {PulseChannel} or an {AcquireChannel}, "
                f"{RegisterSlot}, or a {MemorySlot}."
            )

        return chan.index

    def get_parameter_value(
        self,
        param: Union[Parameter, str],
        qubits: Union[int, Tuple[int, ...]],
        schedule: Union[ScheduleBlock, str, None] = None,
        valid_only: bool = True,
        group: str = "default",
        cutoff_date: datetime = None,
    ) -> Union[int, float, complex]:
        """
        Retrieves the value of a parameter.

        Parameters may be linked. get_parameter_value does the following steps:
        1) Retrieve the parameter object corresponding to (param, qubits, schedule)
        2) The values of this parameter may be stored under another schedule since
           schedules can share parameters. To deal with this, a list of candidate keys
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
            group: The calibration group from which to draw the parameters.
                If not specified this defaults to the 'default' group.
            cutoff_date: Retrieve the most recent parameter up until the cutoff date. Parameters
                generated after the cutoff date will be ignored. If the cutoff_date is None then
                all parameters are considered. This allows users to discard more recent values that
                may be erroneous.

        Returns:
            value: The value of the parameter.

        Raises:
            CalibrationError:
                - If there is no parameter value for the given parameter name and pulse channel.
        """
        qubits = self._to_tuple(qubits)

        # 1) Identify the parameter object.
        param_name = param.name if isinstance(param, Parameter) else param
        sched_name = schedule.name if isinstance(schedule, ScheduleBlock) else schedule

        param = self.calibration_parameter(param_name, qubits, sched_name)

        # 2) Get a list of candidate keys restricted to the qubits of interest.
        candidate_keys = []
        for key in self._parameter_map_r[param]:
            candidate_keys.append(ParameterKey(key.parameter, qubits, key.schedule))

        # 3) Loop though the candidate keys to candidate values
        candidates = []
        for key in candidate_keys:
            if key in self._params:
                candidates += self._params[key]

        # If no candidate parameter values were found look for default parameters
        # i.e. parameters that do not specify a qubit.
        if len(candidates) == 0:
            for key in candidate_keys:
                if ParameterKey(key.parameter, (), key.schedule) in self._params:
                    candidates += self._params[ParameterKey(key.parameter, (), key.schedule)]

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

            if sched_name:
                msg += f"in schedule {sched_name} "

            if cutoff_date:
                msg += f"with cutoff date: {cutoff_date}"

            raise CalibrationError(msg)

        # 5) Return the most recent parameter.
        return max(candidates, key=lambda x: x.date_time).value

    def get_schedule(
        self,
        name: str,
        qubits: Union[int, Tuple[int, ...]],
        free_params: List[Union[str, ParameterKey]] = None,
        group: Optional[str] = "default",
        cutoff_date: datetime = None,
    ) -> ScheduleBlock:
        """
        Get the schedule with the non-free parameters assigned to their values.

        Args:
            name: The name of the schedule to get.
            qubits: The qubits for which to get the schedule.
            free_params: The parameters that should remain unassigned. Each free parameter is
                specified by a ParameterKey a named tuple of the form (parameter name, qubits,
                schedule name). Each entry in free_params can also be a string corresponding
                to the name of the parameter. In this case, the schedule name and qubits of the
                corresponding ParameterKey will be the name and qubits given as arguments to
                get_schedule.
            group: The calibration group from which to draw the
                parameters. If not specified this defaults to the 'default' group.
            cutoff_date: Retrieve the most recent parameter up until the cutoff date. Parameters
                generated after the cutoff date will be ignored. If the cutoff_date is None then
                all parameters are considered. This allows users to discard more recent values that
                may be erroneous.

        Returns:
            schedule: A copy of the template schedule with all parameters assigned
                except for those specified by free_params.

        Raises:
            CalibrationError:
                - If the name of the schedule is not known.
                - If a parameter could not be found.
        """
        qubits = self._to_tuple(qubits)

        if free_params:
            free_params_ = []
            for free_param in free_params:
                if isinstance(free_param, str):
                    free_params_.append(ParameterKey(free_param, qubits, name))
                else:
                    free_params_.append(free_param)

            free_params = free_params_
        else:
            free_params = []

        if (name, qubits) in self._schedules:
            schedule = self._schedules[ScheduleKey(name, qubits)]
        elif (name, ()) in self._schedules:
            schedule = self._schedules[ScheduleKey(name, ())]
        else:
            raise CalibrationError(f"Schedule {name} is not defined for qubits {qubits}.")

        # Retrieve the channel indices based on the qubits and bind them.
        binding_dict = {}
        for ch in schedule.channels:
            if ch.is_parameterized():
                binding_dict[ch.index] = self._get_channel_index(qubits, ch)

        # Binding the channel indices makes it easier to deal with parameters later on
        schedule = schedule.assign_parameters(binding_dict, inplace=False)

        assigned_schedule = self._assign(schedule, qubits, free_params, group, cutoff_date)

        if len(assigned_schedule.parameters) != len(free_params):
            raise CalibrationError(
                f"The number of free parameters {len(assigned_schedule.parameters)} in "
                f"the assigned schedule differs from the requested number of free "
                f"parameters {len(free_params)}."
            )

        return assigned_schedule

    def _assign(
        self,
        schedule: ScheduleBlock,
        qubits: Tuple[int, ...],
        free_params: List[ParameterKey] = None,
        group: Optional[str] = "default",
        cutoff_date: datetime = None,
    ) -> ScheduleBlock:
        """
        Recursive function to extract and assign parameters from a schedule. The
        recursive behaviour is needed to handle Call instructions as the name of
        the called instruction defines the scope of the parameter. Each time a Call
        is found _assign recurses on the channel-assigned subroutine of the Call
        instruction and the qubits that are in said subroutine. This requires a
        careful extraction of the qubits from the subroutine and in the appropriate
        order. Next, the parameters are identified and assigned. This is needed to
        handle situations where the same parameterized schedule is called but on
        different channels. For example,

        .. code-block:: python

            ch0 = Parameter("ch0")
            ch1 = Parameter("ch1")

            with pulse.build(name="xp") as xp:
                pulse.play(Gaussian(duration, amp, sigma), DriveChannel(ch0))

            with pulse.build(name="xt_xp") as xt:
                pulse.call(xp)
                pulse.call(xp, value_dict={ch0: ch1})

        Here, we define the xp schedule for all qubits as a Gaussian. Next, we define a
        schedule where both xp schedules are called simultaneously on different channels.

        Args:
            schedule: The schedule with assigned channel indices for which we wish to
                assign values to non-channel parameters.
            qubits: The qubits for which to get the schedule.
            free_params: The parameters that are to be left free. See get_schedules for details.
            group: The calibration group of the parameters.
            cutoff_date: Retrieve the most recent parameter up until the cutoff date. Parameters
                generated after the cutoff date will be ignored. If the cutoff_date is None then
                all parameters are considered. This allows users to discard more recent values that
                may be erroneous.

        Returns:
            ret_schedule: The schedule with assigned parameters.

        Raises:
            CalibrationError:
                - If a channel has not been assigned.
                - If a parameter that is needed does not have a value.
        """
        # 1) Restrict the given qubits to those in the given schedule.
        qubit_set = set()
        for chan in schedule.channels:
            if isinstance(chan.index, ParameterExpression):
                raise (
                    CalibrationError(
                        f"All parametric channels must be assigned before searching for "
                        f"non-channel parameters. {chan} is parametric."
                    )
                )
            if isinstance(chan, (DriveChannel, MeasureChannel)):
                qubit_set.add(chan.index)

            if isinstance(chan, ControlChannel):
                for qubit in self._controls_config_r[chan]:
                    qubit_set.add(qubit)

        qubits_ = tuple(qubit for qubit in qubits if qubit in qubit_set)

        # 2) Recursively assign the parameters in the called instructions.
        ret_schedule = ScheduleBlock(
            alignment_context=schedule.alignment_context,
            name=schedule.name,
            metadata=schedule.metadata,
        )

        for inst in schedule.blocks:
            if isinstance(inst, Call):
                inst = self._assign(
                    inst.assigned_subroutine(), qubits_, free_params, group, cutoff_date
                )
            elif isinstance(inst, ScheduleBlock):
                inst = self._assign(inst, qubits_, free_params, group, cutoff_date)

            ret_schedule.append(inst, inplace=True)

        # 3) Get the parameter keys of the remaining instructions. At this point in
        #    _assign all parameters in Call instructions that are supposed to be
        #     assigned have been assigned.
        keys = set()

        if ret_schedule.name in set(key.schedule for key in self._parameter_map):
            for param in ret_schedule.parameters:
                keys.add(ParameterKey(param.name, qubits_, ret_schedule.name))

        # 4) Build the parameter binding dictionary.
        free_params = free_params if free_params else []

        binding_dict = {}
        for key in keys:
            if key not in free_params:
                # Get the parameter object. Since we are dealing with a schedule the name of
                # the schedule is always defined. However, the parameter may be a default
                # parameter for all qubits, i.e. qubits may be an empty tuple.
                if key in self._parameter_map:
                    param = self._parameter_map[key]
                elif ParameterKey(key.parameter, (), key.schedule) in self._parameter_map:
                    param = self._parameter_map[ParameterKey(key.parameter, (), key.schedule)]
                else:
                    raise CalibrationError(
                        f"Bad calibrations {key} is not present and has no default value."
                    )

                if param not in binding_dict:
                    binding_dict[param] = self.get_parameter_value(
                        key.parameter,
                        key.qubits,
                        key.schedule,
                        group=group,
                        cutoff_date=cutoff_date,
                    )

        return ret_schedule.assign_parameters(binding_dict, inplace=False)

    def schedules(self) -> List[Dict[str, Any]]:
        """
        Return the managed schedules in a list of dictionaries to help
        users manage their schedules.

        Returns:
            data: A list of dictionaries with all the schedules in it. The key-value pairs are
                - 'qubits': the qubits to which this schedule applies. This may be () if the
                    schedule is the default for all qubits.
                - 'schedule': The schedule.
                - 'parameters': The parameters in the schedule exposed for convenience.
                This list of dictionaries can easily be converted to a data frame.
        """
        data = []
        for key, sched in self._schedules.items():
            data.append({"qubits": key.qubits, "schedule": sched, "parameters": sched.parameters})

        return data

    def parameters_table(
        self,
        parameters: List[str] = None,
        qubit_list: List[Tuple[int, ...]] = None,
        schedules: List[Union[ScheduleBlock, str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        A convenience function to help users visualize the values of their parameter.

        Args:
            parameters: The parameter names that should be included in the returned
                table. If None is given then all names are included.
            qubit_list: The qubits that should be included in the returned table.
                If None is given then all channels are returned.
            schedules: The schedules to which to restrict the output.

        Returns:
            data: A list of dictionaries with parameter values and metadata which can
                easily be converted to a data frame.
        """
        if qubit_list:
            qubit_list = [self._to_tuple(qubits) for qubits in qubit_list]

        data = []

        # Convert inputs to lists of strings
        if schedules is not None:
            schedules = {sdl.name if isinstance(sdl, ScheduleBlock) else sdl for sdl in schedules}

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

    def save(self, file_type: str = "csv", folder: str = None, overwrite: bool = False):
        """
        Saves the parameterized schedules and parameter values so
        that they can be stored in csv files. This method will create three files:
        - parameter_config.csv: This file stores a table of parameters which indicates
          which parameters appear in which schedules.
        - parameter_values.csv: This file stores the values of the calibrated parameters.
        - schedules.csv: This file stores the parameterized schedules.

        Args:
            file_type: The type of file to which to save. By default this is a csv.
                Other file types may be supported in the future.
            folder: The folder in which to save the calibrations.
            overwrite: If the files already exist then they will not be overwritten
                unless overwrite is set to True.

        Raises:
            CalibrationError: if the files exist and overwrite is not set to True.
        """
        cwd = os.getcwd()
        if folder:
            os.chdir(folder)

        if os.path.isfile("parameter_config.csv") and not overwrite:
            raise CalibrationError("parameter_config.csv already exists. Set overwrite to True.")

        if os.path.isfile("parameter_values.csv") and not overwrite:
            raise CalibrationError("parameter_values.csv already exists. Set overwrite to True.")

        if os.path.isfile("parameter_values.csv") and not overwrite:
            raise CalibrationError("schedules.csv already exists. Set overwrite to True.")

        # Write the parameter configuration.
        header_keys = ["parameter.name", "parameter unique id", "schedule", "qubits"]
        body = []

        for parameter, keys in self.parameters.items():
            for key in keys:
                body.append(
                    {
                        "parameter.name": parameter.name,
                        "parameter unique id": self._hash_to_counter_map[parameter],
                        "schedule": key.schedule,
                        "qubits": key.qubits,
                    }
                )

        if file_type == "csv":
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
                schedules.append(
                    {"name": key.schedule, "qubits": key.qubits, "schedule": str(sched)}
                )

            with open("schedules.csv", "w", newline="") as output_file:
                dict_writer = csv.DictWriter(output_file, header_keys)
                dict_writer.writeheader()
                dict_writer.writerows(schedules)

        else:
            raise CalibrationError(f"Saving to .{file_type} is not yet supported.")

        os.chdir(cwd)

    def load_parameter_values(self, file_name: str = "parameter_values.csv"):
        """
        Load parameter values from a given file into self._params.

        Args:
            file_name: The name of the file that stores the parameters. Will default to
                parameter_values.csv.
        """
        with open(file_name) as fp:
            reader = csv.DictReader(fp, delimiter=",", quotechar='"')

            for row in reader:
                param_val = ParameterValue(
                    row["value"], row["date_time"], row["valid"], row["exp_id"], row["group"]
                )
                key = ParameterKey(row["parameter"], self._to_tuple(row["qubits"]), row["schedule"])
                self.add_parameter_value(param_val, *key)

    @classmethod
    def load(cls, files: List[str]) -> "Calibrations":
        """
        Retrieves the parameterized schedules and pulse parameters from the
        given location.
        """
        raise CalibrationError("Full calibration loading is not implemented yet.")

    @staticmethod
    def _to_tuple(qubits: Union[str, int, Tuple[int, ...]]) -> Tuple[int, ...]:
        """
        Ensure that qubits is a tuple of ints.

        Args:
            qubits: An int, a tuple of ints, or a string representing a tuple of ints.

        Returns:
            qubits: A tuple of ints.

        Raises:
            CalibrationError: If the given input does not conform to an int or
                tuple of ints.
        """
        if not qubits:
            return tuple()

        if isinstance(qubits, str):
            try:
                return tuple(int(qubit) for qubit in qubits.strip("( )").split(",") if qubit != "")
            except ValueError:
                pass

        if isinstance(qubits, int):
            return (qubits,)

        if isinstance(qubits, tuple):
            if all(isinstance(n, int) for n in qubits):
                return qubits

        raise CalibrationError(
            f"{qubits} must be int, tuple of ints, or str  that can be parsed"
            f"to a tuple if ints. Received {qubits}."
        )
