# This code is part of Qiskit.
#
# (C) Copyright IBM 2019-2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Calibration helper functions"""

from typing import Dict, List, Set, Tuple
import regex as re

from qiskit.circuit import ParameterExpression, Parameter
from qiskit.pulse import ScheduleBlock, Call

from qiskit_experiments.exceptions import CalibrationError
from qiskit_experiments.calibration_management.calibration_key_types import ParameterKey


def used_in_calls(schedule_name: str, schedules: List[ScheduleBlock]) -> Set[str]:
    """Find the schedules in the given list that call a given schedule by name.

    Args:
        schedule_name: The name of the callee to identify.
        schedules: A list of potential caller schedules to search.

    Returns:
        A set of schedule names that call the given schedule.
    """
    caller_names = set()

    for schedule in schedules:
        if _used_in_calls(schedule_name, schedule):
            caller_names.add(schedule.name)

    return caller_names


def _used_in_calls(schedule_name: str, schedule: ScheduleBlock) -> bool:
    """Recursively find if the schedule calls a schedule with name ``schedule_name``.

    Args:
        schedule_name: The name of the callee to identify.
        schedule: The schedule to parse.

    Returns:
        True if ``schedule``calls a ``ScheduleBlock`` with name ``schedule_name``.
    """
    blocks_have_schedule = False

    for block in schedule.blocks:
        if isinstance(block, Call):
            if block.subroutine.name == schedule_name:
                return True
            else:
                blocks_have_schedule = blocks_have_schedule or _used_in_calls(
                    schedule_name, block.subroutine
                )

        if isinstance(block, ScheduleBlock):
            blocks_have_schedule = blocks_have_schedule or _used_in_calls(schedule_name, block)

    return blocks_have_schedule


def validate_channels(schedule: ScheduleBlock) -> Set[Parameter]:
    """Validate amd get the parameters in the channels of the schedule.

    Note that channels implicitly defined in references will be ignored.

    Args:
        schedule: The schedule for which to get the parameters in the channels.

    Returns:
        The set of parameters explicitly defined in the schedule.

    Raises:
        CalibrationError: If a channel is parameterized by more than one parameter.
        CalibrationError: If the parameterized channel index is not formatted properly.
    """

    # The channel indices need to be parameterized following this regex.
    __channel_pattern__ = r"^ch\d[\.\d]*\${0,1}[\d]*$"

    param_indices = set()

    # Schedules with references do not explicitly have channels. This needs special handling.
    if schedule.is_referenced():
        for block in schedule.blocks:
            if isinstance(block, ScheduleBlock):
                param_indices.update(validate_channels(block))

        return param_indices

    for ch in schedule.channels:
        if isinstance(ch.index, ParameterExpression):
            if len(ch.index.parameters) != 1:
                raise CalibrationError(f"Channel {ch} can only have one parameter.")

            param_indices.add(ch.index)
            if re.compile(__channel_pattern__).match(ch.index.name) is None:
                raise CalibrationError(
                    f"Parameterized channel must correspond to {__channel_pattern__}"
                )

    return param_indices


def standardize_assign_params(
    assign_params: Dict,
    qubits: Tuple[int, ...],
    schedule_name: str,
) -> Tuple[Dict, Dict]:
    """Standardize the format of manually specified parameter assignments.

    This method is used in the ``get_schedule`` method of the :class:`.Calibrations` class.
    It standardizes the input of the ``assign_params`` variable and provides support
    for the edge case in which parameters in referenced schedules are assigned.

    Args:
        assign_params: The dictionary that specifies how to assign parameters.
        qubits: The qubits for which to get a schedule.
        schedule_name: The name of the schedule in which the parameters can be found.

    Returns:
        Two dictionaries: the first dict is for parameters that are in the schedule
        under consideration by ``get_schedule`` and that are not contained in any
        referenced schedule. This dict has :class:`.ParameterKey`s as keys and
        :class:`.ParameterValueType`s as values. The second dict enables the
        assignment of values to parameters in referenced schedules. Its keys are a tuple
        of referenced schedule name (a str) and the referenced qubits (a tuple).
        The values are a dictionary.

    Raises:
        CalibrationError: If the assign_params contains parameter assignments for
            schedule references and the key is not a tuple of reference name and
            corresponding qubits.
    """
    assign_params_ = dict()  # keys: ParameterKey, values: ParameterValueType
    ref_assign_params_ = dict()  # keys: Tuple[str, Tuple]: values: Dict[str, any]

    if assign_params:

        for param, value in assign_params.items():
            # This corresponds to a parameter assignment for a referenced schedule.
            if isinstance(value, dict):
                if not isinstance(param, tuple) and len(param) != 2:
                    raise CalibrationError(
                        "When assigning parameter values to referenced schedules the key "
                        "must be a tuple of length 2."
                    )

                ref_sched_name, ref_qubits = param[0], param[1]
                if not isinstance(ref_sched_name, str) and not isinstance(ref_qubits, tuple):
                    raise CalibrationError(
                        "Parameter reference assignment should be tuple of reference name "
                        f"and qubits. Found {param}"
                    )

                # Change the values
                ref_assign_params_[param] = dict()
                for ref_param_name, ref_param_value in value.items():
                    ref_assign_params_[param][ref_param_name] = ref_param_value

            # Parameters that are not contained in a referenced schedule.
            elif isinstance(param, str):
                assign_params_[ParameterKey(param, qubits, schedule_name)] = value
            else:
                assign_params_[ParameterKey(*param)] = value

    return assign_params_, ref_assign_params_
