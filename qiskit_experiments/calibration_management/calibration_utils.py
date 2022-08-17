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

from typing import List, Optional, Set, Tuple
import regex as re

from qiskit.circuit import ParameterExpression, Parameter
from qiskit.pulse import ScheduleBlock, Call

from qiskit_experiments.exceptions import CalibrationError


def used_in_references(schedule_name: str, schedules: List[ScheduleBlock]) -> Set[str]:
    """Find the schedules in the given list that reference a given schedule by name.

    Args:
        schedule_name: The name of the referencer to identify.
        schedules: A list of potential referencer schedules to search.

    Returns:
        A set of schedule names that call the given schedule.
    """
    caller_names = set()

    for schedule in schedules:
        if schedule_name in set(ref[0] for ref in schedule.references):
            caller_names.add(schedule.name)

    return caller_names


def _used_in_calls(schedule_name: str, schedule: ScheduleBlock) -> bool:
    """Recursively find if the schedule calls a schedule with name ``schedule_name``.

    TODO Remove?

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


def reference_info(
    reference: Tuple[str, ...],
    qubits: Optional[Tuple[int, ...]] = None,
) -> Tuple[str, Tuple[int, ...]]:
    """Extract reference information from the reference tuple.

    Args:
        reference: The reference of a Reference instruction in a ScheduleBlock.
        qubits: Optional argument to reorder the references.

    Returns:
        A string corresponding to the name of the referenced schedule and the qubits that
        this schedule applies to.

    Raises:
        CalibrationError: If ``reference`` is not a tuple.
        CalibrationError: If ``reference`` is not a tuple of reference name and the qubits that
            that the schedule applies to.
    """
    if not isinstance(reference, tuple):
        raise CalibrationError(f"A schedule reference must be a tuple. Found {reference}.")

    ref_schedule_name, ref_qubits = reference[0], reference[1:]

    # Convert a single-qubit reference to a tuple type.
    if isinstance(ref_qubits, str):
        ref_qubits = (ref_qubits,)

    if not isinstance(ref_schedule_name, str) and not isinstance(ref_qubits, tuple):
        raise CalibrationError(
            f"A schedule reference is a name and qubits tuple. Found {reference}"
        )

    ref_qubits = tuple(int(qubit[1:]) for qubit in ref_qubits)

    # get the qubit indices for which we are getting the schedules
    if qubits is not None and qubits != ():
        ref_qubits = tuple(qubits[idx] for idx in ref_qubits)

    return ref_schedule_name, ref_qubits
