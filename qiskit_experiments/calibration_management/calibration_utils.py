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

from typing import List, Set
from qiskit.pulse import ScheduleBlock, Call


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
