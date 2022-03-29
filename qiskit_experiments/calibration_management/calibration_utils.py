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

from typing import List, Optional, Set
from qiskit.pulse import ScheduleBlock, Call


def compare_schedule_blocks(schedule1: ScheduleBlock, schedule2: ScheduleBlock) -> bool:
    """Recursively compare schedule blocks in a parameter value insensitive fashion.

    This is needed because the alignment contexts of the pulse builder creates
    ScheduleBlock instances with :code:`f"block{itertools.count()}"` names making
    it impossible to compare two schedule blocks via equality. Furthermore, for calibrations
    only the name of the parameters (as opposed to the instance) is relevant.
    """
    all_blocks_equal = []
    for idx, block1 in enumerate(schedule1.blocks):
        block2 = schedule2.blocks[idx]
        if isinstance(block1, ScheduleBlock) and isinstance(block2, ScheduleBlock):
            all_blocks_equal.append(compare_schedule_blocks(block1, block2))
        else:
            all_blocks_equal.append(str(block1) == str(block2))

    return all(all_blocks_equal)


def get_called_subroutines(schedule: ScheduleBlock) -> List[ScheduleBlock]:
    """Returns the list of subroutines that the given schedule calls.

    Args:
        schedule: A schedule to parse and find the called subroutines.

    Returns:
        A list of schedules called by ``schedule``.
    """
    subroutines = []
    _get_called_subroutines(schedule, subroutines)
    return subroutines


def _get_called_subroutines(schedule: ScheduleBlock, subroutines: List[ScheduleBlock]):
    """Helper method to recursively find called subroutines."""
    for block in schedule.blocks:
        if isinstance(block, ScheduleBlock):
            _get_called_subroutines(block, subroutines)

        if isinstance(block, Call):
            subroutines.append(block.subroutine)

    return subroutines


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
