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


class CalUtils:
    """A collection of utility functions for for calibration management."""

    @staticmethod
    def used_in_calls(schedule_name: str, schedules: List[ScheduleBlock]) -> Set[str]:
        """Find the schedules in the given list that call a given schedule by name.

        Args:
            schedule_name: The name of the callee to identify.
            schedules: A list of schedule over which to search.

        Returns:
            A set of schedule names that call the given schedule.
        """
        sub_routines = set()

        for schedule in schedules:
            if CalUtils._used_in_calls(schedule_name, schedule):
                sub_routines.add(schedule.name)

        return sub_routines

    @staticmethod
    def _used_in_calls(schedule_name: str, schedule: ScheduleBlock):
        """Recursively find if the schedule calls a schedule with name ``schedule_name``.

        Args:
            schedule_name: The name of the callee to identify.
            schedule: The schedule to parse.
        """
        blocks_have_schedule = set()

        for block in schedule.blocks:
            if isinstance(block, Call):
                if block.subroutine.name == schedule_name:
                    blocks_have_schedule.add(True)
                else:
                    blocks_have_schedule.add(
                        CalUtils._used_in_calls(schedule_name, block.subroutine)
                    )

            if isinstance(block, ScheduleBlock):
                blocks_have_schedule.add(CalUtils._used_in_calls(schedule_name, block))

        return any(blocks_have_schedule)
