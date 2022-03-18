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

from qiskit.pulse import ScheduleBlock


class CalUtils:
    """A collection of utility functions for for calibration management."""

    @staticmethod
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
                all_blocks_equal.append(CalUtils.compare_schedule_blocks(block1, block2))
            else:
                all_blocks_equal.append(str(block1) == str(block2))

        return all(all_blocks_equal)
