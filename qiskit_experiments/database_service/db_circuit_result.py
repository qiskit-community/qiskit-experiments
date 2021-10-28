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

"""DB class for experiment result of a single circuit."""

import dataclasses
import numpy as np

from qiskit.qobj.utils import MeasLevel, MeasReturnType
from typing import Sequence, Dict, Any, Optional, Union


@dataclasses.dataclass(frozen=True)
class CircuitResultData:
    """Experimental result from a single circuit execution."""

    # job id
    job_id: Optional[str] = None

    # circuit index in the job
    index: Optional[int] = None

    # number of shots
    shots: Optional[int] = None

    # measurement level (0: Raw, 1: Kerneled, 2: Discriminated)
    meas_level: Optional[int] = None

    # return format (avg: averaged data, single: sequence of single shot data)
    meas_return: Optional[str] = None

    # number of memory slots
    memory_slots: Optional[int] = None

    # number of quantum registers
    qreg_sizes: Optional[int] = None

    # number of classical registers
    creg_sizes: Optional[int] = None

    # formatted count data
    counts: Optional[Dict[str, float]] = None

    # formatted memory data, shape depends on execution condition
    # ============  =============  =====
    # `meas_level`  `meas_return`  shape
    # ============  =============  =====
    # 0             `single`       np.ndarray[shots, memory_slots, memory_slot_size]
    # 0             `avg`          np.ndarray[memory_slots, memory_slot_size]
    # 1             `single`       np.ndarray[shots, memory_slots]
    # 1             `avg`          np.ndarray[memory_slots]
    # 2             `memory=True`  list
    # ============  =============  =====
    memory: Optional[Sequence[Any]] = None

    # metadata
    metadata: Optional[Dict[str, Any]] = None

    def memory_slot_data(self, index: int) -> Union[None, complex, np.ndarray]:
        """Get memory slot value of target index.

        Args:
            index: Target memory slot index.

        Returns:
            Data stored in the target slot.

        Raises:
            IndexError: When index is out of range.
            ValueError: When classified data is stored, or empty memory slot.
        """
        if self.memory is None:
            raise ValueError("No memory slot exist in this result data.")

        if index > self.memory_slots:
            raise IndexError(
                f"Index {index} doesn't exist. This data has only {self.memory_slots} slots."
            )

        # level 0 data
        if self.meas_level == MeasLevel.RAW:
            if self.meas_return == MeasReturnType.SINGLE:
                return np.asarray(self.memory[:, index, :], dtype=complex)
            return np.asarray(self.memory[index, :], dtype=complex)

        # level 1 data
        if self.meas_level == MeasLevel.KERNELED:
            if self.meas_return == MeasReturnType.SINGLE:
                return np.asarray(self.memory[:, index], dtype=complex)
            return complex(self.memory[index])

        # level 2 data
        if self.meas_level == MeasLevel.CLASSIFIED:
            raise ValueError(
                "Memory slot for classified data doesn't return slot-wise data stream."
            )

        return None
