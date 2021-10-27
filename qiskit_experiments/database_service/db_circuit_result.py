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

from typing import Sequence, Dict, Any, Optional


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
    counts: Optional[Sequence[Dict[str, float]]] = None

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
