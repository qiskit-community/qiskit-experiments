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

"""
A set of utility functions.
"""

from typing import Union

import numpy as np
from uncertainties.core import UFloat


def check_if_nominal_significant(val: Union[float, UFloat], fraction: float = 1.0) -> bool:
    """Check if the nominal part of the given value is larger than the standard error.

    Args:
        val: Input value to evaluate.
        fraction: Valid fraction of the nominal part to its standard error.
            This function returns ``False`` if the nominal part is
            smaller than the error by this fraction.

    Returns:
        ``True`` if the nominal part is significant.
    """
    if isinstance(val, float):
        return True

    if np.isnan(val.std_dev) or val.std_dev < fraction * val.nominal_value:
        return True

    return False
