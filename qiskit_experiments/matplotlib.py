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
Optional matplotlib helper functions
"""
import functools

try:
    # pylint: disable = unused-import
    from matplotlib import pyplot

    HAS_MATPLOTLIB = True
except ImportError:
    pyplot = None
    HAS_MATPLOTLIB = False


def requires_matplotlib(func):
    """Decorator for functions requiring matplotlib"""

    @functools.wraps(func)
    def wrapped(*args, **kwargs):
        if not HAS_MATPLOTLIB:
            raise ImportError(
                f"{func} requires matplotlib to generate curve fit plot."
                ' Run "pip install matplotlib" before.'
            )
        return func(*args, **kwargs)

    return wrapped
