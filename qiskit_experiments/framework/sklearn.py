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
Optional sklearn helper functions
"""
import functools

try:
    # pylint: disable = unused-import
    from sklearn.mixture import GaussianMixture

    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


def requires_sklearn(func):
    """Decorator for functions requiring matplotlib"""

    @functools.wraps(func)
    def wrapped(*args, **kwargs):
        if not HAS_SKLEARN:
            raise ImportError(
                f"{func} requires scikit-learn, which can be installed with "
                '"pip install scikit-learn" to install.'
            )
        return func(*args, **kwargs)

    return wrapped
