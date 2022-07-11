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
Optional pyGSTi helper functions
"""
import functools

try:
    # pylint: disable = unused-import
    import pygsti
    from pygsti.processors import QubitProcessorSpec as QPS
    from pygsti.processors import CliffordCompilationRules as CCR
    from pygsti.baseobjs import QubitGraph as QG

    HAS_PYGSTI = True
except ImportError:
    HAS_PYGSTI = False


def requires_pygsti(func):
    """Decorator for functions requiring matplotlib"""

    @functools.wraps(func)
    def wrapped(*args, **kwargs):
        if not HAS_PYGSTI:
            raise ImportError(
                f"{func} requires pyGSTi to generate circuits." ' Run "pip install pygsti" before.'
            )

    return wrapped
