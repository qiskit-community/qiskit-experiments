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
Matplotlib helper functions
"""

import functools

from matplotlib import pyplot


def requires_matplotlib(func):
    """Decorator for functions requiring matplotlib"""

    @functools.wraps(func)
    def wrapped(*args, **kwargs):
        # Analysis/plotting is done in a separate thread (so it doesn't block the
        # main thread), but matplotlib doesn't support GUI mode in a child thread.
        # The code below switches to a non-GUI backend "Agg" when creating the
        # plot. An alternative is to run this in a separate process, but then
        # we'd need to deal with pickling issues.

        saved_backend = pyplot.get_backend()
        pyplot.switch_backend("Agg")
        try:
            ret_val = func(*args, **kwargs)
        finally:
            pyplot.switch_backend(saved_backend)
        return ret_val

    return wrapped
