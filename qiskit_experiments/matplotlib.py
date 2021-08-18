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
import threading

from matplotlib import pyplot as plt

from qiskit.exceptions import QiskitError


def requires_matplotlib(func):
    """Decorator for functions requiring matplotlib"""

    @functools.wraps(func)
    def wrapped(*args, **kwargs):
        """
        Analysis/plotting is done in a separate thread (so it doesn't block the
        main thread), but matplotlib doesn't support GUI mode in a child thread.
        Therefore, we have to switch to a non-GUI backend.
        This has to be done carefully, because switching backends in one of the
        threads (either the main thread or one of the child threads) closes all
        the existing figures in all the threads. In addition, switching backends
        (either by `use` or `switch_backend`) inside a child thread sometimes
        makes Windows trigger an exception, stating that this can be done
        only in the main thread.
        The code below switches to a non-GUI backend "agg" before running the
        function, if the current thread is the main thread.
        An alternative is to run this in a separate process, but then
        we'd need to deal with pickling issues.

        Returns:
            Any: the return value of `func`

        Raises:
            QiskitError: if the current backend is not "agg" and the current
                thread is not the main thread.
        """

        current_backend = plt.get_backend()
        if current_backend != "agg":
            if threading.current_thread() is not threading.main_thread():
                raise QiskitError(
                    "Trying to switch from backend "
                    + current_backend
                    + " to agg in a child thread."
                )
            plt.switch_backend("agg")

        return func(*args, **kwargs)

    return wrapped
