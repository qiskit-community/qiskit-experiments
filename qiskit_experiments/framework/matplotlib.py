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

from matplotlib.figure import Figure
from matplotlib.backends.backend_svg import FigureCanvasSVG

default_figure_canvas = FigureCanvasSVG  # pylint: disable=invalid-name
"""Matplotlib canvas to use when rendering a figure. This needs to be a
canvas for a `non-interactive backend
<https://matplotlib.org/stable/tutorials/introductory/usage.html#the-builtin-backends>`_.
The default is `FigureCanvasSVG`."""


def get_non_gui_ax():
    """Return a matplotlib axes that can be used in a child thread.

    Analysis/plotting is done in a separate thread (so it doesn't block the
    main thread), but matplotlib doesn't support GUI mode in a child thread.
    This function creates a separate Figure and attaches a non-GUI
    SVG canvas to it.

    Returns:
        matplotlib.axes.Axes: A matplotlib axes that can be used in a child thread.
    """
    figure = Figure()
    _ = default_figure_canvas(figure)
    return figure.subplots()
