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
Configurable stylesheet.
"""
import dataclasses
from typing import Tuple, List


@dataclasses.dataclass
class PlotterStyle:
    """A stylesheet for curve analysis figure."""

    # size of figure (width, height)
    figsize: Tuple[int, int] = (8, 5)

    # legent location (vertical, horizontal)
    legend_loc: str = "center right"

    # size of tick label
    tick_label_size: int = 14

    # size of axis label
    axis_label_size: int = 16

    # relative position of fit report
    fit_report_rpos: Tuple[float, float] = (0.6, 0.95)

    # size of fit report text
    fit_report_text_size: int = 14

    # sigma values for confidence interval, which are the tuple of (sigma, alpha).
    # the alpha indicates the transparency of the corresponding interval plot.
    plot_sigma: List[Tuple[float, float]] = dataclasses.field(
        default_factory=lambda: [(1.0, 0.7), (3.0, 0.3)]
    )
