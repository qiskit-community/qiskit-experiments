# This code is part of Qiskit.
#
# (C) Copyright IBM 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
"""
Configurable stylesheet for :class:`BasePlotter` and :class:`BaseDrawer`.
"""
from typing import Tuple
from qiskit_experiments.framework import Options
from copy import copy


class PlotStyle(Options):
    """A stylesheet for :class:`BasePlotter` and :class:`BaseDrawer`.

    This style class is used by :class:`BasePlotter` and :class:`BaseDrawer`, and must not be confused
    with :class:`~qiskit_experiments.visualization.fit_result_plotters.PlotterStyle`. The default style for Qiskit Experiments is defined in :meth:`default_style`.
    """

    @classmethod
    def default_style(cls) -> "PlotStyle":
        """The default style across Qiskit Experiments.

        Returns:
            PlotStyle: The default plot style used by Qiskit Experiments.
        """
        new = cls()
        # size of figure (width, height)
        new.figsize: Tuple[int, int] = (8, 5)

        # legent location (vertical, horizontal)
        new.legend_loc: str = "center right"

        # size of tick label
        new.tick_label_size: int = 14

        # size of axis label
        new.axis_label_size: int = 16

        # relative position of fit report
        new.fit_report_rpos: Tuple[float, float] = (0.6, 0.95)

        # size of fit report text
        new.fit_report_text_size: int = 14

        return new

    def update(self, other_style: "PlotStyle"):
        """Updates the plot styles fields with those set in ``other_style``.

        Args:
            other_style: The style with new field values.
        """
        self.update_options(**other_style._fields)

    @classmethod
    def merge(cls, style1: "PlotStyle", style2: "PlotStyle") -> "PlotStyle":
        """Merges two PlotStyle instances.

        The styles are merged such that style fields in ``style2`` have priority. i.e., a field ``foo``,
        defined in both input styles, will have the value :code-block:`style2.foo` in the output.

        Args:
            style1: The first style.
            style2: The second style.

        Returns:
            PlotStyle: A plot style containing the combined fields of both input styles.
        """
        new_style = copy(style1)
        new_style.update(style2)
        return new_style
