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
from copy import copy
from typing import Dict, Tuple

from qiskit_experiments.framework import Options


class PlotStyle(Options):
    """A stylesheet for :class:`BasePlotter` and :class:`BaseDrawer`.

    This style class is used by :class:`BasePlotter` and :class:`BaseDrawer`, and must not be confused
    with :class:`~qiskit_experiments.visualization.fit_result_plotters.PlotterStyle`. The default style
    for Qiskit Experiments is defined in :meth:`default_style`. :class:`PlotStyle` subclasses
    :class:`Options` and has a similar interface. Extra helper methods are included to merge and update
    instances of :class:`PlotStyle`: :meth:`merge` and :meth:`update` respectively.
    """

    @classmethod
    def default_style(cls) -> "PlotStyle":
        """The default style across Qiskit Experiments.

        Returns:
            PlotStyle: The default plot style used by Qiskit Experiments.
        """
        # pylint: disable = attribute-defined-outside-init
        # We disable attribute-defined-outside-init so we can set style parameters outside of the
        # initialization call and thus include type hints.
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
        new.report_rpos: Tuple[float, float] = (0.6, 0.95)

        # size of fit report text
        new.report_text_size: int = 14

        return new

    def update(self, other_style: "PlotStyle"):
        """Updates the plot styles fields with those set in ``other_style``.

        Args:
            other_style: The style with new field values.
        """
        self.update_options(**other_style._fields)

    @classmethod
    def merge(cls, style1: "PlotStyle", style2: "PlotStyle") -> "PlotStyle":
        """Merge two PlotStyle instances into a new instance.

        The styles are merged such that style fields in ``style2`` have priority. i.e., a field ``foo``,
        defined in both input styles, will have the value :code-block:`style2.foo` in the output.

        Args:
            style1: The first style.
            style2: The second style.

        Returns:
            PlotStyle: A plot style containing the combined fields of both input styles.

        Raises:
            RuntimeError: If either of the input styles is not of type :class:`PlotStyle`.
        """
        if not isinstance(style1, PlotStyle) or not isinstance(style2, PlotStyle):
            raise RuntimeError(
                "Incorrect style type for PlotStyle.merge: expected PlotStyle but got "
                f"{type(style1).__name__} and {type(style2).__name__}"
            )
        new_style = copy(style1)
        new_style.update(style2)
        return new_style

    def config(self) -> Dict:
        """Return the config dictionary for this PlotStyle instance.

        .. Note::
            Validators are not currently supported

        Returns:
            dict: A dictionary containing the config of the plot style.
        """
        return {
            "cls": type(self),
            **self._fields,
        }

    def __json_encode__(self):
        return self.config()

    @classmethod
    def __json_decode__(cls, value):
        kwargs = value
        kwargs.pop("cls")
        inst = cls(**kwargs)
        return inst
