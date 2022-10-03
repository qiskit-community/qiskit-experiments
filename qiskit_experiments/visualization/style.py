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


class PlotStyle(dict):
    """A stylesheet for :class:`BasePlotter` and :class:`BaseDrawer`.

    This style class is used by :class:`BasePlotter` and :class:`BaseDrawer`. The default style for
    Qiskit Experiments is defined in :meth:`default_style`. Style parameters are stored as dictionary
    entries, grouped by graphics or figure component. For example, style parameters relating to textboxes
    have the prefix ``textbox.``. For default style parameter names and their values, see the
    :meth:`default_style` method.

    Example:
    .. code-block:: python
        # Create custom style
        custom_style = PlotStyle(
            {
                "legend.loc": "upper right",
                "textbox.rel_pos": (1, 1),
                "textbox.text_size": 14,
            }
        )

        # Create full style, using PEP448 to combine with default style.
        full_style = PlotStyle.merge(PlotStyle.default_style(), custom_style)

        # Query style parameters
        full_style["legend.loc"]        # Returns "upper right"
        full_style["axis.label_size"]   # Returns the value provided in ``PlotStyle.default_style()``
    """

    @classmethod
    def default_style(cls) -> "PlotStyle":
        """The default style across Qiskit Experiments.

        Style Parameters:
            figsize (Tuple[int,int]): The size of the figure ``(width, height)``, in inches.
            legend.loc (str): The location of the legend.
            tick.label_size (int): The font size for tick labels.
            axis.label_size (int): The font size for axis labels.
            textbox.rel_pos (Tuple[float,float]): The relative position ``(horizontal, vertical)`` of
                textboxes, as a percentage of the canvas dimensions.
            textbox.text_size (int): The font size for textboxes.

        Returns:
            PlotStyle: The default plot style used by Qiskit Experiments.
        """
        style = {
            # size of figure (width, height)
            "figsize": (8, 5),  # Tuple[int, int]
            # legend location (vertical, horizontal)
            "legend.loc": "center right",  # str
            # size of tick label
            "tick.label_size": 14,  #  int
            # size of axis label
            "axis.label_size": 16,  # int
            # relative position of a textbox
            "textbox.rel_pos": (0.6, 0.95),  # Tuple[float, float]
            # size of textbox text
            "textbox.text_size": 14,  # int
        }
        return cls(**style)

    @classmethod
    def merge(cls, style1: "PlotStyle", style2: "PlotStyle") -> "PlotStyle":
        """Merge ``style2`` into ``style1`` as a new PlotStyle instance.

        This method merges an additional style ``style2`` into a base instance ``style1``, returning the
        merged style instance instead of modifying the inputs.

        Args:
            style1: Base PlotStyle instance.
            style2: Additional PlotStyle instance.

        Returns:
            PlotStyle: merged style instance.
        """
        return PlotStyle({**style1, **style2})
