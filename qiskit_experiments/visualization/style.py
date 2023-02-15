# This code is part of Qiskit.
#
# (C) Copyright IBM 2022, 2023.
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

    This style class is used by :class:`BasePlotter` and :class:`BaseDrawer`. The
    default style for Qiskit Experiments is defined in :meth:`default_style`. Style
    parameters are stored as dictionary entries, grouped by graphics or figure
    component. For example, style parameters relating to textboxes have the prefix
    ``textbox_``. For default style parameter names and their values, see the
    :meth:`default_style` method.

    .. rubric:: Example
    .. code-block:: python

        # Create custom style
        custom_style = PlotStyle(
            {
                "legend_loc": "upper right",
                "textbox_rel_pos": (1, 1),
                "textbox_text_size": 14,
            }
        )

        # Create full style instance by combining with default style.
        full_style = PlotStyle.merge(PlotStyle.default_style(), custom_style)

        ## Query style parameters
        # Returns "upper right"
        full_style["legend_loc"]
        # Returns the value provided in ``PlotStyle.default_style()``
        full_style["axis_label_size"]
    """

    @classmethod
    def default_style(cls) -> "PlotStyle":
        # Ignore pylint warnings as `Style Parameters` doesn't contain method parameters.
        # pylint: disable=differing-param-doc,differing-type-doc
        """The default style across Qiskit Experiments.

        The following is a description of the default style parameters are what they are
        used for.

        Style Parameters:
            figsize (Tuple[int,int]): The size of the figure ``(width, height)``, in
                inches.
            legend_loc (Optional[str]): The location of the legend in axis coordinates.
                If None, location is automatically determined by the drawer.
            tick_label_size (int): The font size for tick labels.
            axis_label_size (int): The font size for axis labels.
            symbol_size (float): The size of symbols for points/markers, proportional to
                the area of the drawn graphic.
            errorbar_capsize (float): The size of end-caps for error-bars.
            textbox_rel_pos (Tuple[float,float]): The relative position ``(horizontal,
                vertical)`` of textboxes, as a percentage of the canvas dimensions.
            textbox_text_size (int): The font size for textboxes.

        Returns:
            The default plot style used by Qiskit Experiments.
        """
        style = {
            # size of figure (width, height)
            "figsize": (8, 5),  # Tuple[int, int]
            # legend location (vertical, horizontal) or None.
            "legend_loc": None,  # str
            # size of tick label
            "tick_label_size": 14,  #  int
            # size of axis label
            "axis_label_size": 16,  # int
            # relative position of a textbox
            "textbox_rel_pos": (0.5, -0.25),  # Tuple[float, float]
            # size of textbox text
            "textbox_text_size": 12,  # int
            # size of caps for error-bars
            "errorbar_capsize": 4,  # float
            # Default size of symbols, used for graphics where symbols are drawn for points.
            "symbol_size": 6.0,  # float
        }
        return cls(**style)

    @classmethod
    def merge(cls, style1: "PlotStyle", style2: "PlotStyle") -> "PlotStyle":
        """Merge ``style2`` into ``style1`` as a new PlotStyle instance.

        This method merges an additional style ``style2`` into a base instance
        ``style1``, returning the merged style instance instead of modifying the inputs.

        Args:
            style1: Base PlotStyle instance.
            style2: Additional PlotStyle instance.

        Returns:
            Merged style instance.
        """
        return PlotStyle({**style1, **style2})
