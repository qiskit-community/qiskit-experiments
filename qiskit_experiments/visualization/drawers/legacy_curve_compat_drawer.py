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

"""Compatibility wrapper for legacy BaseCurveDrawer."""

import warnings
from typing import Any, Optional, Sequence, Tuple, Union

import numpy as np

from qiskit_experiments.curve_analysis.visualization import BaseCurveDrawer
from qiskit_experiments.warnings import deprecated_class

from ..utils import ExtentTuple
from .base_drawer import BaseDrawer


@deprecated_class(
    "0.6",
    msg="Legacy drawers from `.curve_analysis.visualization are deprecated. This compatibility wrapper "
    "will be removed alongside the deprecated modules removal",
)
class LegacyCurveCompatDrawer(BaseDrawer):
    """A compatibility wrapper for the legacy and deprecated :class:`BaseCurveDrawer`.

    :mod:`qiskit_experiments.curve_analysis.visualization` is deprecated and will be
    replaced with the new :mod:`qiskit_experiments.visualization` module. Analysis
    classes instead use subclasses of :class:`BasePlotter` to generate figures. This
    class wraps the legacy :class:`BaseCurveDrawer` class so it can be used by analysis
    classes, such as :class:`CurveAnalysis`, until it is removed.

    .. note::
        As :class:`BaseCurveDrawer` doesn't support customizing legend entries, the
        ``legend`` and ``label`` parameters in drawing methods (such as
        :meth:`scatter`) are unsupported and do nothing.
    """

    def __init__(self, curve_drawer: BaseCurveDrawer):
        """Create a LegacyCurveCompatDrawer instance.

        Args:
            curve_drawer: A legacy BaseCurveDrawer to wrap in the compatibility drawer.
        """
        super().__init__()
        self._curve_drawer = curve_drawer

    def initialize_canvas(self):
        self._curve_drawer.initialize_canvas()

    def format_canvas(self):
        self._curve_drawer.format_canvas()

    # pylint: disable=unused-argument
    def scatter(
        self,
        x_data: Sequence[float],
        y_data: Sequence[float],
        x_err: Optional[Sequence[float]] = None,
        y_err: Optional[Sequence[float]] = None,
        name: Optional[str] = None,
        label: Optional[str] = None,
        legend: bool = False,
        **options,
    ):
        """Draws scatter points with optional Y errorbars.

        Args:
            x_data: X values.
            y_data: Y values.
            x_err: Unsupported as :class:`BaseCurveDrawer` doesn't support X errorbars.
                Defaults to None.
            y_err: Optional error for Y values.
            name: Name of this series.
            label: Unsupported as :class:`BaseCurveDrawer` doesn't support customizing
                legend entries.
            legend: Unsupported as :class:`BaseCurveDrawer` doesn't support toggling
                legend entries.
            options: Valid options for the drawer backend API.
        """
        if x_err is not None:
            warnings.warn(f"{self.__class__.__name__} doesn't support x_err.")

        if y_err is not None:
            self._curve_drawer.draw_formatted_data(x_data, y_data, y_err, name, **options)
        else:
            self._curve_drawer.draw_raw_data(x_data, y_data, name, **options)

    # pylint: disable=unused-argument
    def line(
        self,
        x_data: Sequence[float],
        y_data: Sequence[float],
        name: Optional[str] = None,
        label: Optional[str] = None,
        legend: bool = False,
        **options,
    ):
        """Draw fit line.

        Args:
            x_data: X values.
            y_data: Fit Y values.
            name: Name of this series.
            label: Unsupported as :class:`BaseCurveDrawer` doesn't support customizing
                legend entries.
            legend: Unsupported as :class:`BaseCurveDrawer` doesn't support toggling
                legend entries.
            options: Valid options for the drawer backend API.
        """
        self._curve_drawer.draw_fit_line(x_data, y_data, name, **options)

    # pylint: disable=unused-argument
    def filled_y_area(
        self,
        x_data: Sequence[float],
        y_ub: Sequence[float],
        y_lb: Sequence[float],
        name: Optional[str] = None,
        label: Optional[str] = None,
        legend: bool = False,
        **options,
    ):
        """Draw filled area as a function of x-values.

        Args:
            x_data: X values.
            y_ub: The upper boundary of Y values.
            y_lb: The lower boundary of Y values.
            name: Name of this series.
            label: Unsupported as :class:`BaseCurveDrawer` doesn't support customizing
                legend entries.
            legend: Unsupported as :class:`BaseCurveDrawer` doesn't support toggling
                legend entries.
            options: Valid options for the drawer backend API.
        """

        self._curve_drawer.draw_confidence_interval(x_data, y_ub, y_lb, name, **options)

    # pylint: disable=unused-argument
    def filled_x_area(
        self,
        x_ub: Sequence[float],
        x_lb: Sequence[float],
        y_data: Sequence[float],
        name: Optional[str] = None,
        label: Optional[str] = None,
        legend: bool = False,
        **options,
    ):
        """Does nothing as this is functionality not supported by :class:`BaseCurveDrawer`."""
        warnings.warn(f"{self.__class__.__name__}.filled_x_area is not supported.")

    # pylint: disable=unused-argument
    def textbox(
        self,
        description: str,
        rel_pos: Optional[Tuple[float, float]] = None,
        **options,
    ):
        """Draw textbox.

        Args:
            description: A string to be drawn inside a text box.
            rel_pos: Unsupported as :class:`BaseCurveDrawer` doesn't support modifying
                the location of text in :meth:`textbox` or
                :meth:`BaseCurveDrawer.draw_fit_report`.
            options: Valid options for the drawer backend API.
        """

        self._curve_drawer.draw_fit_report(description, **options)

    # pylint: disable=unused-argument
    def image(
        self,
        data: np.ndarray,
        extent: Optional[ExtentTuple] = None,
        name: Optional[str] = None,
        label: Optional[str] = None,
        cmap: Optional[Union[str, Any]] = None,
        cmap_use_series_colors: bool = False,
        colorbar: bool = False,
        **options,
    ):
        warnings.warn(f"{self.__class__.__name__}.image is not supported.")

    @property
    def figure(self):
        return self._curve_drawer.figure

    def set_options(self, **fields):
        ## Handle option name changes
        # BaseCurveDrawer used `plot_options` instead of `series_params`
        if "series_params" in fields:
            fields["plot_options"] = fields.pop("series_params")
        # PlotStyle parameters are normal options in BaseCurveDrawer.
        if "custom_style" in fields:
            custom_style = fields.pop("custom_style")
            for key, value in custom_style.items():
                fields[key] = value

        self._curve_drawer.set_options(**fields)

    def set_figure_options(self, **fields):
        self.set_options(**fields)
