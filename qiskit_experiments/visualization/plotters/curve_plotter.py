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
"""Plotter for curve-fits, specifically from :class:`CurveAnalysis`."""
from typing import List

from qiskit_experiments.framework import Options

from .base_plotter import BasePlotter


class CurvePlotter(BasePlotter):
    """A plotter class to plot results from :class:`CurveAnalysis`.

    :class:`CurvePlotter` plots results from curve-fits, which includes:
        Raw results as a scatter plot.
        Processed results with standard-deviations/confidence intervals.
        Interpolated fit-results from the curve analysis.
        Confidence interval for the fit-results.
        A report on the performance of the fit.
    """

    @classmethod
    def expected_series_data_keys(cls) -> List[str]:
        """Returns the expected series data-keys supported by this plotter.

        Data Keys:
            x: X-values for raw results.
            y: Y-values for raw results. Goes with ``x``.
            x_formatted: X-values for processed results.
            y_formatted: Y-values for processed results. Goes with ``x_formatted``.
            y_formatted_err: Error in ``y_formatted``, to be plotted as error-bars.
            x_interp: Interpolated X-values for a curve-fit.
            y_interp: Y-values corresponding to the fit for ``y_interp`` X-values.
            y_interp_err: The standard-deviations of the fit for each X-value in ``y_interp``.
                This data-key relates to the option ``plot_sigma``.
        """
        return [
            "x",
            "y",
            "x_formatted",
            "y_formatted",
            "y_formatted_err",
            "x_interp",
            "y_interp",
            "y_interp_err",
        ]

    @classmethod
    def expected_supplementary_data_keys(cls) -> List[str]:
        """Returns the expected figures data-keys supported by this plotter.

        Data Keys:
            report_text: A string containing any fit report information to be drawn in a box.
                The style and position of the report is controlled by ``text_box_rel_pos`` and
                ``text_box_text_size`` style parameters in :class:`PlotStyle`.
        """
        return [
            "report_text",
        ]

    @classmethod
    def _default_options(cls) -> Options:
        """Return curve-plotter specific default plotter options.

        Options:
            plot_sigma (List[Tuple[float, float]]): A list of two number tuples
                showing the configuration to write confidence intervals for the fit curve.
                The first argument is the relative sigma (n_sigma), and the second argument is
                the transparency of the interval plot in ``[0, 1]``.
                Multiple n_sigma intervals can be drawn for the same curve.

        """
        options = super()._default_options()
        options.plot_sigma = [(1.0, 0.7), (3.0, 0.3)]
        return options

    def _plot_figure(self):
        """Plots a curve-fit figure."""
        for ser in self.series:
            # Scatter plot with error-bars
            plotted_formatted_data = False
            if self.data_exists_for(ser, ["x_formatted", "y_formatted", "y_formatted_err"]):
                x, y, yerr = self.data_for(ser, ["x_formatted", "y_formatted", "y_formatted_err"])
                self.drawer.draw_scatter(x, y, y_err=yerr, name=ser, zorder=2, legend=True)
                plotted_formatted_data = True

            # Scatter plot
            if self.data_exists_for(ser, ["x", "y"]):
                x, y = self.data_for(ser, ["x", "y"])
                options = {
                    "zorder": 1,
                }
                # If we plotted formatted data, differentiate scatter points by setting normal X-Y
                # markers to gray.
                if plotted_formatted_data:
                    options["color"] = "gray"
                # If we didn't plot formatted data, the X-Y markers should be used for the legend. We add
                # it to ``options`` so it's easier to pass to ``draw_scatter``.
                if not plotted_formatted_data:
                    options["legend"] = True
                self.drawer.draw_scatter(
                    x,
                    y,
                    name=ser,
                    **options,
                )

            # Line plot for fit
            if self.data_exists_for(ser, ["x_interp", "y_interp"]):
                x, y = self.data_for(ser, ["x_interp", "y_interp"])
                self.drawer.draw_line(x, y, name=ser, zorder=3)

            # Confidence interval plot
            if self.data_exists_for(ser, ["x_interp", "y_interp", "y_interp_err"]):
                x, y_interp, y_interp_err = self.data_for(
                    ser, ["x_interp", "y_interp", "y_interp_err"]
                )
                for n_sigma, alpha in self.options.plot_sigma:
                    self.drawer.draw_filled_y_area(
                        x,
                        y_interp + n_sigma * y_interp_err,
                        y_interp - n_sigma * y_interp_err,
                        name=ser,
                        alpha=alpha,
                        zorder=5,
                    )

            # Fit report
            if "report_text" in self.supplementary_data:
                report_text = self.supplementary_data["report_text"]
                self.drawer.draw_text_box(report_text)
