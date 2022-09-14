from typing import List

from qiskit_experiments.framework import Options
from qiskit_experiments.visualization import BaseDrawer

from .base_plotter import BasePlotter


class CurvePlotter(BasePlotter):
    def __init__(self, drawer: BaseDrawer):
        super().__init__(drawer)

    @classmethod
    def _default_series_data_keys(cls) -> List[str]:
        return [
            "x",
            "y",
            "x_formatted",
            "y_formatted",
            "y_formatted_err",
            "x_interp",
            "y_mean",
            "sigmas",
            "fit_report",
        ]

    @classmethod
    def _default_options(cls) -> Options:
        """Return curve-plotter specific default plotting options.

        Plot Options:
            plot_sigma (List[Tuple[float, float]]): A list of two number tuples
                showing the configuration to write confidence intervals for the fit curve.
                The first argument is the relative sigma (n_sigma), and the second argument is
                the transparency of the interval plot in ``[0, 1]``.
                Multiple n_sigma intervals can be drawn for the single curve.

        """
        options = super()._default_options()
        options.plot_sigma = [(1.0, 0.7), (3.0, 0.3)]
        return options

    def _plot_figure(self):
        for ser in self.series:
            # Scatter plot
            if self.data_exists_for(ser, ["x", "y"]):
                x, y = self.data_for(ser, ["x", "y"])
                self.drawer.draw_raw_data(x, y, ser)

            # Scatter plot with error-bars
            if self.data_exists_for(ser, ["x_formatted", "y_formatted", "y_formatted_err"]):
                x, y, yerr = self.data_for(ser, ["x_formatted", "y_formatted", "y_formatted_err"])
                self.drawer.draw_formatted_data(x, y, yerr, ser)

            # Line plot for fit
            if self.data_exists_for(ser, ["x_interp", "y_mean"]):
                x, y = self.data_for(ser, ["x_interp", "y_mean"])
                self.drawer.draw_fit_line(x, y, ser)

            # Confidence interval plot
            if self.data_exists_for(ser, ["x_interp", "y_mean", "sigmas"]):
                x, y_mean, sigmas = self.data_for(ser, ["x_interp", "y_mean", "sigmas"])
                for n_sigma, alpha in self.options.plot_sigma:
                    self.drawer.draw_confidence_interval(
                        x,
                        y_mean + n_sigma * sigmas,
                        y_mean - n_sigma * sigmas,
                        ser,
                        alpha=alpha,
                    )

            # Fit report
            if "fit_report" in self.figure_data:
                fit_report_description = self.figure_data["fit_report"]
                self.drawer.draw_fit_report(fit_report_description)
