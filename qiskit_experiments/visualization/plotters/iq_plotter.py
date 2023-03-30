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
"""Plotter for IQ data."""
import warnings
from itertools import product
from typing import List, Optional, Tuple

import numpy as np

from qiskit_experiments.data_processing.discriminator import BaseDiscriminator
from qiskit_experiments.framework import Options

from ..utils import DataExtentCalculator, ExtentTuple
from .base_plotter import BasePlotter


class IQPlotter(BasePlotter):
    """A plotter class to plot IQ data.

    :class:`.IQPlotter` plots results from experiments which used measurement-level 1,
    i.e. IQ data. This class also supports plotting predictions from a discriminator
    (subclass of :class:`.BaseDiscriminator`), which is used to classify IQ results into
    labels. The discriminator labels are matched with the series names to generate an
    image of the predictions. Points that are misclassified by the discriminator are
    flagged in the figure (see the ``flag_misclassified`` option). A canonical
    application of :class:`.IQPlotter` is for classification of single-qubit readout for
    different prepared states.

    Example:
        .. code-block:: python

            # Create plotter
            plotter = IQPlotter(MplDrawer())

            # Iterate over results, one per prepared state. Add points and centroid to
            # plotter and set label for prepared states as |n> where n is the
            # prepared-state number.
            series_params = {}
            for res in results:
                # Get IQ points from result memory.
                points = res.memory

                # Compute centroid as mean of all points.
                centroid = np.mean(points, axis=1)

                # Get ``prep``, which is part of the result metadata.
                prep = res.prep

                # Create label as a ket instead of just a state number (i.e., prep).
                series_params[prep] = {
                    "label":f"|{prep}>",
                }

                plotter.set_series_data(prep, points=points, centroid=centroid)
            plotter.set_figure_options(series_params=series_params)
            ...
            # Optional: Add trained discriminator.
            discrim = MyIQDiscriminator()
            # Discriminator labels are the same as series names.
            discrim.fit(train_data, train_labels)
            plotter.set_supplementary_data(discriminator=discrim)
            ...
            # Plot figure.
            fig = plotter.figure()
    """

    @classmethod
    def expected_series_data_keys(cls) -> List[str]:
        """Returns the expected series data keys supported by this plotter.

        Data Keys:
            points: Single-shot IQ data.
            centroid: Averaged IQ data.
        """
        return [
            "points",
            "centroid",
        ]

    @classmethod
    def expected_supplementary_data_keys(cls) -> List[str]:
        """Returns the expected figures data keys supported by this plotter.

        Data Keys:
            discriminator: A trained discriminator that classifies IQ points. If
                provided, the predictions of the discriminator will be sampled to
                generate a background image, indicating the regions for each predicted
                outcome. The predictions are assumed to be series names (``Union[str,
                int, float]``). The generated image allows viewers to see how well the
                discriminator classifies the provided series data. Must be a subclass of
                :class:`.BaseDiscriminator`. See :attr:`options` for ways to control the
                generation of the discriminator prediction image.
            fidelity: A float representing the fidelity of the discrimination.
        """
        return [
            "discriminator",
            "fidelity",
        ]

    def _compute_extent(self) -> Optional[ExtentTuple]:
        """Computes the extent tuple of the data being plotted.

        Returns:
            The tuple ``(x_min, x_max, y_min, y_max)``, defining a rectangle containing
            all the data for this plotter. If the plotter contains no data, ``None`` is
            returned instead.
        """
        ext_calc = DataExtentCalculator(
            multiplier=self.options.discriminator_multiplier,
            aspect_ratio=self.options.discriminator_aspect_ratio,
        )
        has_registered_data = False
        for series in self.series:
            if self.data_exists_for(series, "points"):
                (points,) = self.data_for(series, "points")
                ext_calc.register_data(points)
                has_registered_data = True
            if self.data_exists_for(series, "centroid"):
                (centroid,) = self.data_for(series, "centroid")
                ext_calc.register_data(np.asarray(centroid).reshape(1, 2))
                has_registered_data = True
        if self.figure_options.xlim:
            ext_calc.register_data(self.figure_options.xlim, dim=0)
            has_registered_data = True
        if self.figure_options.ylim:
            ext_calc.register_data(self.figure_options.ylim, dim=1)
            has_registered_data = True
        if has_registered_data:
            return ext_calc.extent()
        else:
            return None

    def _compute_discriminator_image(
        self,
    ) -> Tuple[Optional[np.ndarray], Optional[ExtentTuple]]:
        """Compute the array/image sampled from the discriminator predictions.

        Returns:
            The tuple ``(img, extent)`` where ``img`` is an optional 2D NumPy array of
            predictions and ``extent`` is a tuple of the extent ``(x_min, x_max, y_min,
            y_max)`` of the image.
        """
        # If the discriminator is not provided, cannot compute the image.
        if "discriminator" not in self.supplementary_data:
            return None, None

        if self.options.discriminator_extent:
            extent = self.options.discriminator_extent
        else:
            extent = self._compute_extent()
        # If ``extent`` is None, we don't have data to plot and thus cannot generate the
        # discriminator image.
        if extent is None:
            return None, None

        # Get the discriminator and check if it is trained. If not, return.
        discrim: BaseDiscriminator = self.supplementary_data["discriminator"]
        if not discrim.is_trained():
            return None, None

        # Compute discriminator resolution.
        extent_range = np.diff(np.asarray(extent).reshape(2, 2), axis=1).flatten()
        resolution = (
            (extent_range / np.max(extent_range)) * self.options.discriminator_max_resolution
        ).astype(int)

        # Create coordinates for each pixel in the image, in the same units as `extent`.
        coords = [
            [x, y]
            for x, y in product(
                np.linspace(extent[0], extent[1], resolution[0]),
                np.linspace(extent[2], extent[3], resolution[1]),
            )
        ]

        # Get predictions for coordinates from the discriminator.
        predictions = discrim.predict(coords)

        # Unwrap predictions into a 2D array
        predictions = np.reshape(predictions, tuple(resolution))

        return predictions, extent

    @classmethod
    def _default_options(cls) -> Options:
        """Return iq-plotter specific default plotter options.

        Options:
            plot_discriminator (bool): Whether to plot an image showing the predictions
                of the ``discriminator`` entry in :attr:`supplementary_data`. If True,
                the "discriminator" supplementary data entry must be set.
            discriminator_multiplier (float): The multiplier to use when computing the
                extent of the discriminator plot. The range of the series data is taken
                as the base value and multiplied by ``discriminator_extent_multiplier``
                to compute the extent of the discriminator predictions. Defaults to 1.1.
            discriminator_aspect_ratio (float): The aspect ratio of the extent of the
                discriminator predictions, being ``width/height``. Defaults to ``1`` for
                a square region.
            discriminator_max_resolution (int): The number of pixels to use for the
                largest edge of the discriminator extent, used when sampling the
                discriminator to create the prediction image. Defaults to 1024.
            discriminator_alpha (float): The transparency of the discriminator
                prediction image. Defaults to 0.2 (i.e., 20%).
            discriminator_extent (Optional[ExtentTuple]): An optional tuple defining the
                extent of the image created by sampling from the discriminator. If
                ``None``, the extent tuple is computed using
                ``discriminator_multiplier``, ``discriminator_aspect_ratio``, and the
                series-data ``points`` and ``centroid``. Defaults to ``None``.
            flag_misclassified (bool): Whether to mark misclassified IQ values from all
                ``points`` series data, based on whether their series name is not the
                same as the prediction from the discriminator provided as supplementary
                data. If ``discriminator`` is not provided, ``flag_misclassified`` has
                no effect. Defaults to True.
            misclassified_symbol (str): Symbol for misclassified points, as a
                drawer-compatible string. Defaults to "x".
            misclassified_color (str | tuple): Color for misclassified points, as a
                drawer-compatible string or RGB tuple. Defaults to "r".

        """
        options = super()._default_options()
        # Discriminator options
        options.plot_discriminator = True
        options.discriminator_multiplier = 1.1
        options.discriminator_aspect_ratio = 1.0
        options.discriminator_max_resolution = 1024
        options.discriminator_alpha = 0.2
        options.discriminator_extent = None
        # Points options
        options.flag_misclassified = True
        options.misclassified_symbol = "x"
        options.misclassified_color = "r"
        return options

    @classmethod
    def _default_figure_options(cls) -> Options:
        fig_opts = super()._default_figure_options()
        fig_opts.xlabel = "In-Phase"
        fig_opts.ylabel = "Quadrature"
        fig_opts.xval_unit = "arb."
        fig_opts.yval_unit = "arb."
        fig_opts.xval_unit_scale = False
        fig_opts.yval_unit_scale = False
        return fig_opts

    def _misclassified_points(self, series_name: str, points: np.ndarray) -> Optional[np.ndarray]:
        """Returns a list of IQ coordinates for points that are misclassified by the discriminator.

        Args:
            series_name: The series name to use as the expected discriminator label. If
                the discriminator returns a prediction that doesn't equal
                ``series_name``, it is marked as misclassified.
            points: The list of points to check for misclassification.

        Returns:
            A NumPy array of IQ points, being those that were misclassified by the
            discriminator. If the discriminator isn't set and trained, then `None` is
            returned. The array may be empty.
        """
        # Check if we have a discriminator, and if it is trained. If not, return None.
        if "discriminator" not in self.supplementary_data:
            return None
        discrim: BaseDiscriminator = self.supplementary_data["discriminator"]
        if not discrim.is_trained():
            return None
        classifications = discrim.predict(points)
        misclassified = np.argwhere(classifications != series_name)
        return points[misclassified, :].reshape(-1, 2)

    def _plot_figure(self):
        """Plots an IQ figure."""
        # Plot discriminator first so that subsequent graphics change the automatic limits. This is a
        # function of the way `imshow` works for Matplotlib, in that the limits are automatically changed
        # to the extents of the image being plotted. If `image` is called after `scatter`, then the
        # automatic axis limits will be set to match the extent of the image.
        if "discriminator" in self.supplementary_data and self.options.plot_discriminator:
            if self.options.subplots != (1, 1):
                warnings.warn(
                    "Plotting discriminator predictions with subplots is not well supported by "
                    "IQPlotter as the predictions image will only be drawn on one subplot."
                )
            discrim, extent = self._compute_discriminator_image()
            if discrim is None:
                warnings.warn(
                    "Discriminator was provided but the sampled predictions image could not be "
                    "generated."
                )
            else:
                self.drawer.image(
                    np.flip(discrim.transpose(), axis=0),
                    extent,
                    name="discriminator",
                    cmap_use_series_colors=True,
                    alpha=self.options.discriminator_alpha,
                )
        # Plot points and centroids
        for ser in self.series:
            has_plotted_centroid = False
            if self.data_exists_for(ser, "centroid"):
                (centroid,) = self.data_for(ser, "centroid")
                self.drawer.scatter(
                    centroid[0],
                    centroid[1],
                    name=ser,
                    legend=True,
                    zorder=4,
                    s=20,
                    edgecolor="k",
                    marker="o",
                )
                has_plotted_centroid = True
            if self.data_exists_for(ser, "points"):
                (points,) = self.data_for(ser, "points")
                self.drawer.scatter(
                    points[:, 0],
                    points[:, 1],
                    name=ser,
                    legend=not has_plotted_centroid,
                    zorder=2,
                    s=10,
                    alpha=0.2,
                    marker=".",
                )
                if self.options.flag_misclassified:
                    misclassified_points = self._misclassified_points(ser, points)
                    if misclassified_points is not None:
                        self.drawer.scatter(
                            misclassified_points[:, 0],
                            misclassified_points[:, 1],
                            name="misclassified",
                            legend=False,
                            zorder=3,
                            s=10,
                            alpha=0.4,
                            marker=self.options.misclassified_symbol,
                            color=self.options.misclassified_color,
                        )

        # Fidelity report
        report = self._write_report()
        if len(report) > 0:
            self.drawer.textbox(report)

    def _write_report(self) -> str:
        """Write fidelity report with supplementary_data.

        Subclass can override this method to customize fidelity report. By default, this
        writes the fidelity of the discriminator in the fidelity report.

        Returns:
            Fidelity report.
        """
        report = ""

        if "fidelity" in self.supplementary_data:
            fidelity = self.supplementary_data["fidelity"]
            report += f"fidelity = {fidelity: .4g}"

        return report
