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
Mock plotter for testing.
"""

from typing import List

from qiskit_experiments.visualization import BaseDrawer, BasePlotter


class MockPlotter(BasePlotter):
    """Mock plotter for visualization tests.

    If :attr:`plotting_enabled` is true, :class:`MockPlotter` will plot formatted data.
    :attr:`plotting_enabled` defaults to false as most test usage of the class uses :class:`MockDrawer`,
    which doesn't generate a useful figure.
    """

    def __init__(self, drawer: BaseDrawer, plotting_enabled: bool = False):
        """Construct a mock plotter instance for testing.

        Args:
            drawer: The drawer to use for plotting
            plotting_enabled: Whether to actually plot using :attr:`drawer` or not. Defaults to False.
        """
        super().__init__(drawer)
        self._plotting_enabled = plotting_enabled

    @property
    def plotting_enabled(self):
        """Whether :class:`MockPlotter` should plot data.

        Defaults to False during construction.
        """
        return self._plotting_enabled

    def _plot_figure(self):
        """Plots a figure if :attr:`plotting_enabled` is True.

        If :attr:`plotting_enabled` is True, :class:`MockPlotter` calls
        :meth:`~BaseDrawer.scatter` for a series titled ``seriesA`` with ``x``, ``y``, and
        ``z`` data-keys assigned to the x and y values and the y-error/standard deviation respectively.
        If :attr:`drawer` generates a figure, then :meth:`figure` should return a scatterplot figure with
        error-bars.
        """
        if self.plotting_enabled:
            self.drawer.scatter(*self.data_for("seriesA", ["x", "y", "z"]), name="seriesA")

    @classmethod
    def expected_series_data_keys(cls) -> List[str]:
        """Dummy data-keys.

        Data Keys:
            x: Dummy value.
            y: Dummy value.
            z: Dummy value.
        """
        return ["x", "y", "z"]

    @classmethod
    def expected_supplementary_data_keys(cls) -> List[str]:
        return []
