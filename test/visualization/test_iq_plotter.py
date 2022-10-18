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
Test IQ plotter.
"""

from test.base import QiskitExperimentsTestCase
from typing import Any, Dict, List, Tuple

import ddt
import numpy as np

from qiskit_experiments.data_processing.discriminator import BaseDiscriminator
from qiskit_experiments.visualization import IQPlotter, MplDrawer


class MockDiscriminator(BaseDiscriminator):
    """A mock discriminator for testing."""

    def __init__(self, is_trained: bool = False):
        """Create a MockDiscriminator instance.

        Args:
            is_trained: Whether the discriminator is trained or not. Defaults to False.
        """
        super().__init__()
        self._is_trained = is_trained
        self.predict_was_called = False
        """Whether :meth:`predict` was called at least once."""

    def predict(self, data: List):
        """Returns dummy predictions where everything has the label ``0``."""
        self.predict_was_called = True
        if isinstance(data, list):
            return [0] * len(data)
        return [0] * data.shape[0]

    def config(self) -> Dict[str, Any]:
        return {
            "predict_was_called": self.predict_was_called,
            "is_trained": self._is_trained,
        }

    def is_trained(self) -> bool:
        return self._is_trained


@ddt.ddt
class TestIQPlotter(QiskitExperimentsTestCase):
    """Test IQPlotter"""

    @classmethod
    def _dummy_data(
        cls,
        is_trained: bool = True,
        n_series: int = 3,
    ) -> Tuple[List, List, BaseDiscriminator]:
        """Create dummy data for the tests.

        Args:
            is_trained: Whether the discriminator should be trained or not. Defaults to True.
            n_series: The number of series to generate dummy data for. Defaults to 3.

        Returns:
            tuple: the tuple ``(points, names, discrim)`` where ``points`` is a list of NumPy arrays of
                IQ points, ``names`` is a list of series names (one for each NumPy array), and
                ``discrim`` is a :class:`MockDiscriminator` instance.
        """
        points = []
        labels = []
        for i in range(n_series):
            points.append(np.random.rand(128, 2))
            labels.append(f"{i}")
        mock_discrim = MockDiscriminator(is_trained)
        return points, labels, mock_discrim

    @ddt.data(True, False)
    def test_discriminator_trained(self, is_trained: bool):
        """Test that the discriminator is only sampled if it is trained.

        Args:
            is_trained: Whether the mock discriminator should be trained or not.
        """
        plotter = IQPlotter(MplDrawer())
        points, labels, discrim = self._dummy_data(is_trained)

        # Add dummy data
        for series_points, series_name in zip(points, labels):
            plotter.set_series_data(series_name, points=series_points)

        # Add un-trained discriminator
        plotter.set_supplementary_data(discriminator=discrim)

        # Call figure() to generate discriminator image, if possible.
        plotter.figure()

        # Assert that MockDiscriminator.predict() was/wasn't called, depending on whether it was trained
        # or not.
        self.assertEqual(is_trained, discrim.predict_was_called)

    def test_end_to_end(self):
        """Test end-to-end functionality of IQPlotter."""
        plotter = IQPlotter(MplDrawer())
        points, labels, discrim = self._dummy_data(is_trained=True)
        for points, series_name in zip(points, labels):
            centroid = np.mean(points, axis=0)
            plotter.set_series_data(series_name, points=points, centroid=centroid)
        plotter.set_supplementary_data(discriminator=discrim)

        plotter.figure()
