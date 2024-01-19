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

import warnings
from itertools import product
from test.base import QiskitExperimentsTestCase
from typing import Any, Dict, List, Tuple

import ddt
import numpy as np

from qiskit_experiments.data_processing.discriminator import BaseDiscriminator
from qiskit_experiments.visualization import IQPlotter, MplDrawer


class MockDiscriminatorNotTrainedException(Exception):
    """Mock exception to be raised when :meth:`MockDiscriminator.predict` is called on an untrained
    :class:`.MockDiscriminator`."""

    pass


class MockDiscriminator(BaseDiscriminator):
    """A mock discriminator for testing."""

    def __init__(
        self, is_trained: bool = False, n_states: int = 3, raise_predict_not_trained: bool = False
    ):
        """Create a MockDiscriminator instance.

        Args:
            is_trained: Whether the discriminator is trained or not. Defaults to False.
            n_states: The number of states/labels. Defaults to 3.
            raise_predict_not_trained: Whether to raise an exception if :meth:`predict` is called and
                :attr:`is_trained` is ``False``. Raises
        """
        super().__init__()
        self._is_trained = is_trained
        self._n_states = n_states
        self._raise_predict_not_trained = raise_predict_not_trained
        self.predict_was_called = False
        """Whether :meth:`predict` was called at least once."""

    def predict(self, data: List):
        """Returns dummy predictions with random labels."""
        self.predict_was_called = True
        if self._raise_predict_not_trained and not self.is_trained():
            raise MockDiscriminatorNotTrainedException()
        if isinstance(data, list):
            return np.random.choice([f"{i}" for i in range(self._n_states)], len(data)).tolist()
        return np.random.choice([f"{i}" for i in range(self._n_states)], data.shape[0])

    def config(self) -> Dict[str, Any]:
        return {
            "predict_was_called": self.predict_was_called,
            "is_trained": self._is_trained,
            "n_states": self._n_states,
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
        raise_predict_not_trained: bool = False,
        factor: float = 1,
    ) -> Tuple[List, List, BaseDiscriminator]:
        """Create dummy data for the tests.

        Args:
            is_trained: Whether the discriminator should be trained or not. Defaults to True.
            n_series: The number of series to generate dummy data for. Defaults to 3.
            raise_predict_not_trained: Passed to the discriminator :class:`.MockDiscriminator` class.
            factor: A scaler factor by which to multipl all data.


        Returns:
            tuple: the tuple ``(points, names, discrim)`` where ``points`` is a list of NumPy arrays of
                IQ points, ``names`` is a list of series names (one for each NumPy array), and
                ``discrim`` is a :class:`.MockDiscriminator` instance.
        """
        points = []
        labels = []
        for i in range(n_series):
            points.append(np.random.rand(128, 2) * factor)
            labels.append(f"{i}")
        mock_discrim = MockDiscriminator(
            is_trained, n_states=n_series, raise_predict_not_trained=raise_predict_not_trained
        )
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
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", "Discriminator was provided but")
            plotter.figure()

        # Assert that MockDiscriminator.predict() was/wasn't called, depending on whether it was trained
        # or not.
        self.assertEqual(
            is_trained,
            discrim.predict_was_called,
            msg=f"Discriminator `predict()` {'was' if is_trained else 'was not'} meant to be called, "
            f"but {'was' if discrim.predict_was_called else 'was not'} called. is_trained={is_trained}.",
        )

    @ddt.data(*list(product([True, False], repeat=3)))
    def test_end_to_end(self, args):
        """Test end-to-end functionality of IQPlotter."""
        # Expand args
        with_centroids, with_misclassified, with_discriminator = args

        # Create plotter and add data
        plotter = IQPlotter(MplDrawer())
        plotter.set_options(flag_misclassified=with_misclassified)
        points, labels, discrim = self._dummy_data(
            is_trained=True,
        )
        for series_points, series_name in zip(points, labels):
            plotter.set_series_data(series_name, points=series_points)
            if with_centroids:
                centroid = np.mean(series_points, axis=0)
                plotter.set_series_data(series_name, centroid=centroid)
        if with_discriminator:
            plotter.set_supplementary_data(discriminator=discrim)

        # Generate figure
        plotter.figure()

        # Verify that the correct number of series colours were created. If we are flagging misclassified
        # points, we have one extra series. This assumes we are using `MplDrawer`. The discriminator
        # should label each input as one of the series-names, which means we should have the same number
        # of entries in `plotter.drawer._series` as colours queried from `MplDrawer._get_default_color`
        # (stored in `MplDrawer._series`).
        self.assertEqual(
            len(plotter.drawer._series),
            len(points) + (1 if with_misclassified and with_discriminator else 0),
            msg="Number of series plotted by IQPlotter does not match the number of series from "
            f"the dummy data. Expected {len(points)} but got {len(plotter.drawer._series)}. Series="
            f"{plotter.drawer._series}.",
        )
