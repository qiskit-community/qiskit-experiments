# This code is part of Qiskit.
#
# (C) Copyright IBM 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
"""
Experiment Data class
"""
import logging

from qiskit.providers import Backend
from qiskit.providers.experiment import ExperimentDataV1

LOG = logging.getLogger(__name__)


class ExperimentData(ExperimentDataV1):
    """Qiskit Experiments Data container class"""

    def __init__(
        self,
        experiment=None,
        backend=None,
    ):
        """Initialize experiment data.

        Args:
            experiment (BaseExperiment): Optional, experiment object that generated the data.
            backend (Backend): Optional, Backend the experiment runs on.

        Raises:
            ExperimentError: If an input argument is invalid.
        """
        self._experiment = experiment

        super().__init__(experiment_type=experiment.experiment_type if experiment else "unknown",
                         backend=backend)

    @property
    def experiment(self):
        """Return Experiment object.

        Returns:
            BaseExperiment: the experiment object.
        """
        return self._experiment

