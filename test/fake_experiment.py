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

"""A FakeExperiment for testing."""

import numpy as np
from qiskit_experiments.framework import BaseExperiment, BaseAnalysis, Options, AnalysisResultData


class FakeAnalysis(BaseAnalysis):
    """
    Dummy analysis class for test purposes only.
    """

    def __init__(self, **kwargs):
        super().__init__()
        self._kwargs = kwargs

    def _run_analysis(self, experiment_data, **options):
        seed = options.get("seed", None)
        rng = np.random.default_rng(seed=seed)
        analysis_results = [
            AnalysisResultData(f"result_{i}", value) for i, value in enumerate(rng.random(3))
        ]
        return analysis_results, None


class FakeExperiment(BaseExperiment):
    """Fake experiment class for testing."""

    @classmethod
    def _default_experiment_options(cls) -> Options:
        return Options(dummyoption=None)

    def __init__(self, qubits=None):
        """Initialise the fake experiment."""
        if qubits is None:
            qubits = [0]
        super().__init__(qubits, analysis=FakeAnalysis())

    def circuits(self):
        """Fake circuits."""
        return []
