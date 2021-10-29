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

from qiskit_experiments.framework import BaseExperiment, BaseAnalysis, Options


class FakeAnalysis(BaseAnalysis):
    """
    Dummy analysis class for test purposes only.
    """

    def _run_analysis(self, experiment_data, **options):
        return [], None


class FakeExperiment(BaseExperiment):
    """Fake experiment class for testing."""

    __analysis_class__ = FakeAnalysis

    @classmethod
    def _default_experiment_options(cls) -> Options:
        return Options(dummyoption=None)

    def __init__(self, qubits=1):
        """Initialise the fake experiment."""
        super().__init__(qubits)

    def circuits(self):
        """Fake circuits."""
        return []
