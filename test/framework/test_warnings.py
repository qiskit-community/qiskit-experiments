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

# pylint: disable=unused-argument, unused-variable
"""Test warning helper."""

from test.base import QiskitExperimentsTestCase

from qiskit_experiments.framework import BaseExperiment
from qiskit_experiments.warnings import deprecated_class


class TestWarningsHelper(QiskitExperimentsTestCase):
    """Test case for warnings decorator with tricky behavior."""

    def test_switch_class(self):
        """Test old class is instantiated as a new class instance."""

        # Here we assume we want to rename class but want to have deprecation period
        class TempExperiment(BaseExperiment):
            """Fake experiment."""

            def __init__(self, qubits):
                super().__init__(qubits)

            def circuits(self):
                pass

        class NewExperiment(TempExperiment):
            """Experiment to be renamed."""

            pass

        @deprecated_class(
            new_cls=NewExperiment,
        )
        class OldExperiment(TempExperiment):
            """Original experiment."""

            pass

        with self.assertWarns(DeprecationWarning):
            instance = OldExperiment([0])

        self.assertIsInstance(instance, NewExperiment)
