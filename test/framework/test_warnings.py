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
from qiskit_experiments.warnings import deprecate_arguments
from qiskit_experiments.warnings import qubit_deprecate


class TempExperiment(BaseExperiment):
    """Fake experiment."""

    def __init__(self, physical_qubits):
        super().__init__(physical_qubits)

    def circuits(self):
        pass


class TestWarningsHelper(QiskitExperimentsTestCase):
    """Test case for warnings decorator with tricky behavior."""

    def test_switch_class(self):
        """Test old class is instantiated as a new class instance."""

        # Here we assume we want to rename class but want to have deprecation period
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

    def test_deprecated_argument(self):
        """Test that deprecating arguments works as expected."""

        class OldExperiment(TempExperiment):
            """Original experiment."""

            @deprecate_arguments({"qubits": "physical_qubits"})
            def __init__(self, physical_qubits):
                super().__init__(physical_qubits)

        # Test that providing both old and new kwargs throws an error
        with self.assertRaises(TypeError):
            instance = OldExperiment(qubits=[0], physical_qubits=[0])
        with self.assertWarns(DeprecationWarning):
            instance = OldExperiment(qubits=[0])
        self.assertEqual(instance._physical_qubits, (0,))

    def test_deprecated_qubit(self):
        """Test for the temporary qubit_deprecate wrapper."""

        class OldExperiment(TempExperiment):
            """Original experiment."""

            @qubit_deprecate()
            def __init__(self, physical_qubits):
                super().__init__(physical_qubits)

        with self.assertWarns(DeprecationWarning):
            instance = OldExperiment(qubit=0)
        self.assertEqual(instance._physical_qubits, (0,))
