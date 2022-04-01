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

"""Test warning helper."""

from test.base import QiskitExperimentsTestCase
from qiskit_experiments.warnings import deprecated_init_args, deprecated_class
from qiskit_experiments.framework import BaseExperiment, ExperimentConfig


class TestWarningsHelper(QiskitExperimentsTestCase):

    def test_creating_experiment_with_old_args(self):
        """Test raise warning and update configuration when create instance with old kwargs."""
        @deprecated_init_args(
            arguments_map={"some_old_arg1": "some_new_arg1", "some_old_arg2": None},
        )
        class FakeExperiment(BaseExperiment):
            """Fake experiment."""
            def __init__(self, qubits, some_new_arg1):
                super().__init__(qubits)

            def circuits(self):
                pass

        with self.assertWarns(DeprecationWarning):
            instance = FakeExperiment(qubits=[0], some_old_arg1=1, some_old_arg2=2)

        config = instance.config()
        self.assertDictEqual(config.kwargs, {"qubits": [0], "some_new_arg1": 1})

    def test_roundtrip_old_instance(self):
        """Test raise warning and update configuration when create instance from old config."""

        @deprecated_init_args(
            arguments_map={"some_old_arg1": "some_new_arg1", "some_old_arg2": None},
        )
        class FakeExperiment(BaseExperiment):
            """Fake experiment."""
            def __init__(self, qubits, some_new_arg1):
                super().__init__(qubits)

            def circuits(self):
                pass

        old_config = ExperimentConfig(
            cls=FakeExperiment,
            args=([0], ),
            kwargs={"some_old_arg1": 1, "some_old_arg2": 2},
        )

        with self.assertWarns(DeprecationWarning):
            instance = FakeExperiment.from_config(old_config)

        new_config = ExperimentConfig(
            cls=FakeExperiment,
            args=([0,]),
            kwargs={"some_old_arg1": 1}
        )

        self.json_equiv(instance.config(), new_config)

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
            pass

        @deprecated_class(
            new_cls=NewExperiment,
        )
        class OldExperiment(TempExperiment):
            pass

        with self.assertWarns(DeprecationWarning):
            instance = OldExperiment([0])

        self.assertIsInstance(instance, NewExperiment)
