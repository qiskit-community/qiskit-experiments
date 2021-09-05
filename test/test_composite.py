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

"""Class to test composite experiments."""

from typing import Optional, List

from test.fake_backend import FakeBackend
from test.fake_experiment import FakeExperiment

from qiskit import QuantumCircuit
from qiskit.test import QiskitTestCase
from qiskit.test.mock import FakeMelbourne
from qiskit.providers.backend import Backend

from qiskit_experiments.framework import (
    ParallelExperiment,
    Options,
    CompositeExperimentData,
    BaseExperiment,
    BatchExperiment,
)


class TestComposite(QiskitTestCase):
    """
    Test composite experiment behavior.
    """

    def test_parallel_options(self):
        """
        Test parallel experiments overriding sub-experiment run and transpile options.
        """
        # These options will all be overridden
        exp0 = FakeExperiment(0)
        exp0.set_transpile_options(optimization_level=1)
        exp2 = FakeExperiment(2)
        exp2.set_experiment_options(dummyoption="test")
        exp2.set_run_options(shots=2000)
        exp2.set_transpile_options(optimization_level=1)
        exp2.set_analysis_options(dummyoption="test")

        par_exp = ParallelExperiment([exp0, exp2])

        with self.assertWarnsRegex(
            Warning,
            "Sub-experiment run and transpile options"
            " are overridden by composite experiment options.",
        ):
            self.assertEqual(par_exp.experiment_options, Options())
            self.assertEqual(par_exp.run_options, Options(meas_level=2))
            self.assertEqual(par_exp.transpile_options, Options(optimization_level=0))
            self.assertEqual(par_exp.analysis_options, Options())

            par_exp.run(FakeBackend())


class DummyExperiment(BaseExperiment):
    """
    An experiment that does nothing, to fill in the experiment tree
    """

    def circuits(self, backend: Optional[Backend] = None) -> List[QuantumCircuit]:
        return []


class TestCompositeExperimentData(QiskitTestCase):
    """
    Test operations on objects of CompositeExperimentData
    """

    def setUp(self):
        super().setUp()

        self.backend = FakeMelbourne()
        self.job_ids = [1, 2, 3, 4, 5]

        exp1 = DummyExperiment([0, 2])
        exp2 = DummyExperiment([1, 3])
        par_exp = ParallelExperiment([exp1, exp2])
        exp3 = DummyExperiment(4)
        batch_exp = BatchExperiment([par_exp, exp3])

        self.rootdata = CompositeExperimentData(
            batch_exp, backend=self.backend, job_ids=self.job_ids
        )

    def check_attributes(self, expdata):
        """
        Recursively traverse the tree to verify attributes
        """
        self.assertEqual(expdata.backend, self.backend)
        self.assertEqual(expdata.job_ids, self.job_ids)

        if isinstance(expdata, CompositeExperimentData):
            for childdata in expdata.component_experiment_data():
                self.check_attributes(childdata)

    def test_composite_experiment_data_attributes(self):
        """
        Verify correct attributes of parents and children
        """
        self.check_attributes(self.rootdata)

    def test_composite_save_load(self):
        """
        Verify that saving and loading restores the original composite experiment data object
        """
        pass
