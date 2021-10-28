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

import copy

from test.fake_backend import FakeBackend
from test.fake_experiment import FakeExperiment
from test.fake_service import FakeService

from qiskit.test import QiskitTestCase

from qiskit_experiments.framework import (
    ParallelExperiment,
    Options,
    CompositeExperimentData,
    BatchExperiment,
)

# pylint: disable=missing-raises-doc


class TestComposite(QiskitTestCase):
    """
    Test composite experiment behavior.
    """

    def test_parallel_options(self):
        """
        Test parallel experiments overriding sub-experiment run and transpile options.
        """
        # These options will all be overridden
        exp0 = FakeExperiment([0])
        exp0.set_transpile_options(optimization_level=1)
        exp2 = FakeExperiment([2])
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


class TestCompositeExperimentData(QiskitTestCase):
    """
    Test operations on objects of CompositeExperimentData
    """

    def setUp(self):
        super().setUp()

        self.backend = FakeBackend()
        self.share_level = "hey"

        exp1 = FakeExperiment([0, 2])
        exp2 = FakeExperiment([1, 3])
        par_exp = ParallelExperiment([exp1, exp2])
        exp3 = FakeExperiment(4)
        batch_exp = BatchExperiment([par_exp, exp3])

        self.rootdata = CompositeExperimentData(batch_exp, backend=self.backend)

        self.rootdata.share_level = self.share_level

    def check_attributes(self, expdata):
        """
        Recursively traverse the tree to verify attributes
        """
        self.assertEqual(expdata.backend, self.backend)
        self.assertEqual(expdata.share_level, self.share_level)

        if isinstance(expdata, CompositeExperimentData):
            components = expdata.component_experiment_data()
            comp_ids = expdata.metadata["component_ids"]
            comp_classes = expdata.metadata["component_classes"]
            for childdata, comp_id, comp_class in zip(components, comp_ids, comp_classes):
                self.check_attributes(childdata)
                self.assertEqual(childdata.parent_id, expdata.experiment_id)
                self.assertEqual(childdata.experiment_id, comp_id)
                self.assertEqual(childdata.__class__.__name__, comp_class)

    def check_if_equal(self, expdata1, expdata2, is_a_copy):
        """
        Recursively traverse the tree and check equality of expdata1 and expdata2
        """
        self.assertEqual(expdata1.backend.name(), expdata2.backend.name())
        self.assertEqual(expdata1.tags, expdata2.tags)
        self.assertEqual(expdata1.experiment_type, expdata2.experiment_type)
        self.assertEqual(expdata1.share_level, expdata2.share_level)

        metadata1 = copy.copy(expdata1.metadata)
        metadata2 = copy.copy(expdata2.metadata)
        if is_a_copy:
            comp_ids1 = metadata1.pop("component_ids", None)
            comp_ids2 = metadata2.pop("component_ids", None)
            if comp_ids1 is None:
                self.assertEqual(comp_ids2, None)
            else:
                self.assertNotEqual(comp_ids1, comp_ids2)
            if expdata1.parent_id is None:
                self.assertEqual(expdata2.parent_id, None)
            else:
                self.assertNotEqual(expdata1.parent_id, expdata2.parent_id)
        else:
            self.assertEqual(expdata1.parent_id, expdata2.parent_id)
        self.assertEqual(metadata1, metadata2)

        if isinstance(expdata1, CompositeExperimentData):
            for childdata1, childdata2 in zip(
                expdata1.component_experiment_data(), expdata2.component_experiment_data()
            ):
                self.check_if_equal(childdata1, childdata2, is_a_copy)

    def test_composite_experiment_data_attributes(self):
        """
        Verify correct attributes of parents and children
        """
        self.check_attributes(self.rootdata)
        self.assertEqual(self.rootdata.parent_id, None)

    def test_composite_save_load(self):
        """
        Verify that saving and loading restores the original composite experiment data object
        """

        self.rootdata.service = FakeService()
        self.rootdata.save()
        loaded_data = CompositeExperimentData.load(
            self.rootdata.experiment_id, self.rootdata.service
        )

        self.check_if_equal(loaded_data, self.rootdata, is_a_copy=False)

    def test_composite_save_metadata(self):
        """
        Verify that saving metadata and loading restores the original composite experiment data object
        """

        self.rootdata.service = FakeService()
        self.rootdata.save_metadata()
        loaded_data = CompositeExperimentData.load(
            self.rootdata.experiment_id, self.rootdata.service
        )

        self.check_if_equal(loaded_data, self.rootdata, is_a_copy=False)

    def test_composite_copy_metadata(self):
        """
        Test CompositeExperimentData._copy_metadata
        """
        new_instance = self.rootdata._copy_metadata()
        self.check_if_equal(new_instance, self.rootdata, is_a_copy=True)
        self.check_attributes(new_instance)
