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
import uuid

from test.fake_backend import FakeBackend
from test.fake_experiment import FakeExperiment
from test.fake_service import FakeService
from test.base import QiskitExperimentsTestCase

from qiskit import QuantumCircuit
from qiskit.result import Result

from qiskit_experiments.test.utils import FakeJob
from qiskit_experiments.framework import (
    ParallelExperiment,
    Options,
    ExperimentData,
    BatchExperiment,
)

# pylint: disable=missing-raises-doc


class TestComposite(QiskitExperimentsTestCase):
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
        exp2.analysis.set_options(dummyoption="test")

        par_exp = ParallelExperiment([exp0, exp2])

        with self.assertWarnsRegex(
            Warning,
            "Sub-experiment run and transpile options"
            " are overridden by composite experiment options.",
        ):
            self.assertEqual(par_exp.experiment_options, Options())
            self.assertEqual(par_exp.run_options, Options(meas_level=2))
            self.assertEqual(par_exp.transpile_options, Options(optimization_level=0))
            self.assertEqual(par_exp.analysis.options, Options())

            par_exp.run(FakeBackend())


class TestCompositeExperimentData(QiskitExperimentsTestCase):
    """
    Test operations on objects of composite ExperimentData
    """

    def setUp(self):
        super().setUp()

        self.backend = FakeBackend()
        self.share_level = "hey"

        exp1 = FakeExperiment([0, 2])
        exp2 = FakeExperiment([1, 3])
        par_exp = ParallelExperiment([exp1, exp2])
        exp3 = FakeExperiment([0, 1, 2, 3])
        batch_exp = BatchExperiment([par_exp, exp3])

        self.rootdata = batch_exp.run(backend=self.backend).block_for_results()
        self.assertEqual(len(self.rootdata.child_data()), 2)

        self.rootdata.share_level = self.share_level

    def check_attributes(self, expdata):
        """
        Recursively traverse the tree to verify attributes
        """
        self.assertEqual(expdata.backend, self.backend)
        self.assertEqual(expdata.share_level, self.share_level)

        components = expdata.child_data()
        for childdata in components:
            self.check_attributes(childdata)
            self.assertEqual(childdata.parent_id, expdata.experiment_id)

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
        metadata1.pop("child_data_ids", [])
        metadata2.pop("child_data_ids", [])
        self.assertDictEqual(metadata1, metadata2, msg="metadata not equal")

        if is_a_copy:
            self.assertNotEqual(expdata1.experiment_id, expdata2.experiment_id)
        else:
            self.assertEqual(expdata1.experiment_id, expdata2.experiment_id)

        self.assertEqual(len(expdata1.child_data()), len(expdata2.child_data()))
        for childdata1, childdata2 in zip(expdata1.child_data(), expdata2.child_data()):
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
        loaded_data = ExperimentData.load(self.rootdata.experiment_id, self.rootdata.service)
        self.check_if_equal(loaded_data, self.rootdata, is_a_copy=False)

    def test_composite_save_metadata(self):
        """
        Verify that saving metadata and loading restores the original composite experiment data object
        """
        self.rootdata.service = FakeService()
        self.rootdata.save_metadata()
        loaded_data = ExperimentData.load(self.rootdata.experiment_id, self.rootdata.service)

        self.check_if_equal(loaded_data, self.rootdata, is_a_copy=False)

    def test_composite_copy(self):
        """
        Test composite ExperimentData.copy
        """
        new_instance = self.rootdata.copy()
        self.check_if_equal(new_instance, self.rootdata, is_a_copy=True)
        self.check_attributes(new_instance)
        self.assertEqual(new_instance.parent_id, None)

    def test_analysis_replace_results_true(self):
        """
        Test replace results when analyzing composite experiment data
        """
        exp1 = FakeExperiment([0, 2])
        exp2 = FakeExperiment([1, 3])
        par_exp = ParallelExperiment([exp1, exp2])
        data1 = par_exp.run(FakeBackend()).block_for_results()

        # Additional data not part of composite experiment
        exp3 = FakeExperiment([0, 1])
        extra_data = exp3.run(FakeBackend())
        data1.add_child_data(extra_data)

        # Replace results
        data2 = par_exp.analysis.run(data1, replace_results=True)
        self.assertEqual(data1, data2)
        self.assertEqual(len(data1.child_data()), len(data2.child_data()))
        for sub1, sub2 in zip(data1.child_data(), data2.child_data()):
            self.assertEqual(sub1, sub2)

    def test_analysis_replace_results_false(self):
        """
        Test replace_results of composite experiment data
        """
        exp1 = FakeExperiment([0, 2])
        exp2 = FakeExperiment([1, 3])
        par_exp = BatchExperiment([exp1, exp2])
        data1 = par_exp.run(FakeBackend()).block_for_results()

        # Additional data not part of composite experiment
        exp3 = FakeExperiment([0, 1])
        extra_data = exp3.run(FakeBackend())
        data1.add_child_data(extra_data)

        # Replace results
        data2 = par_exp.analysis.run(data1, replace_results=False)
        self.assertNotEqual(data1.experiment_id, data2.experiment_id)
        self.assertEqual(len(data1.child_data()), len(data2.child_data()))
        for sub1, sub2 in zip(data1.child_data(), data2.child_data()):
            self.assertNotEqual(sub1.experiment_id, sub2.experiment_id)

    def test_composite_tags(self):
        """
        Test the tags setter, add_tags_recursive, remove_tags_recursive
        """
        exp1 = FakeExperiment([0, 2])
        exp2 = FakeExperiment([1, 3])
        par_exp = BatchExperiment([exp1, exp2])
        expdata = par_exp.run(FakeBackend()).block_for_results()
        data1 = expdata.child_data(0)
        data2 = expdata.child_data(1)

        expdata.tags = ["a", "c", "a"]
        data1.tags = ["b"]
        print(expdata.tags)
        self.assertEqual(sorted(expdata.tags), ["a", "c"])
        self.assertEqual(sorted(data1.tags), ["b"])
        self.assertEqual(sorted(data2.tags), [])

        expdata.add_tags_recursive(["d", "c"])
        self.assertEqual(sorted(expdata.tags), ["a", "c", "d"])
        self.assertEqual(sorted(data1.tags), ["b", "c", "d"])
        self.assertEqual(sorted(data2.tags), ["c", "d"])

        expdata.remove_tags_recursive(["a", "b"])
        self.assertEqual(sorted(expdata.tags), ["c", "d"])
        self.assertEqual(sorted(data1.tags), ["c", "d"])
        self.assertEqual(sorted(data2.tags), ["c", "d"])

    def test_composite_subexp_data(self):
        """
        Verify that sub-experiment data of parallel and batch
        experiments are correctly marginalized
        """
        counts = [
            {
                "0000": 1,
                "0010": 6,
                "0011": 3,
                "0100": 4,
                "0101": 2,
                "0110": 1,
                "0111": 3,
                "1000": 5,
                "1001": 3,
                "1010": 4,
                "1100": 4,
                "1101": 3,
                "1110": 8,
                "1111": 5,
            },
            {
                "0001": 3,
                "0010": 4,
                "0011": 5,
                "0100": 2,
                "0101": 1,
                "0111": 7,
                "1000": 3,
                "1001": 2,
                "1010": 1,
                "1011": 1,
                "1100": 7,
                "1101": 8,
                "1110": 2,
            },
            {
                "0000": 1,
                "0001": 1,
                "0010": 8,
                "0011": 7,
                "0100": 2,
                "0101": 2,
                "0110": 2,
                "0111": 1,
                "1000": 6,
                "1010": 4,
                "1011": 4,
                "1100": 5,
                "1101": 2,
                "1110": 2,
                "1111": 5,
            },
            {
                "0000": 4,
                "0001": 5,
                "0101": 4,
                "0110": 8,
                "0111": 2,
                "1001": 6,
                "1010": 8,
                "1011": 8,
                "1101": 1,
                "1110": 3,
                "1111": 3,
            },
            {
                "0000": 3,
                "0001": 6,
                "0010": 7,
                "0011": 1,
                "0100": 1,
                "0101": 5,
                "0110": 4,
                "1000": 2,
                "1001": 4,
                "1011": 3,
                "1100": 6,
                "1111": 1,
            },
        ]

        class Backend(FakeBackend):
            """
            Bacekend to be used in test_composite_subexp_data
            """

            def run(self, run_input, **options):
                results = []
                for circ, cnt in zip(run_input, counts):
                    results.append(
                        {
                            "shots": -1,
                            "success": True,
                            "header": {"metadata": circ.metadata},
                            "data": {"counts": cnt},
                        }
                    )

                res = {
                    "backend_name": "backend",
                    "backend_version": "0",
                    "qobj_id": uuid.uuid4().hex,
                    "job_id": uuid.uuid4().hex,
                    "success": True,
                    "results": results,
                }
                return FakeJob(backend=self, result=Result.from_dict(res))

        class Experiment(FakeExperiment):
            """
            Experiment to be used in test_composite_subexp_data
            """

            def __init__(self, qubits, num_circs):
                super().__init__(qubits)
                self._ncircs = num_circs

            def circuits(self):
                nqubits = len(self._physical_qubits)
                circs = []
                for _ in range(self._ncircs):
                    circ = QuantumCircuit(nqubits, nqubits)
                    circ.metadata = {}
                    circs.append(circ)
                return circs

        exp1 = Experiment([0, 2], 5)
        exp2 = Experiment([1], 2)
        exp3 = Experiment([3], 2)
        exp4 = Experiment([1, 3], 3)
        par_exp = ParallelExperiment(
            [exp1, BatchExperiment([ParallelExperiment([exp2, exp3]), exp4])]
        )
        expdata = par_exp.run(Backend()).block_for_results()

        self.assertEqual(len(expdata.data()), len(counts))
        for circ_data, circ_counts in zip(expdata.data(), counts):
            self.assertDictEqual(circ_data["counts"], circ_counts)

        counts1 = [
            [
                {"00": 14, "10": 19, "11": 11, "01": 8},
                {"01": 14, "10": 7, "11": 13, "00": 12},
                {"00": 14, "01": 5, "10": 16, "11": 17},
                {"00": 4, "01": 16, "10": 19, "11": 13},
                {"00": 12, "01": 15, "10": 11, "11": 5},
            ],
            [
                {"00": 10, "01": 10, "10": 12, "11": 20},
                {"00": 12, "01": 10, "10": 7, "11": 17},
                {"00": 17, "01": 7, "10": 14, "11": 14},
                {"00": 9, "01": 14, "10": 22, "11": 7},
                {"00": 17, "01": 10, "10": 9, "11": 7},
            ],
        ]

        self.assertEqual(len(expdata.child_data()), len(counts1))
        for childdata, child_counts in zip(expdata.child_data(), counts1):
            self.assertEqual(len(childdata.data()), len(child_counts))
            for circ_data, circ_counts in zip(childdata.data(), child_counts):
                self.assertDictEqual(circ_data["counts"], circ_counts)

        counts2 = [
            [{"00": 10, "01": 10, "10": 12, "11": 20}, {"00": 12, "01": 10, "10": 7, "11": 17}],
            [
                {"00": 17, "01": 7, "10": 14, "11": 14},
                {"00": 9, "01": 14, "10": 22, "11": 7},
                {"00": 17, "01": 10, "10": 9, "11": 7},
            ],
        ]

        self.assertEqual(len(expdata.child_data(1).child_data()), len(counts2))
        for childdata, child_counts in zip(expdata.child_data(1).child_data(), counts2):
            for circ_data, circ_counts in zip(childdata.data(), child_counts):
                self.assertDictEqual(circ_data["counts"], circ_counts)

        counts3 = [
            [{"0": 22, "1": 30}, {"0": 19, "1": 27}],
            [{"0": 20, "1": 32}, {"0": 22, "1": 24}],
        ]

        self.assertEqual(len(expdata.child_data(1).child_data(0).child_data()), len(counts3))
        for childdata, child_counts in zip(
            expdata.child_data(1).child_data(0).child_data(), counts3
        ):
            self.assertEqual(len(childdata.data()), len(child_counts))
            for circ_data, circ_counts in zip(childdata.data(), child_counts):
                self.assertDictEqual(circ_data["counts"], circ_counts)
