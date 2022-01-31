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
Test the fake service
"""

from datetime import datetime
from test.base import QiskitExperimentsTestCase
from qiskit_experiments.test.fake_service import FakeService


class TestFakeService(QiskitExperimentsTestCase):
    """
    Test the fake service
    """

    def setUp(self):
        super().setUp()

        self.service = FakeService()

        self.expdict = {}
        expid = 0
        for experiment_type in range(2):
            for backend_name in range(2):
                for tags in range(2):
                    expentry = {
                        "experiment_id": str(expid),
                        "experiment_type": str(experiment_type),
                        "backend_name": str(backend_name),
                        "tags": ["a" + str(tags), "b" + str(tags)],
                    }
                    if expid > 2:
                        expentry["parent_id"] = str(expid % 3)
                    else:
                        expentry["parent_id"] = None
                    self.service.create_experiment(**expentry)
                    if expid in [0, 1, 6, 7]:
                        expentry["device_components"] = [0, 1]
                    else:
                        expentry["device_components"] = []
                    expentry["start_datetime"] = datetime(2022, 1, 1, expid)
                    self.expdict[str(expid)] = expentry
                    expid += 1

        self.resdict = {}
        resid = 0
        for experiment_id in [0, 1, 6, 7]:
            for result_type in range(2):
                for samebool4all in range(2):
                    tags = samebool4all
                    quality = samebool4all
                    verified = samebool4all
                    result_data = samebool4all
                    device_components = samebool4all
                    resentry = {
                        "experiment_id": str(experiment_id),
                        "result_type": str(result_type),
                        "result_id": str(resid),
                        "tags": ["a" + str(tags), "b" + str(tags)],
                        "quality": quality,
                        "verified": verified,
                        "result_data": {"value": result_data},
                        "device_components": [device_components],
                        "creation_datetime": self.expdict[str(experiment_id)]["start_datetime"],
                    }
                    self.service.create_analysis_result(**resentry)
                    resentry["backend_name"] = self.expdict[str(experiment_id)]["backend_name"]
                    self.resdict[str(resid)] = resentry
                    resid += 1

    def test_creation(self):
        for df, reference_dict, id_field in zip(
            [self.service.exps, self.service.results],
            [self.expdict, self.resdict],
            ["experiment_id", "result_id"],
        ):
            self.assertEqual(len(df), len(reference_dict))
            is_in_frame = []
            for i in range(len(df)):
                full_entry = df.loc[i, :].to_dict()
                id = full_entry[id_field]
                self.assertTrue(id not in is_in_frame)
                is_in_frame.append(id)
                self.assertTrue(id in reference_dict.keys())
                entry = reference_dict[id]
                self.assertTrue(entry.items() <= full_entry.items())

    def test_single_experiment_query(self):
        for expid in range(8):
            full_entry = self.service.experiment(str(expid))
            entry = self.expdict[str(expid)]
            self.assertTrue(entry.items() <= full_entry.items())

    def test_experiments_query(self):
        for experiment_type in range(2):
            expids = sorted(
                [
                    exp["experiment_id"]
                    for exp in self.service.experiments(
                        experiment_type=str(experiment_type), limit=None
                    )
                ]
            )
            ref_expids = sorted(
                [
                    exp["experiment_id"]
                    for exp in self.expdict.values()
                    if exp["experiment_type"] == str(experiment_type)
                ]
            )
            self.assertTrue(len(expids) > 0)
            self.assertEqual(expids, ref_expids)

        for backend_name in range(2):
            expids = sorted(
                [
                    exp["experiment_id"]
                    for exp in self.service.experiments(backend_name=str(backend_name), limit=None)
                ]
            )
            ref_expids = sorted(
                [
                    exp["experiment_id"]
                    for exp in self.expdict.values()
                    if exp["backend_name"] == str(backend_name)
                ]
            )
            self.assertTrue(len(expids) > 0)
            self.assertEqual(expids, ref_expids)

        for parent_id in range(3):
            expids = sorted(
                [
                    exp["experiment_id"]
                    for exp in self.service.experiments(parent_id=str(parent_id), limit=None)
                ]
            )
            ref_expids = sorted(
                [
                    exp["experiment_id"]
                    for exp in self.expdict.values()
                    if exp["parent_id"] == str(parent_id)
                ]
            )
            self.assertTrue(len(expids) > 0)
            self.assertEqual(expids, ref_expids)

        expids = sorted(
            [
                exp["experiment_id"]
                for exp in self.service.experiments(
                    tags=["a1", "b1"], tags_operator="AND", limit=None
                )
            ]
        )
        ref_expids = sorted(
            [
                exp["experiment_id"]
                for exp in self.expdict.values()
                if "a1" in exp["tags"] and "b1" in exp["tags"]
            ]
        )
        self.assertTrue(len(expids) > 0)
        self.assertEqual(expids, ref_expids)

        expids = sorted(
            [
                exp["experiment_id"]
                for exp in self.service.experiments(
                    tags=["a1", "c1"], tags_operator="AND", limit=None
                )
            ]
        )
        self.assertEqual(len(expids), 0)

        expids = sorted(
            [
                exp["experiment_id"]
                for exp in self.service.experiments(tags=["a0", "c0"], limit=None)
            ]
        )
        ref_expids = sorted(
            [exp["experiment_id"] for exp in self.expdict.values() if "a0" in exp["tags"]]
        )
        self.assertTrue(len(expids) > 0)
        self.assertEqual(expids, ref_expids)

        expids = sorted(
            [
                exp["experiment_id"]
                for exp in self.service.experiments(
                    start_datetime_before=datetime(2022, 1, 1, 6),
                    start_datetime_after=datetime(2022, 1, 1, 3),
                    limit=None,
                )
            ]
        )
        self.assertEqual(expids, ["3", "4", "5", "6"])

        datetimes = [exp["start_datetime"] for exp in self.service.experiments(limit=None)]
        self.assertTrue(len(datetimes) > 0)
        for i in range(len(datetimes) - 1):
            self.assertTrue(datetimes[i] >= datetimes[i + 1])

        datetimes = [
            exp["start_datetime"]
            for exp in self.service.experiments(sort_by="start_datetime:asc", limit=None)
        ]
        self.assertTrue(len(datetimes) > 0)
        for i in range(len(datetimes) - 1):
            self.assertTrue(datetimes[i] <= datetimes[i + 1])

        self.assertEqual(len(self.service.experiments(limit=4)), 4)

    def test_update_experiment(self):
        self.service.update_experiment(experiment_id="1", metadata="hey", notes="hi")
        exp = self.service.experiment(experiment_id="1")
        self.assertEqual(exp["metadata"], "hey")
        self.assertEqual(exp["notes"], "hi")

    def test_delete_experiment(self):
        exps = self.service.experiments(
            start_datetime_before=datetime(2022, 1, 1, 2),
            start_datetime_after=datetime(2022, 1, 1, 2),
        )
        self.assertEqual(len(exps), 1)
        self.service.delete_experiment(experiment_id="2")
        exps = self.service.experiments(
            start_datetime_before=datetime(2022, 1, 1, 2),
            start_datetime_after=datetime(2022, 1, 1, 2),
        )
        self.assertEqual(len(exps), 0)

    def test_single_result_query(self):
        for resid in range(16):
            full_entry = self.service.analysis_result(str(resid))
            entry = self.resdict[str(resid)]
            self.assertTrue(entry.items() <= full_entry.items())

    def test_update_result(self):
        self.service.update_analysis_result(result_id="1", tags=["hey"])
        res = self.service.analysis_result(result_id="1")
        self.assertEqual(res["tags"], "hey")

    def test_results_query(self):
        for result_type in range(2):
            resids = sorted(
                [
                    res["result_id"]
                    for res in self.service.analysis_results(
                        result_type=str(result_type), limit=None
                    )
                ]
            )
            ref_resids = sorted(
                [
                    res["result_id"]
                    for res in self.resdict.values()
                    if res["result_type"] == str(result_type)
                ]
            )
            self.assertTrue(len(resids) > 0)
            self.assertEqual(resids, ref_resids)

        for experiment_id in range(2):
            resids = sorted(
                [
                    res["result_id"]
                    for res in self.service.analysis_results(
                        experiment_id=str(experiment_id), limit=None
                    )
                ]
            )
            ref_resids = sorted(
                [
                    res["result_id"]
                    for res in self.resdict.values()
                    if res["experiment_id"] == str(experiment_id)
                ]
            )
            self.assertTrue(len(resids) > 0)
            self.assertEqual(resids, ref_resids)

        for quality in range(2):
            resids = sorted(
                [
                    res["result_id"]
                    for res in self.service.analysis_results(quality=quality, limit=None)
                ]
            )
            ref_resids = sorted(
                [res["result_id"] for res in self.resdict.values() if res["quality"] == quality]
            )
            self.assertTrue(len(resids) > 0)
            self.assertEqual(resids, ref_resids)

        for verified in range(2):
            resids = sorted(
                [
                    res["result_id"]
                    for res in self.service.analysis_results(verified=verified, limit=None)
                ]
            )
            ref_resids = sorted(
                [res["result_id"] for res in self.resdict.values() if res["verified"] == verified]
            )
            self.assertTrue(len(resids) > 0)
            self.assertEqual(resids, ref_resids)

        for backend_name in range(2):
            resids = sorted(
                [
                    res["result_id"]
                    for res in self.service.analysis_results(
                        backend_name=str(backend_name), limit=None
                    )
                ]
            )
            ref_resids = sorted(
                [
                    res["result_id"]
                    for res in self.resdict.values()
                    if res["backend_name"] == str(backend_name)
                ]
            )
            self.assertTrue(len(resids) > 0)
            self.assertEqual(resids, ref_resids)

        resids = sorted(
            [
                res["result_id"]
                for res in self.service.analysis_results(
                    tags=["a1", "b1"], tags_operator="AND", limit=None
                )
            ]
        )
        ref_resids = sorted(
            [
                res["result_id"]
                for res in self.resdict.values()
                if "a1" in res["tags"] and "b1" in res["tags"]
            ]
        )
        self.assertTrue(len(resids) > 0)
        self.assertEqual(resids, ref_resids)

        resids = sorted(
            [
                res["result_id"]
                for res in self.service.analysis_results(
                    tags=["a1", "c1"], tags_operator="AND", limit=None
                )
            ]
        )
        self.assertEqual(len(resids), 0)

        resids = sorted(
            [
                res["result_id"]
                for res in self.service.analysis_results(tags=["a0", "c0"], limit=None)
            ]
        )
        ref_resids = sorted(
            [res["result_id"] for res in self.resdict.values() if "a0" in res["tags"]]
        )
        self.assertTrue(len(resids) > 0)
        self.assertEqual(resids, ref_resids)

        datetimes = [res["creation_datetime"] for res in self.service.analysis_results(limit=None)]
        self.assertTrue(len(datetimes) > 0)
        for i in range(len(datetimes) - 1):
            self.assertTrue(datetimes[i] >= datetimes[i + 1])

        datetimes = [
            res["creation_datetime"]
            for res in self.service.analysis_results(sort_by="creation_datetime:asc", limit=None)
        ]
        self.assertTrue(len(datetimes) > 0)
        for i in range(len(datetimes) - 1):
            self.assertTrue(datetimes[i] <= datetimes[i + 1])

        self.assertEqual(len(self.service.analysis_results(limit=4)), 4)

    def test_delete_result(self):
        results = self.service.analysis_results(experiment_id="6")
        old_number = len(results)
        to_delete = results[0]["result_id"]
        self.service.delete_analysis_result(result_id=to_delete)
        results = self.service.analysis_results(experiment_id="6")
        self.assertEqual(len(results), old_number - 1)
