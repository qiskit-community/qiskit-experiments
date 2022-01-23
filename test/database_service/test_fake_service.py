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
                    expentry = {"experiment_id": str(expid), "experiment_type": str(experiment_type), "backend_name": str(backend_name), "tags": ["a"+str(tags), "b"+str(tags)]}
                    if expid>2:
                        expentry["parent_id"] = str(expid%3)
                    self.service.create_experiment(**expentry)
                    self.expdict[str(expid)] = expentry
                    expid += 1

        self.resdict = {}
        resid = 0
        for experiment_id in [0, 1, 6, 7]:
            for result_type in range(2):
                for tags in range(2):
                    for quality in [True, False]:
                        for verified in [True, False]:
                            for result_data in range(2):
                                resentry = {"experiment_id": str(experiment_id), "result_type": str(result_type), "result_id": str(resid), "tags": ["a"+str(tags), "b"+str(tags)], "quality": quality, "verified": verified, "result_data": {"value": result_data}}
                                self.service.create_analysis_result(**resentry)
                                self.resdict[str(resid)] = resentry
                                resid += 1                                
                                
    def test_create_experiment(self):
        self.assertEqual(len(self.service.exps), 8)
        is_in_frame = []
        for i in range(8):
            full_entry = self.service.exps.loc[i, :].to_dict()
            expid = full_entry["experiment_id"]
            self.assertTrue(expid not in is_in_frame)
            is_in_frame.append(expid)
            self.assertTrue(expid in self.expdict.keys())
            entry = self.expdict[expid]
            self.assertTrue(entry.items() <= full_entry.items())

    def test_single_experiment_query(self):
        for expid in range(8):
            full_entry = self.service.experiment(str(expid))
            entry = self.expdict[str(expid)]
            self.assertTrue(entry.items() <= full_entry.items())

    def test_create_analysis_result(self):
        self.assertEqual(len(self.service.results), 128)
        is_in_frame = []
        for i in range(128):
            full_entry = self.service.results.loc[i, :].to_dict()
            resid = full_entry["result_id"]
            self.assertTrue(resid not in is_in_frame)
            is_in_frame.append(resid)
            self.assertTrue(resid in self.resdict.keys())
            entry = self.resdict[resid]
            self.assertTrue(entry.items() <= full_entry.items())
