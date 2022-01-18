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

        num = 0
        for experiment_type in range(2):
            for backend_name in range(2):
                for tags in range(2):
                    entry = self.generate_entry(experiment_type, backend_name, tags, num)
                    self.service.create_experiment(**entry)
                    self.expdict[str(num)] = entry
                    num += 1

    @staticmethod
    def generate_entry(experiment_type, backend_name, tags, num):
        entry = {"experiment_id": str(num), "experiment_type": str(experiment_type), "backend_name": str(backend_name), "tags": ["a"+str(tags), "b"+str(tags)]}
        if num>2:
            entry["parent_id"] = str(num%3)
        return entry
                        
    def test_create_experiment(self):
        self.assertEqual(len(self.service.exps), 8)
        is_in_frame = []
        for i in range(len(self.service.exps)):
            full_entry = self.service.exps.loc[i, :].to_dict()
            self.assertTrue(full_entry["experiment_id"] not in is_in_frame)
            is_in_frame.append(full_entry["experiment_id"])
            self.assertTrue(full_entry["experiment_id"] in self.expdict.keys())
            entry = self.expdict[full_entry["experiment_id"]]
            self.assertTrue(entry.items() <= full_entry.items())

    def test_single_experiment_query(self):
        pass
