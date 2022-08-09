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

"""Test parameter guess functions."""
# pylint: disable=invalid-name

from test.base import QiskitExperimentsTestCase

import json
from ddt import ddt, data

import uncertainties

from qiskit_experiments.database_service.db_fitval import FitVal
from qiskit_experiments.database_service.utils import experiments_version
from qiskit_experiments.framework import ExperimentDecoder


@ddt
class TestFitVal(QiskitExperimentsTestCase):
    """Test for serialization."""

    __signal_value__ = [
        [3.0123, 0.001, "s"],
        [2, 0, "m"],
        [-5.678, 0.02, "g"],
        [1e3, 120, "Hz"],
        [-2.4e-3, 1e-3, "ab/cdef**2"],
        [-1.3e1, 0.36, "a.u."],
    ]

    def test_deprecation(self):
        """Test if fit val shows deprecation warning and being typecasted."""
        with self.assertWarns(FutureWarning):
            instance = FitVal(0.1, 0.2, unit="ab/cde**2")

        self.assertIsInstance(instance, uncertainties.core.Variable)

    @data(*__signal_value__)
    def test_can_load(self, val):
        """Test if we can still load cache data from old Qiskit Experiments."""
        value, stderr, unit = val

        # This is necessary because we cannot instantiate FitVal
        # Now FitVal is immediately typecasted to Variable before
        # the instance is created, i.e. __new__
        # This mimics the behavior of loading analysis result created with
        # old Qiskit Experiments.
        hard_coded_json_str = f"""
        {{
            "__type__": "object", 
            "__value__": {{
                "class": {{
                    "__type__": "type", 
                    "__value__": {{
                        "name": "FitVal", 
                        "module": "qiskit_experiments.database_service.db_fitval",
                        "version": "{experiments_version}"
                    }}
                }}, 
                "settings": {{
                    "value": {value}, 
                    "stderr": {stderr},
                    "unit": {f'"{unit}"' if unit else "null"}
                }}, 
                "version": "{experiments_version}"
            }}
        }}
        """
        with self.assertWarns(FutureWarning):
            loaded_val = json.loads(hard_coded_json_str, cls=ExperimentDecoder)

        self.assertIsInstance(loaded_val, uncertainties.core.Variable)
        self.assertEqual(loaded_val.nominal_value, value)
        self.assertEqual(loaded_val.std_dev, stderr)
        self.assertEqual(loaded_val.tag, unit)

    @data(*__signal_value__)
    def test_can_access(self, val):
        """Test if we can still use old properties."""
        with self.assertWarns(FutureWarning):
            value, stderr, unit = val
            val = FitVal(value=value, stderr=stderr, unit=unit)

        with self.assertWarns(DeprecationWarning):
            self.assertEqual(val.value, value)

        with self.assertWarns(DeprecationWarning):
            self.assertEqual(val.stderr, stderr)

        with self.assertWarns(DeprecationWarning):
            self.assertEqual(val.unit, unit)
