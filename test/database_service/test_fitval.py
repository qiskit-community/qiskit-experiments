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
from ddt import ddt, data
from qiskit_experiments.database_service.db_fitval import FitVal


@ddt
class TestFitVal(QiskitExperimentsTestCase):
    """Test for serialization."""

    __signle_value__ = [
        [3.0123, 0.001, "s"],
        [2, 0, "m"],
        [-5.678, 0.02, "g"],
        [1e3, 120, "Hz"],
        [-2.4e-3, 1e-3, "ab/cdef**2"],
        [-1.3e1, 0.36, "a.u."],
    ]

    @data(*__signle_value__)
    def test_serialize(self, val):
        """Test serialization of data."""
        val = FitVal(*val)
        self.assertRoundTripSerializable(val)

    @data(*__signle_value__)
    def test_str(self, val):
        """Test str."""
        v = FitVal(*val)
        ret = str(v)

        self.assertEqual(ret, f"{val[0]} \u00B1 {val[1]} {val[2]}")
