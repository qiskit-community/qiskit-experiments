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

import json
import numpy as np
from ddt import ddt, data, unpack
from qiskit.test import QiskitTestCase

from qiskit_experiments.database_service.fit_value_type import fitval
from qiskit_experiments.database_service.json import ExperimentEncoder, ExperimentDecoder


@ddt
class TestBinOperFitValAndFloat(QiskitTestCase):
    """Test binary operator for fitval and float value."""

    __two_values__ = [
        [(1.234, 0.567), 1.234],
        [(-0.12, 1.365), -0.2],
        [(1e5, 0.3e2), 1e3],
        [(-6.1e6, 0.34e4), -3e6],
        [(3, 0.2), 3],
    ]

    @data(*__two_values__)
    @unpack
    def test_add(self, val1, val2):
        """Test adding two fit values."""
        v1 = fitval(*val1)
        ret = v1 + val2

        ref_val = val1[0] + val2

        self.assertIsInstance(ret, float)
        self.assertEqual(ret, ref_val)

    @data(*__two_values__)
    @unpack
    def test_sub(self, val1, val2):
        """Test subtracting two fit values."""
        v1 = fitval(*val1)
        ret = v1 - val2

        ref_val = val1[0] - val2

        self.assertIsInstance(ret, float)
        self.assertEqual(ret, ref_val)

    @data(*__two_values__)
    @unpack
    def test_mul(self, val1, val2):
        """Test multiplying two fit values."""
        v1 = fitval(*val1)
        ret = v1 * val2

        ref_val = val1[0] * val2

        self.assertIsInstance(ret, float)
        self.assertEqual(ret, ref_val)

    @data(*__two_values__)
    @unpack
    def test_div(self, val1, val2):
        """Test dividing two fit values."""
        v1 = fitval(*val1)
        ret = v1 / val2

        ref_val = val1[0] / val2

        self.assertIsInstance(ret, float)
        self.assertEqual(ret, ref_val)

    @data(*__two_values__)
    @unpack
    def test_ge(self, val1, val2):
        """Test comparison ge: >=."""
        v1 = fitval(*val1)

        self.assertEqual(v1 >= val2, val1[0] >= val2)

    @data(*__two_values__)
    @unpack
    def test_le(self, val1, val2):
        """Test comparison le: <=."""
        v1 = fitval(*val1)

        self.assertEqual(v1 <= val2, val1[0] <= val2)

    @data(*__two_values__)
    @unpack
    def test_gt(self, val1, val2):
        """Test comparison gt: >."""
        v1 = fitval(*val1)

        self.assertEqual(v1 > val2, val1[0] > val2)

    @data(*__two_values__)
    @unpack
    def test_lt(self, val1, val2):
        """Test comparison lt: <."""
        v1 = fitval(*val1)

        self.assertEqual(v1 < val2, val1[0] < val2)

    @data(*__two_values__)
    @unpack
    def test_eq(self, val1, val2):
        """Test comparison eq: ==."""
        v1 = fitval(*val1)

        self.assertEqual(v1 == val2, val1[0] == val2)


@ddt
class TestBinOperFitValAndFitVal(QiskitTestCase):
    """Test binary operator for fitval and fitval."""

    __two_values__ = [
        [(3.0123, 0.001), (6.3, 0.13)],
        [(2, 0), (-4.05, 0.2)],
        [(-5.678, 0.02), (-138, 0.5)],
        [(1e3, 120), (-5e3, 1.5)],
        [(-2.4e-3, 1e-3), (6e-1, 1.3e-2)],
        [(-1.3e1, 0.36), (-1.3e1, 0.36)],
    ]

    @data(*__two_values__)
    @unpack
    def test_add(self, val1, val2):
        """Test adding two fit values."""
        v1 = fitval(*val1)
        v2 = fitval(*val2)
        ret = v1 + v2

        ref_val = val1[0] + val2[0]
        ref_std = np.sqrt(val1[1] ** 2 + val2[1] ** 2)

        self.assertEqual(ret.value, ref_val)
        self.assertEqual(ret.stdev, ref_std)

    @data(*__two_values__)
    @unpack
    def test_sub(self, val1, val2):
        """Test subtracting two fit values."""
        v1 = fitval(*val1)
        v2 = fitval(*val2)
        ret = v1 - v2

        ref_val = val1[0] - val2[0]
        ref_std = np.sqrt(val1[1] ** 2 + val2[1] ** 2)

        self.assertEqual(ret.value, ref_val)
        self.assertEqual(ret.stdev, ref_std)

    @data(*__two_values__)
    @unpack
    def test_mul(self, val1, val2):
        """Test multiplying two fit values."""
        v1 = fitval(*val1)
        v2 = fitval(*val2)
        ret = v1 * v2

        ref_val = val1[0] * val2[0]
        ref_std = np.sqrt((val2[0] * val1[1]) ** 2 + (val1[0] * val2[1]) ** 2)

        self.assertEqual(ret.value, ref_val)
        self.assertEqual(ret.stdev, ref_std)

    @data(*__two_values__)
    @unpack
    def test_div(self, val1, val2):
        """Test dividing two fit values."""
        v1 = fitval(*val1)
        v2 = fitval(*val2)
        ret = v1 / v2

        ref_val = val1[0] / val2[0]
        ref_std = np.sqrt((val1[1] / val2[0]) ** 2 + (val2[1] * (val1[0] / val2[0] ** 2)) ** 2)

        self.assertEqual(ret.value, ref_val)
        self.assertEqual(ret.stdev, ref_std)

    @data(*__two_values__)
    @unpack
    def test_ge(self, val1, val2):
        """Test comparison ge: >=."""
        v1 = fitval(*val1)
        v2 = fitval(*val2)

        self.assertEqual(v1 >= v2, val1[0] >= val2[0])

    @data(*__two_values__)
    @unpack
    def test_le(self, val1, val2):
        """Test comparison le: <=."""
        v1 = fitval(*val1)
        v2 = fitval(*val2)

        self.assertEqual(v1 <= v2, val1[0] <= val2[0])

    @data(*__two_values__)
    @unpack
    def test_gt(self, val1, val2):
        """Test comparison gt: >."""
        v1 = fitval(*val1)
        v2 = fitval(*val2)

        self.assertEqual(v1 > v2, val1[0] > val2[0])

    @data(*__two_values__)
    @unpack
    def test_lt(self, val1, val2):
        """Test comparison lt: <."""
        v1 = fitval(*val1)
        v2 = fitval(*val2)

        self.assertEqual(v1 < v2, val1[0] < val2[0])

    @data(*__two_values__)
    @unpack
    def test_eq(self, val1, val2):
        """Test comparison eq: ==."""
        v1 = fitval(*val1)
        v2 = fitval(*val2)

        self.assertEqual(v1 == v2, val1[0] == val2[0] and val1[1] == val2[1])


@ddt
class TestBinOperFitValAndFitValNoStdev(QiskitTestCase):
    """Test binary operator for fitval and fitval without stdev."""

    __two_values__ = [
        [(3.0123, 0.001), (6.3,)],
        [(2, 0), (-4.05,)],
        [(-5.678, 0.02), (-138,)],
        [(1e3, 120), (-5e3,)],
        [(-2.4e-3, 1e-3), (6e-1,)],
        [(-1.3e1, 0.36), (-1.3e1,)],
    ]

    @data(*__two_values__)
    @unpack
    def test_add(self, val1, val2):
        """Test adding two fit values."""
        v1 = fitval(*val1)
        v2 = fitval(*val2)
        ret = v1 + v2

        ref_val = val1[0] + val2[0]

        self.assertIsInstance(ret, float)
        self.assertEqual(ret, ref_val)

    @data(*__two_values__)
    @unpack
    def test_sub(self, val1, val2):
        """Test subtracting two fit values."""
        v1 = fitval(*val1)
        v2 = fitval(*val2)
        ret = v1 - v2

        ref_val = val1[0] - val2[0]

        self.assertIsInstance(ret, float)
        self.assertEqual(ret, ref_val)

    @data(*__two_values__)
    @unpack
    def test_mul(self, val1, val2):
        """Test multiplying two fit values."""
        v1 = fitval(*val1)
        v2 = fitval(*val2)
        ret = v1 * v2

        ref_val = val1[0] * val2[0]

        self.assertIsInstance(ret, float)
        self.assertEqual(ret, ref_val)

    @data(*__two_values__)
    @unpack
    def test_div(self, val1, val2):
        """Test dividing two fit values."""
        v1 = fitval(*val1)
        v2 = fitval(*val2)
        ret = v1 / v2

        ref_val = val1[0] / val2[0]

        self.assertIsInstance(ret, float)
        self.assertEqual(ret, ref_val)

    @data(*__two_values__)
    @unpack
    def test_ge(self, val1, val2):
        """Test comparison ge: >=."""
        v1 = fitval(*val1)
        v2 = fitval(*val2)

        self.assertEqual(v1 >= v2, val1[0] >= val2[0])

    @data(*__two_values__)
    @unpack
    def test_le(self, val1, val2):
        """Test comparison le: <=."""
        v1 = fitval(*val1)
        v2 = fitval(*val2)

        self.assertEqual(v1 <= v2, val1[0] <= val2[0])

    @data(*__two_values__)
    @unpack
    def test_gt(self, val1, val2):
        """Test comparison gt: >."""
        v1 = fitval(*val1)
        v2 = fitval(*val2)

        self.assertEqual(v1 > v2, val1[0] > val2[0])

    @data(*__two_values__)
    @unpack
    def test_lt(self, val1, val2):
        """Test comparison lt: <."""
        v1 = fitval(*val1)
        v2 = fitval(*val2)

        self.assertEqual(v1 < v2, val1[0] < val2[0])

    @data(*__two_values__)
    @unpack
    def test_eq(self, val1, val2):
        """Test comparison eq: ==."""
        v1 = fitval(*val1)
        v2 = fitval(*val2)

        self.assertEqual(v1 == v2, val1[0] == val2[0])


@ddt
class TestUnaryOpers(QiskitTestCase):
    """Test for unary operator."""

    __signle_value__ = [
        [3.0123, 0.001, "s"],
        [2, 0, "m"],
        [-5.678, 0.02, "g"],
        [1e3, 120, "Hz"],
        [-2.4e-3, 1e-3, "ab/cdef**2"],
        [-1.3e1, 0.36, "a.u."],
    ]

    @data(*__signle_value__)
    def test_abs(self, val):
        """Test abs."""
        v = fitval(*val)
        ret = abs(v)

        self.assertEqual(ret.value, abs(val[0]))
        self.assertEqual(ret.stdev, val[1])

    @data(*__signle_value__)
    def test_pos(self, val):
        """Test pos."""
        v = fitval(*val)
        ret = +v

        self.assertEqual(ret.value, val[0])
        self.assertEqual(ret.stdev, val[1])

    @data(*__signle_value__)
    def test_neg(self, val):
        """Test neg."""
        v = fitval(*val)
        ret = -v

        self.assertEqual(ret.value, -val[0])
        self.assertEqual(ret.stdev, val[1])

    @data(*__signle_value__)
    def test_float(self, val):
        """Test pos."""
        v = fitval(*val)
        ret = float(v)

        self.assertEqual(ret, val[0])

    @data(*__signle_value__)
    def test_repr(self, val):
        """Test pos."""
        v = fitval(*val)
        ret = repr(v)

        self.assertEqual(ret, f"fitval(value={val[0]}, stdev={val[1]}, unit={val[2]})")

    @data(*__signle_value__)
    def test_str(self, val):
        """Test pos."""
        v = fitval(*val)
        ret = str(v)

        self.assertEqual(ret, f"{val[0]}\u00B1{val[1]} [{val[2]}]")


@ddt
class TestSerialize(QiskitTestCase):
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
        val_orig = fitval(*val)

        ser = json.dumps(val_orig, cls=ExperimentEncoder)
        val_deser = json.loads(ser, cls=ExperimentDecoder)

        self.assertEqual(val_orig, val_deser)


class TestFitValInvalidOpers(QiskitTestCase):
    """Test for invalid situations."""

    def test_cannot_create_with_non_real_value(self):
        """Test create with non-real value."""
        with self.assertRaises(TypeError):
            fitval(1j, 0.1)

    def test_cannot_create_with_non_real_stdev(self):
        """Test create with non-real stdev."""
        with self.assertRaises(TypeError):
            fitval(0.3, 1j)

    def test_cannot_create_with_negative_stdev(self):
        """Test create with negative stdev."""
        with self.assertRaises(ValueError):
            fitval(0.3, -0.1)

    def test_cannot_oper_different_unit(self):
        """Test operation with different units."""
        v1 = fitval(1.2, 0.34, "s")
        v2 = fitval(3.4, 0.56, "m")

        # pylint: disable=pointless-statement
        with self.assertRaises(ValueError):
            v1 + v2
