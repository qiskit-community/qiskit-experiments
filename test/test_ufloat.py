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

"""
A Tester for Experiment ufloat
"""

from test.base import QiskitExperimentsTestCase

import copy
import json

import numpy as np
from ddt import ddt, data, unpack
from uncertainties import umath, unumpy, correlated_values

from qiskit_experiments.framework import (
    ExperimentVariable,
    ExperimentEncoder,
    ExperimentDecoder,
)


@ddt
class TestErrorPropagationComputation(QiskitExperimentsTestCase):
    """Test for error propagation computation.

    Since Qiskit Experiments monkey patches the uncertainties package,
    this test checks if computed results are equivalent to the result of
    one computed with the pure uncertainties package, or with propagation formula.
    """

    @data(
        [0.1, 0.2, 0.3, 0.4],
        [0.1, 0.2, 0.1, 0.2],
        [-0.3, 0.4, 0.5, 0.1],
    )
    @unpack
    def test_sum(self, v1n, v1s, v2n, v2s):
        """Test if error propagation with summation works."""
        v1 = ExperimentVariable(v1n, v1s)
        v2 = ExperimentVariable(v2n, v2s)

        res = v1 + v2
        self.assertAlmostEqual(res.n, v1n + v2n)
        self.assertAlmostEqual(res.s, np.sqrt(v1s**2 + v2s**2))

    @data(
        [0.1, 0.2, 0.3, 0.4],
        [0.1, 0.2, 0.1, 0.2],
        [-0.3, 0.4, 0.5, 0.1],
    )
    @unpack
    def test_sub(self, v1n, v1s, v2n, v2s):
        """Test if error propagation with subtraction works."""
        v1 = ExperimentVariable(v1n, v1s)
        v2 = ExperimentVariable(v2n, v2s)

        res = v1 - v2
        self.assertAlmostEqual(res.n, v1n - v2n)
        self.assertAlmostEqual(res.s, np.sqrt(v1s**2 + v2s**2))

    @data(
        [0.1, 0.2, 0.3, 0.4],
        [0.1, 0.2, 0.1, 0.2],
        [-0.3, 0.4, 0.5, 0.1],
    )
    @unpack
    def test_mul(self, v1n, v1s, v2n, v2s):
        """Test if error propagation with multiplication works."""
        v1 = ExperimentVariable(v1n, v1s)
        v2 = ExperimentVariable(v2n, v2s)

        res = v1 * v2
        self.assertAlmostEqual(res.n, v1n * v2n)
        self.assertAlmostEqual(res.s, np.sqrt((v2n * v1s) ** 2 + (v1n * v2s) ** 2))

    @data(
        [0.1, 0.2, 0.3, 0.4],
        [0.1, 0.2, 0.1, 0.2],
        [-0.3, 0.4, 0.5, 0.1],
    )
    @unpack
    def test_div(self, v1n, v1s, v2n, v2s):
        """Test if error propagation with division works."""
        v1 = ExperimentVariable(v1n, v1s)
        v2 = ExperimentVariable(v2n, v2s)

        res = v1 / v2
        self.assertAlmostEqual(res.n, v1n / v2n)
        self.assertAlmostEqual(res.s, np.sqrt((v1s / v2n) ** 2 + (v1n / v2n**2 * v2s) ** 2))

    @data(
        [1.57, 0.01],
        [0.78, 0.03],
        [-0.46, 0.02],
    )
    @unpack
    def test_func(self, nom, std):
        """Test if error propagation with function works."""
        res = umath.sin(ExperimentVariable(nom, std))
        self.assertAlmostEqual(res.n, np.sin(nom))
        self.assertAlmostEqual(res.s, np.cos(nom) * std)

    @data(
        [[1.57, -1.23], [0.03, 0.05]],
        [[0.78, 0.78], [0.02, 0.05]],
        [[-0.56, 0.32], [0.03, 0.04]],
    )
    @unpack
    def test_func_vectorized(self, noms, stds):
        """Test if error propagation with function taking vector works.

        Note that ufloat value vector generator is defined in the uncertainties package.
        Return type is also checked to guarantee the packages is monkey patched.
        """
        # This should return array of ExperimentVariables since its monkey patched
        ufloats = unumpy.uarray(noms, stds)
        self.assertIsInstance(ufloats[0], ExperimentVariable)
        self.assertIsInstance(ufloats[1], ExperimentVariable)

        res_v = unumpy.sin(ufloats)
        res_ns = unumpy.nominal_values(res_v)
        res_ss = unumpy.std_devs(res_v)

        np.testing.assert_array_almost_equal(res_ns, np.sin(noms))
        np.testing.assert_array_almost_equal(res_ss, np.cos(noms) * stds)

    def test_correlated_propagation(self):
        """Check if propagation can consider parameter correlation.

        The reference values are computed by the original uncertainties package and hardcoded.
        """
        v1 = ExperimentVariable(0.8, 0.3)
        v2 = ExperimentVariable(0.3, 0.5)
        v3 = ExperimentVariable(-0.4, 0.2)

        v4 = v1 + v2
        v5 = v3 - v2

        # uncertainty from V2 should be correlated
        res = v4 * v5

        self.assertAlmostEqual(res.n, -0.77)
        self.assertAlmostEqual(res.s, 0.95)

    def test_non_correlated_propagation(self):
        """Check if propagation can consider parameter correlation.

        The reference values are computed by the original uncertainties package and hardcoded.
        """
        v1 = ExperimentVariable(0.8, 0.3)
        v2a = ExperimentVariable(0.3, 0.5)
        v2b = ExperimentVariable(0.3, 0.5)
        v3 = ExperimentVariable(-0.4, 0.2)

        v4 = v1 + v2a
        v5 = v3 - v2b

        # uncertainty from V2 should NOT be correlated
        res = v4 * v5

        self.assertAlmostEqual(res.n, -0.77)
        self.assertAlmostEqual(res.s, 0.719374728496908)

    def test_very_complicated(self):
        """Check if propagation is properly computed within some complicated math operation.

        The reference values are computed by the original uncertainties package and hardcoded.
        """
        cov_mat = np.array(
            [
                [0.06920779, 0.17794139, 0.22187949],
                [0.46286943, 0.98551139, 0.08438406],
                [0.13404688, 0.59558274, 0.00957874],
            ]
        )
        noms = np.array([0.1, -0.5, 0.3])
        correlated_vals = correlated_values(noms, covariance_mat=cov_mat)

        v1 = umath.exp(correlated_vals[0]) + umath.cos(correlated_vals[1]) ** correlated_vals[2]
        v2 = correlated_vals[0] / correlated_vals[1] * correlated_vals[2]

        res = v1 - v2

        self.assertAlmostEqual(res.n, 2.1267530738954776)
        self.assertAlmostEqual(res.s, 1.2017366800495946)


@ddt
class TestExtraFunctionality(QiskitExperimentsTestCase):
    """This test checks if extra functionalities added by monkey patch works as expected."""

    def test_variable_can_have_unit(self):
        """Test if experiment uflot can take unit and show it in repr."""
        val = ExperimentVariable(0.1, 0.2, tag="f", unit="Hz")
        self.assertEqual(val.n, 0.1)
        self.assertEqual(val.s, 0.2)
        self.assertEqual(val.tag, "f")
        self.assertEqual(val.unit, "Hz")
        self.assertEqual(repr(val), "< f = 0.1+/-0.2 Hz >")

    def test_affine_func_can_have_unit(self):
        """Test if operated result can take unit and show it in repr."""
        val1 = ExperimentVariable(0.1, 0.2)
        val2 = ExperimentVariable(0.3, 0.4)

        res = val1 + val2
        res.unit = "abc/def**2"

        self.assertEqual(repr(res), "0.4+/-0.447213595499958 abc/def**2")

    def test_copy(self):
        """Test if experiment ufloat can be copied."""
        val = ExperimentVariable(0.1, 0.2, tag="f", unit="Hz")
        val_copied = copy.copy(val)

        # Copied value is not equivalent in sense of correlation
        # This is behavior defined in the original uncertainties package
        self.assertNotEqual(val, val_copied)

        self.assertEqual(val_copied.n, val.n)
        self.assertEqual(val_copied.s, val.s)
        self.assertEqual(val_copied.tag, val.tag)
        self.assertEqual(val_copied.unit, val.unit)

    @data(
        [0.1, 0.2, None, "Hz"],
        [-0.4, 0.8, "abc", None],
        [1.234, 5.678, "abc", "abcd/ef"],
    )
    @unpack
    def test_value_json_serializable(self, nom, std, tag, unit):
        """Test if experiment ufloat is serializable."""
        val = ExperimentVariable(nom, std, tag=tag, unit=unit)
        self.assertRoundTripSerializable(val)

    @data(
        [0.1234, 0.2345, 0.7654, 0.4567],
        [-5.432, 0.02, -5.432, 0.02],
        [-1.234, 0.90, 3.421, 0.32],
    )
    @unpack
    def test_loaded_value_keep_correlation(self, v1n, v1s, v2n, v2s):
        """Test if round trip value can keep parameter correlation."""
        v1 = ExperimentVariable(v1n, v1s)
        v2 = ExperimentVariable(v2n, v2s)

        v3 = v1 * v2

        encoded_v3 = json.dumps(v3, cls=ExperimentEncoder)
        loaded_v3 = json.loads(encoded_v3, cls=ExperimentDecoder)

        res = v1 + v3
        res_with_loaded = v1 + loaded_v3

        self.assertEqual(res_with_loaded.n, res.n)
        self.assertEqual(res_with_loaded.s, res.s)
