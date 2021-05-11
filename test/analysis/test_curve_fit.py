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

"""Test curve fitting base class."""

from typing import List, Callable

import numpy as np
from qiskit.test import QiskitTestCase

from qiskit_experiments import ExperimentData
from qiskit_experiments.analysis import CurveAnalysis, SeriesDef
from qiskit_experiments.base_experiment import BaseExperiment


class FakeExperiment(BaseExperiment):
    """A fake experiment class."""

    def __init__(self):
        super().__init__(qubits=(0,), experiment_type="fake_experiment")

    def circuits(self, backend=None, **circuit_options):
        return []


def simulate_output_data(func, xvals, *params, **metadata):
    """Generate arbitrary fit data."""
    __shots = 1024

    expected_probs = func(xvals, *params)
    counts = np.asarray(expected_probs * __shots, dtype=int)

    data = [
        {
            "counts": {"0": __shots - count, "1": count},
            "metadata": dict(xval=xi, qubits=(0,), experiment_type="fake_experiment", **metadata),
        }
        for xi, count in zip(xvals, counts)
    ]

    expdata = ExperimentData(experiment=FakeExperiment())
    for datum in data:
        expdata.add_data(datum)

    return expdata


def create_new_analysis(
    x_key: str = "xval",
    series: List[SeriesDef] = None,
    fit_funcs: List[Callable] = None,
    param_names: List[str] = None,
) -> CurveAnalysis:
    """A helper function to create a mock analysis class instance."""

    class TestAnalysis(CurveAnalysis):
        """A mock analysis class to test."""

        __x_key__ = x_key
        __series__ = series
        __fit_funcs__ = fit_funcs
        __param_names__ = param_names

    return TestAnalysis()


class TestCurveAnalysis(QiskitTestCase):
    """Unittest for curve fit analysis. Assuming several fitting situations."""

    def setUp(self):
        super().setUp()
        self.xvalues = np.linspace(0.1, 1, 30)

        # fit functions
        self.exp_func = lambda x, p0, p1, p2: p0 * np.exp(p1 * x) + p2
        self.cos_func = lambda x, p0, p1, p2, p3: p0 * np.cos(2 * np.pi * p1 * x + p2) + p3
        self.sin_func = lambda x, p0, p1, p2, p3: p0 * np.sin(2 * np.pi * p1 * x + p2) + p3

    def test_run_single_curve_analysis(self):
        """Test analysis for single curve."""
        analysis = create_new_analysis(fit_funcs=[self.exp_func], param_names=["p0", "p1", "p2"])
        ref_p0 = 0.9
        ref_p1 = -2.5
        ref_p2 = 0.1

        test_data = simulate_output_data(self.exp_func, self.xvalues, ref_p0, ref_p1, ref_p2)
        results, _ = analysis._run_analysis(test_data, p0=[ref_p0, ref_p1, ref_p2])

        ref_popt = np.asarray([ref_p0, ref_p1, ref_p2])

        # check result data
        np.testing.assert_array_almost_equal(results["popt"], ref_popt, decimal=1)
        self.assertEqual(results["dof"], 27)
        self.assertListEqual(results["xrange"], [0.1, 1.0])
        self.assertListEqual(results["popt_keys"], ["p0", "p1", "p2"])
        self.assertTrue(results["success"])

    def test_run_single_curve_fail(self):
        """Test analysis returns status when it fails."""
        analysis = create_new_analysis(fit_funcs=[self.exp_func], param_names=["p0", "p1", "p2"])
        ref_p0 = 0.9
        ref_p1 = -2.5
        ref_p2 = 0.1

        test_data = simulate_output_data(self.exp_func, self.xvalues, ref_p0, ref_p1, ref_p2)

        # Try to fit with infeasible parameter boundary. This should fail.
        results, _ = analysis._run_analysis(
            test_data, p0=[ref_p0, ref_p1, ref_p2], bounds=([-10, -10, -10], [0, 0, 0])
        )

        self.assertFalse(results["success"])

        ref_result_keys = ["raw_data", "error_message", "success"]
        self.assertListEqual(list(results.keys()), ref_result_keys)

    def test_run_two_curves_with_same_fitfunc(self):
        """Test analysis for two curves. Curves shares fit model."""
        analysis = create_new_analysis(
            series=[
                SeriesDef(
                    name="curve1",
                    param_names=["p0", "p1", "p3"],
                    fit_func_index=0,
                    filter_kwargs={"exp": 0},
                ),
                SeriesDef(
                    name="curve2",
                    param_names=["p0", "p2", "p3"],
                    fit_func_index=0,
                    filter_kwargs={"exp": 1},
                ),
            ],
            fit_funcs=[self.exp_func],
            param_names=["p0", "p1", "p2", "p3"],
        )
        ref_p0 = 0.9
        ref_p1 = -7.0
        ref_p2 = -5.0
        ref_p3 = 0.1

        test_data0 = simulate_output_data(
            self.exp_func, self.xvalues, ref_p0, ref_p1, ref_p3, exp=0
        )
        test_data1 = simulate_output_data(
            self.exp_func, self.xvalues, ref_p0, ref_p2, ref_p3, exp=1
        )

        # merge two experiment data
        for datum in test_data1.data:
            test_data0.add_data(datum)

        results, _ = analysis._run_analysis(test_data0, p0=[ref_p0, ref_p1, ref_p2, ref_p3])

        ref_popt = np.asarray([ref_p0, ref_p1, ref_p2, ref_p3])

        # check result data
        np.testing.assert_array_almost_equal(results["popt"], ref_popt, decimal=1)
        self.assertTrue(results["success"])

    def test_run_two_curves_with_two_fitfuncs(self):
        """Test analysis for two curves. Curves shares fit parameters."""
        analysis = create_new_analysis(
            series=[
                SeriesDef(
                    name="curve1",
                    param_names=["p0", "p1", "p2", "p3"],
                    fit_func_index=0,
                    filter_kwargs={"exp": 0},
                ),
                SeriesDef(
                    name="curve2",
                    param_names=["p0", "p1", "p2", "p3"],
                    fit_func_index=1,
                    filter_kwargs={"exp": 1},
                ),
            ],
            fit_funcs=[self.cos_func, self.sin_func],
            param_names=["p0", "p1", "p2", "p3"],
        )
        ref_p0 = 0.1
        ref_p1 = 2
        ref_p2 = -0.3
        ref_p3 = 0.5

        test_data0 = simulate_output_data(
            self.cos_func, self.xvalues, ref_p0, ref_p1, ref_p2, ref_p3, exp=0
        )
        test_data1 = simulate_output_data(
            self.sin_func, self.xvalues, ref_p0, ref_p1, ref_p2, ref_p3, exp=1
        )

        # merge two experiment data
        for datum in test_data1.data:
            test_data0.add_data(datum)

        results, _ = analysis._run_analysis(test_data0, p0=[ref_p0, ref_p1, ref_p2, ref_p3])

        ref_popt = np.asarray([ref_p0, ref_p1, ref_p2, ref_p3])

        # check result data
        np.testing.assert_array_almost_equal(results["popt"], ref_popt, decimal=1)
        self.assertTrue(results["success"])
