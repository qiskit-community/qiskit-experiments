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
# pylint: disable=invalid-name

from typing import List

import numpy as np
from qiskit.test import QiskitTestCase

from qiskit_experiments import ExperimentData
from qiskit_experiments.analysis import CurveAnalysis, SeriesDef, fit_functions, FitOptions
from qiskit_experiments.base_experiment import BaseExperiment


class FakeExperiment(BaseExperiment):
    """A fake experiment class."""

    def __init__(self):
        super().__init__(qubits=(0,), experiment_type="fake_experiment")

    def circuits(self, backend=None, **circuit_options):
        return []


def simulate_output_data(func, xvals, param_dict, **metadata):
    """Generate arbitrary fit data."""
    __shots = 100000

    expected_probs = func(xvals, **param_dict)
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
) -> CurveAnalysis:
    """A helper function to create a mock analysis class instance."""

    class TestAnalysis(CurveAnalysis):
        """A mock analysis class to test."""

        __x_key__ = x_key
        __series__ = series
        __processing_options__ = ["outcome"]

    return TestAnalysis()


class TestCurveAnalysisUnit(QiskitTestCase):
    """Unittest for curve fit analysis."""

    def setUp(self):
        super().setUp()
        self.xvalues = np.linspace(1.0, 5.0, 10)

        # Description of test setting
        #
        # - This model contains three curves, namely, curve1, curve2, curve3
        # - Each curve can be represented by the same function
        # - Parameter amp and baseline are shared among all curves
        # - Each curve has unique lamb
        # - In total 5 parameters in the fit, namely, p0, p1, p2, p3
        #
        self.analysis = create_new_analysis(
            series=[
                SeriesDef(
                    name="curve1",
                    fit_func=lambda x, p0, p1, p2, p3, p4: fit_functions.exponential_decay(
                        x, amp=p0, lamb=p1, baseline=p4
                    ),
                    filter_kwargs={"type": 1, "valid": True},
                ),
                SeriesDef(
                    name="curve2",
                    fit_func=lambda x, p0, p1, p2, p3, p4: fit_functions.exponential_decay(
                        x, amp=p0, lamb=p2, baseline=p4
                    ),
                    filter_kwargs={"type": 2, "valid": True},
                ),
                SeriesDef(
                    name="curve3",
                    fit_func=lambda x, p0, p1, p2, p3, p4: fit_functions.exponential_decay(
                        x, amp=p0, lamb=p3, baseline=p4
                    ),
                    filter_kwargs={"type": 3, "valid": True},
                ),
            ],
        )
        self.err_decimal = 3

    @staticmethod
    def data_processor(data, outcome):
        """A helper method to format input data."""
        counts = data["counts"]
        outcome = outcome or "1" * len(list(counts.keys())[0])

        shots = sum(counts.values())
        p_mean = counts.get(outcome, 0.0) / shots
        p_var = p_mean * (1 - p_mean) / shots

        return p_mean, p_var

    def test_data_extraction(self):
        """Test data extraction method."""
        # data to analyze
        test_data0 = simulate_output_data(
            func=fit_functions.exponential_decay,
            xvals=self.xvalues,
            param_dict={"amp": 1.0},
            type=1,
            valid=True,
            outcome="1",
        )

        # fake data
        test_data1 = simulate_output_data(
            func=fit_functions.exponential_decay,
            xvals=self.xvalues,
            param_dict={"amp": 0.5},
            type=2,
            valid=False,
            outcome="1",
        )
        # merge two experiment data
        for datum in test_data1.data():
            test_data0.add_data(datum)

        xdata, ydata, sigma, series = self.analysis._extract_curves(test_data0, self.data_processor)

        # check if the module filter off data: valid=False
        self.assertEqual(len(xdata), 20)

        # check x values
        ref_x = np.concatenate((self.xvalues, self.xvalues))
        np.testing.assert_array_almost_equal(xdata, ref_x)

        # check y values
        ref_y = np.concatenate(
            (
                fit_functions.exponential_decay(self.xvalues, amp=1.0),
                fit_functions.exponential_decay(self.xvalues, amp=0.5),
            )
        )
        np.testing.assert_array_almost_equal(ydata, ref_y, decimal=self.err_decimal)

        # check series
        ref_series = np.concatenate((np.zeros(10, dtype=int), -1 * np.ones(10, dtype=int)))
        self.assertListEqual(list(series), list(ref_series))

        # check y errors
        ref_yerr = ref_y * (1 - ref_y) / 100000
        np.testing.assert_array_almost_equal(sigma, ref_yerr, decimal=self.err_decimal)

    def test_get_subset(self):
        """Test that get subset data from full data array."""

        xdata = np.asarray([1, 2, 3, 4, 5, 6], dtype=float)
        ydata = np.asarray([1, 2, 3, 4, 5, 6], dtype=float)
        sigma = np.asarray([1, 2, 3, 4, 5, 6], dtype=float)
        series = np.asarray([0, 1, 0, 2, 2, -1], dtype=int)

        subx, suby, subs = self.analysis._subset_data("curve1", xdata, ydata, sigma, series)
        np.testing.assert_array_almost_equal(subx, np.asarray([1, 3], dtype=float))
        np.testing.assert_array_almost_equal(suby, np.asarray([1, 3], dtype=float))
        np.testing.assert_array_almost_equal(subs, np.asarray([1, 3], dtype=float))

        subx, suby, subs = self.analysis._subset_data("curve2", xdata, ydata, sigma, series)
        np.testing.assert_array_almost_equal(subx, np.asarray([2], dtype=float))
        np.testing.assert_array_almost_equal(suby, np.asarray([2], dtype=float))
        np.testing.assert_array_almost_equal(subs, np.asarray([2], dtype=float))

        subx, suby, subs = self.analysis._subset_data("curve3", xdata, ydata, sigma, series)
        np.testing.assert_array_almost_equal(subx, np.asarray([4, 5], dtype=float))
        np.testing.assert_array_almost_equal(suby, np.asarray([4, 5], dtype=float))
        np.testing.assert_array_almost_equal(subs, np.asarray([4, 5], dtype=float))

    def test_formatting_options(self):
        """Test option formatter."""
        test_options = FitOptions(
            p0=[0, 1, 2, 3, 4], bounds=[(-1, 1), (-2, 2), (-3, 3), (-4, 4), (-5, 5)]
        )
        formatted_options = self.analysis._format_fit_options(test_options)

        ref_p0 = {"p0": 0, "p1": 1, "p2": 2, "p3": 3, "p4": 4}
        self.assertDictEqual(formatted_options["p0"], ref_p0)

        ref_bounds = {"p0": (-1, 1), "p1": (-2, 2), "p2": (-3, 3), "p3": (-4, 4), "p4": (-5, 5)}
        self.assertDictEqual(formatted_options["bounds"], ref_bounds)


class TestCurveAnalysisIntegration(QiskitTestCase):
    """Integration test for curve fit analysis through entire analysis.run function."""

    def setUp(self):
        super().setUp()
        self.xvalues = np.linspace(0.1, 1, 50)
        self.err_decimal = 2

    def test_run_single_curve_analysis(self):
        """Test analysis for single curve."""
        analysis = create_new_analysis(
            series=[
                SeriesDef(
                    name="curve1",
                    fit_func=lambda x, p0, p1, p2, p3: fit_functions.exponential_decay(
                        x, amp=p0, lamb=p1, x0=p2, baseline=p3
                    ),
                )
            ],
        )
        ref_p0 = 0.9
        ref_p1 = 2.5
        ref_p2 = 0.0
        ref_p3 = 0.1

        test_data = simulate_output_data(
            func=fit_functions.exponential_decay,
            xvals=self.xvalues,
            param_dict={"amp": ref_p0, "lamb": ref_p1, "x0": ref_p2, "baseline": ref_p3},
            outcome="1",
        )
        results, _ = analysis._run_analysis(test_data, p0=[ref_p0, ref_p1, ref_p2, ref_p3])
        result = results[0]

        ref_popt = np.asarray([ref_p0, ref_p1, ref_p2, ref_p3])

        # check result data
        self.assertTrue(result["success"])

        np.testing.assert_array_almost_equal(result["popt"], ref_popt, decimal=self.err_decimal)
        self.assertEqual(result["dof"], 46)
        self.assertListEqual(result["xrange"], [0.1, 1.0])
        self.assertListEqual(result["popt_keys"], ["p0", "p1", "p2", "p3"])

    def test_run_single_curve_fail(self):
        """Test analysis returns status when it fails."""
        analysis = create_new_analysis(
            series=[
                SeriesDef(
                    name="curve1",
                    fit_func=lambda x, p0, p1, p2, p3: fit_functions.exponential_decay(
                        x, amp=p0, lamb=p1, x0=p2, baseline=p3
                    ),
                )
            ],
        )
        ref_p0 = 0.9
        ref_p1 = 2.5
        ref_p2 = 0.0
        ref_p3 = 0.1

        test_data = simulate_output_data(
            func=fit_functions.exponential_decay,
            xvals=self.xvalues,
            param_dict={"amp": ref_p0, "lamb": ref_p1, "x0": ref_p2, "baseline": ref_p3},
            outcome="1",
        )

        # Try to fit with infeasible parameter boundary. This should fail.
        results, _ = analysis._run_analysis(
            test_data,
            p0=[ref_p0, ref_p1, ref_p2, ref_p3],
            bounds=([-10, -10, -10, -10], [0, 0, 0, 0]),
        )
        result = results[0]

        self.assertFalse(result["success"])

        ref_result_keys = ["raw_data", "error_message", "success"]
        self.assertSetEqual(set(result.keys()), set(ref_result_keys))

    def test_run_two_curves_with_same_fitfunc(self):
        """Test analysis for two curves. Curves shares fit model."""
        analysis = create_new_analysis(
            series=[
                SeriesDef(
                    name="curve1",
                    fit_func=lambda x, p0, p1, p2, p3, p4: fit_functions.exponential_decay(
                        x, amp=p0, lamb=p1, x0=p3, baseline=p4
                    ),
                    filter_kwargs={"exp": 0},
                ),
                SeriesDef(
                    name="curve1",
                    fit_func=lambda x, p0, p1, p2, p3, p4: fit_functions.exponential_decay(
                        x, amp=p0, lamb=p2, x0=p3, baseline=p4
                    ),
                    filter_kwargs={"exp": 1},
                ),
            ],
        )
        ref_p0 = 0.9
        ref_p1 = 7.0
        ref_p2 = 5.0
        ref_p3 = 0.0
        ref_p4 = 0.1

        test_data0 = simulate_output_data(
            func=fit_functions.exponential_decay,
            xvals=self.xvalues,
            param_dict={"amp": ref_p0, "lamb": ref_p1, "x0": ref_p3, "baseline": ref_p4},
            exp=0,
            outcome="1",
        )

        test_data1 = simulate_output_data(
            func=fit_functions.exponential_decay,
            xvals=self.xvalues,
            param_dict={"amp": ref_p0, "lamb": ref_p2, "x0": ref_p3, "baseline": ref_p4},
            exp=1,
            outcome="1",
        )

        # merge two experiment data
        for datum in test_data1.data():
            test_data0.add_data(datum)

        results, _ = analysis._run_analysis(test_data0, p0=[ref_p0, ref_p1, ref_p2, ref_p3, ref_p4])
        result = results[0]

        ref_popt = np.asarray([ref_p0, ref_p1, ref_p2, ref_p3, ref_p4])

        # check result data
        self.assertTrue(result["success"])
        np.testing.assert_array_almost_equal(result["popt"], ref_popt, decimal=self.err_decimal)

    def test_run_two_curves_with_two_fitfuncs(self):
        """Test analysis for two curves. Curves shares fit parameters."""
        analysis = create_new_analysis(
            series=[
                SeriesDef(
                    name="curve1",
                    fit_func=lambda x, p0, p1, p2, p3: fit_functions.cos(
                        x, amp=p0, freq=p1, phase=p2, baseline=p3
                    ),
                    filter_kwargs={"exp": 0},
                ),
                SeriesDef(
                    name="curve2",
                    fit_func=lambda x, p0, p1, p2, p3: fit_functions.sin(
                        x, amp=p0, freq=p1, phase=p2, baseline=p3
                    ),
                    filter_kwargs={"exp": 1},
                ),
            ],
        )
        ref_p0 = 0.1
        ref_p1 = 2
        ref_p2 = -0.3
        ref_p3 = 0.5

        test_data0 = simulate_output_data(
            func=fit_functions.cos,
            xvals=self.xvalues,
            param_dict={"amp": ref_p0, "freq": ref_p1, "phase": ref_p2, "baseline": ref_p3},
            exp=0,
            outcome="1",
        )

        test_data1 = simulate_output_data(
            func=fit_functions.sin,
            xvals=self.xvalues,
            param_dict={"amp": ref_p0, "freq": ref_p1, "phase": ref_p2, "baseline": ref_p3},
            exp=1,
            outcome="1",
        )

        # merge two experiment data
        for datum in test_data1.data():
            test_data0.add_data(datum)

        results, _ = analysis._run_analysis(test_data0, p0=[ref_p0, ref_p1, ref_p2, ref_p3])
        result = results[0]

        ref_popt = np.asarray([ref_p0, ref_p1, ref_p2, ref_p3])

        # check result data
        self.assertTrue(result["success"])
        np.testing.assert_array_almost_equal(result["popt"], ref_popt, decimal=self.err_decimal)
