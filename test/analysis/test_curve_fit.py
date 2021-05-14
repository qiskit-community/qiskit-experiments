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

from typing import List, Callable

import numpy as np
from qiskit.test import QiskitTestCase

from qiskit_experiments import ExperimentData
from qiskit_experiments.analysis import CurveAnalysis, SeriesDef, fit_functions
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
        # - In total 4 parameters in the fit, namely, p0, p1, p2, p3
        #
        self.analysis = create_new_analysis(
            series=[
                SeriesDef(
                    name="curve1",
                    p0_signature={"amp": "p0", "lamb": "p1", "baseline": "p4"},
                    fit_func_index=0,
                    filter_kwargs={"type": 1, "valid": True},
                    data_option_keys=["outcome"],
                ),
                SeriesDef(
                    name="curve2",
                    p0_signature={"amp": "p0", "lamb": "p2", "baseline": "p4"},
                    fit_func_index=0,
                    filter_kwargs={"type": 2, "valid": True},
                    data_option_keys=["outcome"],
                ),
                SeriesDef(
                    name="curve3",
                    p0_signature={"amp": "p0", "lamb": "p3", "baseline": "p4"},
                    fit_func_index=0,
                    filter_kwargs={"type": 3, "valid": True},
                    data_option_keys=["outcome"],
                ),
            ],
            fit_funcs=[fit_functions.exponential_decay],
            param_names=["p0", "p1", "p2", "p3", "p4"],
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
            dummy_val="test_val1",
            outcome="1",
        )

        # fake data
        test_data1 = simulate_output_data(
            func=fit_functions.exponential_decay,
            xvals=self.xvalues,
            param_dict={"amp": 1.0},
            type=2,
            valid=False,
            dummy_val="test_val2",
            outcome="1",
        )
        # merge two experiment data
        for datum in test_data1.data():
            test_data0.add_data(datum)

        curve_entries = self.analysis._extract_curves(test_data0, self.data_processor)

        # check if the module filter off data: valid=False
        self.assertEqual(len(curve_entries), 3)

        # check name is passed
        self.assertEqual(curve_entries[0].curve_name, "curve1")

        # check x values
        np.testing.assert_array_almost_equal(curve_entries[0].x_values, self.xvalues)

        # check y values
        ref_y = fit_functions.exponential_decay(self.xvalues, amp=1.0)
        np.testing.assert_array_almost_equal(
            curve_entries[0].y_values, ref_y, decimal=self.err_decimal
        )

        # check y errors
        ref_yerr = ref_y * (1 - ref_y) / 1024
        np.testing.assert_array_almost_equal(
            curve_entries[0].y_sigmas, ref_yerr, decimal=self.err_decimal
        )

        # check metadata
        ref_meta = {
            "experiment_type": {"fake_experiment"},
            "qubits": {(0,)},
            "dummy_val": {"test_val1"},
        }
        self.assertDictEqual(curve_entries[0].metadata, ref_meta)

    def test_curve_calculation(self):
        """Test series curve calculation."""
        params = [1.0, 0.7, 0.9, 1.1, 0.1]

        y1 = self.analysis._calculate_curve("curve1", self.xvalues, *params)
        ref_y1 = fit_functions.exponential_decay(self.xvalues, amp=1.0, lamb=0.7, baseline=0.1)
        np.testing.assert_array_almost_equal(y1, ref_y1, decimal=self.err_decimal)

        y2 = self.analysis._calculate_curve("curve2", self.xvalues, *params)
        ref_y2 = fit_functions.exponential_decay(self.xvalues, amp=1.0, lamb=0.9, baseline=0.1)
        np.testing.assert_array_almost_equal(y2, ref_y2, decimal=self.err_decimal)

        y3 = self.analysis._calculate_curve("curve3", self.xvalues, *params)
        ref_y3 = fit_functions.exponential_decay(self.xvalues, amp=1.0, lamb=1.1, baseline=0.1)
        np.testing.assert_array_almost_equal(y3, ref_y3, decimal=self.err_decimal)

    def test_default_setup_fitting(self):
        """Test default behavior of fitter setup."""
        curve_data = []

        options = self.analysis._setup_fitting(curve_data)

        ref_p0 = {"p0": 0.0, "p1": 0.0, "p2": 0.0, "p3": 0.0, "p4": 0.0}
        self.assertDictEqual(options[0]["p0"], ref_p0)

        ref_lb = {"p0": -np.inf, "p1": -np.inf, "p2": -np.inf, "p3": -np.inf, "p4": -np.inf}
        ref_ub = {"p0": np.inf, "p1": np.inf, "p2": np.inf, "p3": np.inf, "p4": np.inf}

        lb, ub = options[0]["bounds"]
        self.assertDictEqual(lb, ref_lb)
        self.assertDictEqual(ub, ref_ub)

    def test_default_setup_fitting_with_parameter(self):
        """Test default behavior of fitter setup when user parameter is provided."""
        curve_data = []

        options = self.analysis._setup_fitting(
            curve_data,
            p0=[1.0, 2.0, 3.0, 4.0, 5.0],
            bounds=([-1.0, -2.0, -3.0, -4.0, -5.0], [1.0, 2.0, 3.0, 4.0, 5.0]),
        )

        ref_p0 = {"p0": 1.0, "p1": 2.0, "p2": 3.0, "p3": 4.0, "p4": 5.0}
        self.assertDictEqual(options[0]["p0"], ref_p0)

        ref_lb = {"p0": -1.0, "p1": -2.0, "p2": -3.0, "p3": -4.0, "p4": -5.0}
        ref_ub = {"p0": 1.0, "p1": 2.0, "p2": 3.0, "p3": 4.0, "p4": 5.0}

        lb, ub = options[0]["bounds"]
        self.assertDictEqual(lb, ref_lb)
        self.assertDictEqual(ub, ref_ub)


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
                    p0_signature={"amp": "p0", "lamb": "p1", "x0": "p2", "baseline": "p3"},
                    fit_func_index=0,
                    filter_kwargs=None,
                    data_option_keys=None,
                )
            ],
            fit_funcs=[fit_functions.exponential_decay],
            param_names=["p0", "p1", "p2", "p3"],
        )
        ref_p0 = 0.9
        ref_p1 = 2.5
        ref_p2 = 0.0
        ref_p3 = 0.1

        test_data = simulate_output_data(
            func=fit_functions.exponential_decay,
            xvals=self.xvalues,
            param_dict={"amp": ref_p0, "lamb": ref_p1, "x0": ref_p2, "baseline": ref_p3},
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
                    p0_signature={"amp": "p0", "lamb": "p1", "x0": "p2", "baseline": "p3"},
                    fit_func_index=0,
                    filter_kwargs=None,
                    data_option_keys=None,
                )
            ],
            fit_funcs=[fit_functions.exponential_decay],
            param_names=["p0", "p1", "p2", "p3"],
        )
        ref_p0 = 0.9
        ref_p1 = 2.5
        ref_p2 = 0.0
        ref_p3 = 0.1

        test_data = simulate_output_data(
            func=fit_functions.exponential_decay,
            xvals=self.xvalues,
            param_dict={"amp": ref_p0, "lamb": ref_p1, "x0": ref_p2, "baseline": ref_p3},
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
                    p0_signature={"amp": "p0", "lamb": "p1", "x0": "p3", "baseline": "p4"},
                    fit_func_index=0,
                    filter_kwargs={"exp": 0},
                    data_option_keys=None,
                ),
                SeriesDef(
                    name="curve2",
                    p0_signature={"amp": "p0", "lamb": "p2", "x0": "p3", "baseline": "p4"},
                    fit_func_index=0,
                    filter_kwargs={"exp": 1},
                    data_option_keys=None,
                ),
            ],
            fit_funcs=[fit_functions.exponential_decay],
            param_names=["p0", "p1", "p2", "p3", "p4"],
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
        )

        test_data1 = simulate_output_data(
            func=fit_functions.exponential_decay,
            xvals=self.xvalues,
            param_dict={"amp": ref_p0, "lamb": ref_p2, "x0": ref_p3, "baseline": ref_p4},
            exp=1,
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
                    p0_signature={"amp": "p0", "freq": "p1", "phase": "p2", "baseline": "p3"},
                    fit_func_index=0,
                    filter_kwargs={"exp": 0},
                    data_option_keys=None,
                ),
                SeriesDef(
                    name="curve2",
                    p0_signature={"amp": "p0", "freq": "p1", "phase": "p2", "baseline": "p3"},
                    fit_func_index=1,
                    filter_kwargs={"exp": 1},
                    data_option_keys=None,
                ),
            ],
            fit_funcs=[fit_functions.cos, fit_functions.sin],
            param_names=["p0", "p1", "p2", "p3"],
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
        )

        test_data1 = simulate_output_data(
            func=fit_functions.sin,
            xvals=self.xvalues,
            param_dict={"amp": ref_p0, "freq": ref_p1, "phase": ref_p2, "baseline": ref_p3},
            exp=1,
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

    def test_fit_with_data_option(self):
        """Test analysis by passing data processing option to the data processor."""

        def inverted_decay(x, amp, lamb, x0, baseline):
            # measure inverse of population
            return 1 - fit_functions.exponential_decay(
                x, amp=amp, lamb=lamb, x0=x0, baseline=baseline
            )

        analysis = create_new_analysis(
            series=[
                SeriesDef(
                    name="curve1",
                    p0_signature={"amp": "p0", "lamb": "p1", "x0": "p2", "baseline": "p3"},
                    fit_func_index=0,
                    filter_kwargs=None,
                    data_option_keys=["outcome"],
                )
            ],
            fit_funcs=[inverted_decay],
            param_names=["p0", "p1", "p2", "p3"],
        )
        ref_p0 = 0.9
        ref_p1 = 2.5
        ref_p2 = 0.0
        ref_p3 = 0.1

        # tell metadata to count zero
        test_data = simulate_output_data(
            func=fit_functions.exponential_decay,
            xvals=self.xvalues,
            param_dict={"amp": ref_p0, "lamb": ref_p1, "x0": ref_p2, "baseline": ref_p3},
            outcome="0",  # metadata, label to count
        )

        results, _ = analysis._run_analysis(test_data, p0=[ref_p0, ref_p1, ref_p2, ref_p3])
        result = results[0]

        ref_popt = np.asarray([ref_p0, ref_p1, ref_p2, ref_p3])

        # check result data
        self.assertTrue(result["success"])

        np.testing.assert_array_almost_equal(result["popt"], ref_popt, decimal=self.err_decimal)

    def test_fit_failure_with_wrong_signature(self):
        """Test if fitting fails when wrong signature is defined."""
        analysis = create_new_analysis(
            series=[
                SeriesDef(
                    name="curve1",
                    p0_signature={"not_defined_parameter": "p0"},  # invalid mapping
                    fit_func_index=0,
                    filter_kwargs=None,
                    data_option_keys=None,
                )
            ],
            fit_funcs=[fit_functions.exponential_decay],
            param_names=["p0"],
        )
        ref_p0 = 0.9

        test_data = simulate_output_data(
            func=fit_functions.cos,
            xvals=self.xvalues,
            param_dict={"amp": ref_p0},
        )

        results, _ = analysis._run_analysis(test_data, p0=[ref_p0])
        result = results[0]

        self.assertFalse(result["success"])

        ref_result_keys = ["raw_data", "error_message", "success"]
        self.assertSetEqual(set(result.keys()), set(ref_result_keys))

    def test_fit_failure_with_unclear_parameter(self):
        """Test if fitting fails when parameter not defined in fit is used.."""
        analysis = create_new_analysis(
            series=[
                SeriesDef(
                    name="curve1",
                    p0_signature={"amp": "not_defined_parameter"},  # this parameter is not defined
                    fit_func_index=0,
                    filter_kwargs=None,
                    data_option_keys=None,
                )
            ],
            fit_funcs=[fit_functions.exponential_decay],
            param_names=["p0"],
        )
        ref_p0 = 0.9

        test_data = simulate_output_data(
            func=fit_functions.cos,
            xvals=self.xvalues,
            param_dict={"amp": ref_p0},
        )

        results, _ = analysis._run_analysis(test_data, p0=[ref_p0])
        result = results[0]

        self.assertFalse(result["success"])

        ref_result_keys = ["raw_data", "error_message", "success"]
        self.assertSetEqual(set(result.keys()), set(ref_result_keys))
