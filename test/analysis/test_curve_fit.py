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
from qiskit_experiments.analysis import CurveAnalysis, SeriesDef, fit_function
from qiskit_experiments.analysis.curve_fitting import multi_curve_fit
from qiskit_experiments.analysis.data_processing import probability
from qiskit_experiments.base_experiment import BaseExperiment
from qiskit_experiments.exceptions import AnalysisError


class FakeExperiment(BaseExperiment):
    """A fake experiment class."""

    def __init__(self):
        super().__init__(qubits=(0,), experiment_type="fake_experiment")

    def circuits(self, backend=None):
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


def create_new_analysis(series: List[SeriesDef]) -> CurveAnalysis:
    """A helper function to create a mock analysis class instance."""

    class TestAnalysis(CurveAnalysis):
        """A mock analysis class to test."""

        __series__ = series

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
                    fit_func=lambda x, p0, p1, p2, p3, p4: fit_function.exponential_decay(
                        x, amp=p0, lamb=p1, baseline=p4
                    ),
                    filter_kwargs={"type": 1, "valid": True},
                ),
                SeriesDef(
                    name="curve2",
                    fit_func=lambda x, p0, p1, p2, p3, p4: fit_function.exponential_decay(
                        x, amp=p0, lamb=p2, baseline=p4
                    ),
                    filter_kwargs={"type": 2, "valid": True},
                ),
                SeriesDef(
                    name="curve3",
                    fit_func=lambda x, p0, p1, p2, p3, p4: fit_function.exponential_decay(
                        x, amp=p0, lamb=p3, baseline=p4
                    ),
                    filter_kwargs={"type": 3, "valid": True},
                ),
            ],
        )
        self.err_decimal = 3

    def test_cannot_create_invalid_series_fit(self):
        """Test we cannot create invalid analysis instance."""
        invalid_series = [
            SeriesDef(
                name="fit1",
                fit_func=lambda x, p0: fit_function.exponential_decay(x, amp=p0),
            ),
            SeriesDef(
                name="fit2",
                fit_func=lambda x, p1: fit_function.exponential_decay(x, amp=p1),
            ),
        ]
        with self.assertRaises(AnalysisError):
            create_new_analysis(series=invalid_series)  # fit1 has param p0 while fit2 has p1

    def test_arg_parse_and_get_option(self):
        """Test if option parsing works correctly."""
        user_option = {"x_key": "test_value", "test_key1": "value1", "test_key2": "value2"}

        # argument not defined in default option should be returned as extra option
        extra_option = self.analysis._arg_parse(**user_option)
        ref_option = {"test_key1": "value1", "test_key2": "value2"}
        self.assertDictEqual(extra_option, ref_option)

        # default option value is stored as class variable
        self.assertEqual(self.analysis._get_option("x_key"), "test_value")

    def test_data_extraction(self):
        """Test data extraction method."""
        self.analysis._arg_parse(x_key="xval")

        # data to analyze
        test_data0 = simulate_output_data(
            func=fit_function.exponential_decay,
            xvals=self.xvalues,
            param_dict={"amp": 1.0},
            type=1,
            valid=True,
        )

        # fake data
        test_data1 = simulate_output_data(
            func=fit_function.exponential_decay,
            xvals=self.xvalues,
            param_dict={"amp": 0.5},
            type=2,
            valid=False,
        )

        # merge two experiment data
        for datum in test_data1.data():
            test_data0.add_data(datum)

        self.analysis._extract_curves(
            experiment_data=test_data0, data_processor=probability(outcome="1")
        )

        raw_data = self.analysis._data(label="raw_data")

        xdata = raw_data.x
        ydata = raw_data.y
        sigma = raw_data.y_err
        d_index = raw_data.data_index

        # check if the module filter off data: valid=False
        self.assertEqual(len(xdata), 20)

        # check x values
        ref_x = np.concatenate((self.xvalues, self.xvalues))
        np.testing.assert_array_almost_equal(xdata, ref_x)

        # check y values
        ref_y = np.concatenate(
            (
                fit_function.exponential_decay(self.xvalues, amp=1.0),
                fit_function.exponential_decay(self.xvalues, amp=0.5),
            )
        )
        np.testing.assert_array_almost_equal(ydata, ref_y, decimal=self.err_decimal)

        # check series
        ref_series = np.concatenate((np.zeros(10, dtype=int), -1 * np.ones(10, dtype=int)))
        self.assertListEqual(list(d_index), list(ref_series))

        # check y errors
        ref_yerr = ref_y * (1 - ref_y) / 100000
        np.testing.assert_array_almost_equal(sigma, ref_yerr, decimal=self.err_decimal)

    def test_get_subset(self):
        """Test that get subset data from full data array."""
        # data to analyze
        fake_data = [
            {"data": 1, "metadata": {"xval": 1, "type": 1, "valid": True}},
            {"data": 2, "metadata": {"xval": 2, "type": 2, "valid": True}},
            {"data": 3, "metadata": {"xval": 3, "type": 1, "valid": True}},
            {"data": 4, "metadata": {"xval": 4, "type": 3, "valid": True}},
            {"data": 5, "metadata": {"xval": 5, "type": 3, "valid": True}},
            {"data": 6, "metadata": {"xval": 6, "type": 4, "valid": True}},  # this if fake
        ]
        expdata = ExperimentData(experiment=FakeExperiment())
        for datum in fake_data:
            expdata.add_data(datum)

        def _processor(datum):
            return datum["data"], datum["data"] * 2

        self.analysis._arg_parse(x_key="xval")
        self.analysis._extract_curves(expdata, data_processor=_processor)

        filt_data = self.analysis._data(series_name="curve1")
        np.testing.assert_array_equal(filt_data.x, np.asarray([1, 3], dtype=float))
        np.testing.assert_array_equal(filt_data.y, np.asarray([1, 3], dtype=float))
        np.testing.assert_array_equal(filt_data.y_err, np.asarray([2, 6], dtype=float))

        filt_data = self.analysis._data(series_name="curve2")
        np.testing.assert_array_equal(filt_data.x, np.asarray([2], dtype=float))
        np.testing.assert_array_equal(filt_data.y, np.asarray([2], dtype=float))
        np.testing.assert_array_equal(filt_data.y_err, np.asarray([4], dtype=float))

        filt_data = self.analysis._data(series_name="curve3")
        np.testing.assert_array_equal(filt_data.x, np.asarray([4, 5], dtype=float))
        np.testing.assert_array_equal(filt_data.y, np.asarray([4, 5], dtype=float))
        np.testing.assert_array_equal(filt_data.y_err, np.asarray([8, 10], dtype=float))

    def test_formatting_options(self):
        """Test option formatter."""
        test_options = {
            "p0": [0, 1, 2, 3, 4],
            "bounds": [(-1, 1), (-2, 2), (-3, 3), (-4, 4), (-5, 5)],
            "other_value": "test",
        }
        formatted_options = self.analysis._format_fit_options(**test_options)

        ref_options = {
            "p0": {"p0": 0, "p1": 1, "p2": 2, "p3": 3, "p4": 4},
            "bounds": {"p0": (-1, 1), "p1": (-2, 2), "p2": (-3, 3), "p3": (-4, 4), "p4": (-5, 5)},
            "other_value": "test",
        }
        self.assertDictEqual(formatted_options, ref_options)

        test_invalid_options = {
            "p0": {"invalid_key1": 0, "invalid_key2": 2, "invalid_key3": 3, "invalid:_key4": 4}
        }
        with self.assertRaises(AnalysisError):
            self.analysis._format_fit_options(**test_invalid_options)


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
                    fit_func=lambda x, p0, p1, p2, p3: fit_function.exponential_decay(
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
            func=fit_function.exponential_decay,
            xvals=self.xvalues,
            param_dict={"amp": ref_p0, "lamb": ref_p1, "x0": ref_p2, "baseline": ref_p3},
        )
        results, _ = analysis._run_analysis(
            test_data,
            p0={"p0": ref_p0, "p1": ref_p1, "p2": ref_p2, "p3": ref_p3},
            curve_fitter=multi_curve_fit,
            data_processor=probability(outcome="1"),
            x_key="xval",
            plot=False,
            axis=None,
            xlabel="x value",
            ylabel="y value",
            fit_reports=None,
            return_data_points=False,
        )
        result = results[0]

        ref_popt = np.asarray([ref_p0, ref_p1, ref_p2, ref_p3])

        # check result data
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
                    fit_func=lambda x, p0, p1, p2, p3: fit_function.exponential_decay(
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
            func=fit_function.exponential_decay,
            xvals=self.xvalues,
            param_dict={"amp": ref_p0, "lamb": ref_p1, "x0": ref_p2, "baseline": ref_p3},
        )

        # Try to fit with infeasible parameter boundary. This should fail.
        results, _ = analysis._run_analysis(
            test_data,
            p0={"p0": ref_p0, "p1": ref_p1, "p2": ref_p2, "p3": ref_p3},
            bounds={"p0": [-10, 0], "p1": [-10, 0], "p2": [-10, 0], "p3": [-10, 0]},
            curve_fitter=multi_curve_fit,
            data_processor=probability(outcome="1"),
            x_key="xval",
            plot=False,
            axis=None,
            xlabel="x value",
            ylabel="y value",
            fit_reports=None,
            return_data_points=True,
        )
        result = results[0]

        self.assertFalse(result["success"])

        ref_result_keys = ["analysis_type", "error_message", "success", "raw_data"]
        self.assertSetEqual(set(result.keys()), set(ref_result_keys))

    def test_run_two_curves_with_same_fitfunc(self):
        """Test analysis for two curves. Curves shares fit model."""
        analysis = create_new_analysis(
            series=[
                SeriesDef(
                    name="curve1",
                    fit_func=lambda x, p0, p1, p2, p3, p4: fit_function.exponential_decay(
                        x, amp=p0, lamb=p1, x0=p3, baseline=p4
                    ),
                    filter_kwargs={"exp": 0},
                ),
                SeriesDef(
                    name="curve1",
                    fit_func=lambda x, p0, p1, p2, p3, p4: fit_function.exponential_decay(
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
            func=fit_function.exponential_decay,
            xvals=self.xvalues,
            param_dict={"amp": ref_p0, "lamb": ref_p1, "x0": ref_p3, "baseline": ref_p4},
            exp=0,
        )

        test_data1 = simulate_output_data(
            func=fit_function.exponential_decay,
            xvals=self.xvalues,
            param_dict={"amp": ref_p0, "lamb": ref_p2, "x0": ref_p3, "baseline": ref_p4},
            exp=1,
        )

        # merge two experiment data
        for datum in test_data1.data():
            test_data0.add_data(datum)

        results, _ = analysis._run_analysis(
            test_data0,
            p0={"p0": ref_p0, "p1": ref_p1, "p2": ref_p2, "p3": ref_p3, "p4": ref_p4},
            curve_fitter=multi_curve_fit,
            data_processor=probability(outcome="1"),
            x_key="xval",
            plot=False,
            axis=None,
            xlabel="x value",
            ylabel="y value",
            fit_reports=None,
            return_data_points=False,
        )
        result = results[0]

        ref_popt = np.asarray([ref_p0, ref_p1, ref_p2, ref_p3, ref_p4])

        # check result data
        np.testing.assert_array_almost_equal(result["popt"], ref_popt, decimal=self.err_decimal)

    def test_run_two_curves_with_two_fitfuncs(self):
        """Test analysis for two curves. Curves shares fit parameters."""
        analysis = create_new_analysis(
            series=[
                SeriesDef(
                    name="curve1",
                    fit_func=lambda x, p0, p1, p2, p3: fit_function.cos(
                        x, amp=p0, freq=p1, phase=p2, baseline=p3
                    ),
                    filter_kwargs={"exp": 0},
                ),
                SeriesDef(
                    name="curve2",
                    fit_func=lambda x, p0, p1, p2, p3: fit_function.sin(
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
            func=fit_function.cos,
            xvals=self.xvalues,
            param_dict={"amp": ref_p0, "freq": ref_p1, "phase": ref_p2, "baseline": ref_p3},
            exp=0,
        )

        test_data1 = simulate_output_data(
            func=fit_function.sin,
            xvals=self.xvalues,
            param_dict={"amp": ref_p0, "freq": ref_p1, "phase": ref_p2, "baseline": ref_p3},
            exp=1,
        )

        # merge two experiment data
        for datum in test_data1.data():
            test_data0.add_data(datum)

        results, _ = analysis._run_analysis(
            test_data0,
            p0={"p0": ref_p0, "p1": ref_p1, "p2": ref_p2, "p3": ref_p3},
            curve_fitter=multi_curve_fit,
            data_processor=probability(outcome="1"),
            x_key="xval",
            plot=False,
            axis=None,
            xlabel="x value",
            ylabel="y value",
            fit_reports=None,
            return_data_points=False,
        )
        result = results[0]

        ref_popt = np.asarray([ref_p0, ref_p1, ref_p2, ref_p3])

        # check result data
        np.testing.assert_array_almost_equal(result["popt"], ref_popt, decimal=self.err_decimal)
