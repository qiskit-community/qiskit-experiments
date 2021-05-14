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
from qiskit_experiments.analysis import CurveAnalysis, SeriesDef
from qiskit_experiments.analysis import fit_functions
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
            fit_functions.exponential_decay, self.xvalues, ref_p0, ref_p1, ref_p2, ref_p3
        )
        results, _ = analysis._run_analysis(test_data, p0=[ref_p0, ref_p1, ref_p2, ref_p3])
        result = results[0]

        ref_popt = np.asarray([ref_p0, ref_p1, ref_p2, ref_p3])

        # check result data
        self.assertTrue(result["success"])

        np.testing.assert_array_almost_equal(result["popt"], ref_popt, decimal=1)
        self.assertEqual(result["dof"], 26)
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
            fit_functions.exponential_decay, self.xvalues, ref_p0, ref_p1, ref_p2, ref_p3
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
            fit_functions.exponential_decay, self.xvalues, ref_p0, ref_p1, ref_p3, ref_p4, exp=0
        )
        test_data1 = simulate_output_data(
            fit_functions.exponential_decay, self.xvalues, ref_p0, ref_p2, ref_p3, ref_p4, exp=1
        )

        # merge two experiment data
        for datum in test_data1.data():
            test_data0.add_data(datum)

        results, _ = analysis._run_analysis(test_data0, p0=[ref_p0, ref_p1, ref_p2, ref_p3, ref_p4])
        result = results[0]

        ref_popt = np.asarray([ref_p0, ref_p1, ref_p2, ref_p3, ref_p4])

        # check result data
        self.assertTrue(result["success"])
        np.testing.assert_array_almost_equal(result["popt"], ref_popt, decimal=1)

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
            fit_functions.cos, self.xvalues, ref_p0, ref_p1, ref_p2, ref_p3, exp=0
        )
        test_data1 = simulate_output_data(
            fit_functions.sin, self.xvalues, ref_p0, ref_p1, ref_p2, ref_p3, exp=1
        )

        # merge two experiment data
        for datum in test_data1.data():
            test_data0.add_data(datum)

        results, _ = analysis._run_analysis(test_data0, p0=[ref_p0, ref_p1, ref_p2, ref_p3])
        result = results[0]

        ref_popt = np.asarray([ref_p0, ref_p1, ref_p2, ref_p3])

        # check result data
        self.assertTrue(result["success"])
        np.testing.assert_array_almost_equal(result["popt"], ref_popt, decimal=1)

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
            fit_functions.exponential_decay,
            self.xvalues,
            ref_p0,
            ref_p1,
            ref_p2,
            ref_p3,
            outcome="0",
        )
        results, _ = analysis._run_analysis(test_data, p0=[ref_p0, ref_p1, ref_p2, ref_p3])
        result = results[0]

        ref_popt = np.asarray([ref_p0, ref_p1, ref_p2, ref_p3])

        # check result data
        self.assertTrue(result["success"])

        np.testing.assert_array_almost_equal(result["popt"], ref_popt, decimal=1)

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

        test_data = simulate_output_data(fit_functions.exponential_decay, self.xvalues, ref_p0)

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

        test_data = simulate_output_data(fit_functions.exponential_decay, self.xvalues, ref_p0)

        results, _ = analysis._run_analysis(test_data, p0=[ref_p0])
        result = results[0]

        self.assertFalse(result["success"])

        ref_result_keys = ["raw_data", "error_message", "success"]
        self.assertSetEqual(set(result.keys()), set(ref_result_keys))
