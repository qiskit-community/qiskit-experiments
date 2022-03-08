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

# pylint: disable=invalid-name, missing-class-docstring, unsubscriptable-object

"""Test curve fitting base class."""
from test.base import QiskitExperimentsTestCase

import numpy as np
from uncertainties import unumpy as unp

from qiskit_experiments.curve_analysis import CurveAnalysis, fit_function
from qiskit_experiments.curve_analysis.curve_data import SeriesDef, FitData, ParameterRepr
from qiskit_experiments.curve_analysis.fit_models import SingleFitFunction, CompositeFitFunction
from qiskit_experiments.data_processing import DataProcessor, Probability
from qiskit_experiments.exceptions import AnalysisError
from qiskit_experiments.framework import ExperimentData, AnalysisResultData, CompositeAnalysis


class TestCompositeFunction(QiskitExperimentsTestCase):
    """Test behavior of CompositeFunction which is a core object of CurveAnalysis.

    This is new fit function wrapper introduced in Qiskit Experiments 0.3.
    This function-like object should manage parameter assignment and mapping to
    manage multiple sub functions (curves) for multi-objective optimization.
    """

    def test_single_function(self):
        """A simple testcase for having only single fit function."""

        def child_function(x, p0, p1):
            return p0 * x + p1

        function = SingleFitFunction(
            fit_functions=[child_function],
            signatures=[["p0", "p1"]],
            fit_models=["p0 x + p1"],
        )

        self.assertListEqual(function.signature, ["p0", "p1"])
        self.assertEqual(function.fit_model, "p0 x + p1")
        self.assertEqual(repr(function), "SingleFitFunction(x, p0, p1)")

        x = np.linspace(0, 1, 10)
        p0 = 1
        p1 = 2
        ref_y = child_function(x, p0, p1)
        test_y = function(x, p0, p1)

        np.testing.assert_array_equal(ref_y, test_y)

    def test_single_function_parameter_fixed(self):
        """Test when some parameters are fixed."""

        def child_function(x, p0, p1):
            return p0 * x + p1

        function = SingleFitFunction(
            fit_functions=[child_function],
            signatures=[["p0", "p1"]],
            fixed_parameters=["p0"],
        )

        self.assertListEqual(function.signature, ["p1"])
        self.assertEqual(repr(function), "SingleFitFunction(x, p1; @ Fixed p0)")

        x = np.linspace(0, 1, 10)
        p0 = 1
        p1 = 2
        ref_y = child_function(x, p0, p1)

        # Need to call parameter binding here
        function.bind_parameters(p0=p0)
        test_y = function(x, p1)

        np.testing.assert_array_equal(ref_y, test_y)

    def test_multiple_functions(self):
        """Test with multiple functions."""

        def child_function1(x, p0, p1):
            return p0 * x + p1

        def child_function2(x, p0, p2):
            return p0 * x - p2

        function = CompositeFitFunction(
            fit_functions=[child_function1, child_function2],
            signatures=[["p0", "p1"], ["p0", "p2"]],
        )

        self.assertListEqual(function.signature, ["p0", "p1", "p2"])

        x1 = np.linspace(0, 1, 10)
        x2 = np.linspace(2, 3, 10)
        p0 = 1
        p1 = 2
        p2 = 3
        ref_y1 = child_function1(x1, p0, p1)
        ref_y2 = child_function2(x2, p0, p2)

        data_index = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
        ref_y = np.zeros(ref_y1.size + ref_y2.size)
        ref_y[data_index == 0] = ref_y1
        ref_y[data_index == 1] = ref_y2

        # Need to set data index
        function.data_allocation = data_index
        test_y = function(np.r_[x1, x2], p0, p1, p2)

        np.testing.assert_array_equal(ref_y, test_y)

    def test_multiple_functions_with_fixed_parameter(self):
        """Test with multiple functions while some parameters are fixed."""

        def child_function1(x, p0, p1):
            return p0 * x + p1

        def child_function2(x, p0, p2):
            return p0 * x - p2

        function = CompositeFitFunction(
            fit_functions=[child_function1, child_function2],
            signatures=[["p0", "p1"], ["p0", "p2"]],
            fixed_parameters=["p1"],
        )

        self.assertListEqual(function.signature, ["p0", "p2"])

        x1 = np.linspace(0, 1, 10)
        x2 = np.linspace(2, 3, 10)
        p0 = 1
        p1 = 2
        p2 = 3
        ref_y1 = child_function1(x1, p0, p1)
        ref_y2 = child_function2(x2, p0, p2)

        data_index = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
        ref_y = np.zeros(ref_y1.size + ref_y2.size)
        ref_y[data_index == 0] = ref_y1
        ref_y[data_index == 1] = ref_y2

        # Need to set data index and fixed parameter here
        function.data_allocation = data_index
        function.bind_parameters(p1=p1)
        test_y = function(np.r_[x1, x2], p0, p2)

        np.testing.assert_array_equal(ref_y, test_y)


class TestCurveFit(QiskitExperimentsTestCase):
    """Test core fitting functionality by bypassing analysis framework.

    CurveAnalysis can provide fit function and fit algorithm via its
    instance property and static method, we can only unittest fitting part.
    This test suite validate fitting function with various situation including
    single curve, mutiple curves, parameter fixsing, etc...
    """

    def test_single_function(self):
        """Test case for single curve entry."""

        def child_function(x, p0, p1):
            return p0 * x + p1

        class MyCurveFit(CurveAnalysis):
            __series__ = [SeriesDef(fit_func=child_function)]

        instance = MyCurveFit()

        x = np.linspace(0, 1, 10)
        p0 = 1
        p1 = 2
        fake_outcome = child_function(x, p0, p1)

        fit_func = instance.fit_model
        result = instance.curve_fit(
            func=fit_func,
            xdata=x,
            ydata=fake_outcome,
            sigma=np.zeros_like(fake_outcome),
            p0={"p0": 0.9, "p1": 2.1},
            bounds={"p0": (0, 2), "p1": (1, 3)},
        )
        self.assertIsInstance(result, FitData)

        self.assertEqual(result.fit_model, "not defined")
        self.assertEqual(result.popt_keys, ["p0", "p1"])
        self.assertEqual(result.dof, 8)
        np.testing.assert_array_almost_equal(unp.nominal_values(result.popt), [p0, p1])

        # test if values are operable
        p0_val = result.fitval("p0")
        p1_val = result.fitval("p1")

        new_quantity = p0_val + p1_val
        self.assertAlmostEqual(new_quantity.s, np.sqrt(p0_val.s**2 + p1_val.s**2))

    def test_multiple_functions(self):
        """Test case for multiple curve entries."""

        def child_function1(x, p0, p1):
            return p0 * x + p1

        def child_function2(x, p0, p2):
            return p0 * x - p2

        class MyCurveFit(CurveAnalysis):
            __series__ = [
                SeriesDef(
                    fit_func=child_function1,
                    model_description="p0 x + p1",
                ),
                SeriesDef(
                    fit_func=child_function2,
                    model_description="p0 x - p2",
                ),
            ]

        instance = MyCurveFit()

        x1 = np.linspace(0, 1, 10)
        x2 = np.linspace(2, 3, 10)
        p0 = 1
        p1 = 2
        p2 = 3
        ref_y1 = child_function1(x1, p0, p1)
        ref_y2 = child_function2(x2, p0, p2)

        data_index = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
        fake_outcome = np.zeros(ref_y1.size + ref_y2.size)
        fake_outcome[data_index == 0] = ref_y1
        fake_outcome[data_index == 1] = ref_y2

        fit_func = instance.fit_model
        fit_func.data_allocation = data_index

        result = instance.curve_fit(
            func=fit_func,
            xdata=np.r_[x1, x2],
            ydata=fake_outcome,
            sigma=np.zeros_like(fake_outcome),
            p0={"p0": 0.9, "p1": 2.1, "p2": 2.9},
            bounds={"p0": (0, 2), "p1": (1, 3), "p2": (2, 4)},
        )

        self.assertEqual(result.fit_model, "p0 x + p1,p0 x - p2")
        self.assertEqual(result.popt_keys, ["p0", "p1", "p2"])
        np.testing.assert_array_almost_equal(unp.nominal_values(result.popt), [p0, p1, p2])

    def test_assert_dof_error(self):
        """Test raise an DOF error when input data size is too small."""

        def child_function(x, p0, p1):
            return p0 * x + p1

        class MyCurveFit(CurveAnalysis):
            __series__ = [SeriesDef(fit_func=child_function)]

        instance = MyCurveFit()

        x = np.array([2])  # DOF = 0
        p0 = 1
        p1 = 2
        fake_outcome = child_function(x, p0, p1)

        fit_func = instance.fit_model
        with self.assertRaises(AnalysisError):
            instance.curve_fit(
                func=fit_func,
                xdata=x,
                ydata=fake_outcome,
                sigma=np.zeros_like(fake_outcome),
                p0={"p0": 0.9, "p1": 2.1},
                bounds={"p0": (0, 2), "p1": (1, 3)},
            )

    def test_assert_invalid_fit(self):
        """Test scipy solver error is converted into AnalysisError."""

        def child_function(x, p0, p1):
            return p0 * x + p1

        class MyCurveFit(CurveAnalysis):
            __series__ = [SeriesDef(fit_func=child_function)]

        instance = MyCurveFit()

        x = np.linspace(0, 1, 10)
        p0 = 1
        p1 = 2
        fake_outcome = child_function(x, p0, p1)

        fit_func = instance.fit_model
        with self.assertRaises(AnalysisError):
            instance.curve_fit(
                func=fit_func,
                xdata=x,
                ydata=fake_outcome,
                sigma=np.zeros_like(fake_outcome),
                p0={"p0": 0, "p1": 2.1},
                bounds={"p0": (-1, 0), "p1": (-1, 0)},  # impossible to fit within this range
            )

    def test_assert_fit_with_bare_calback(self):
        """Test raise error when normal callback is set."""

        def child_function(x, p0, p1):
            return p0 * x + p1

        class MyCurveFit(CurveAnalysis):
            __series__ = [SeriesDef(fit_func=child_function)]

        instance = MyCurveFit()

        x = np.linspace(0, 1, 10)
        p0 = 1
        p1 = 2
        fake_outcome = child_function(x, p0, p1)

        with self.assertRaises(AnalysisError):
            instance.curve_fit(
                func=child_function,  # cannot manage parameter mapping and metadata
                xdata=x,
                ydata=fake_outcome,
                sigma=np.zeros_like(fake_outcome),
                p0={"p0": 0, "p1": 2.1},
                bounds={"p0": (-1, 0), "p1": (-1, 0)},  # impossible to fit within this range
            )

    def test_assert_invalid_fixed_parameter(self):
        """Test we cannot create invalid analysis instance with wrong fixed value name."""
        with self.assertRaises(AnalysisError):
            # pylint: disable=unused-variable
            class InvalidAnalysis(CurveAnalysis):
                __series__ = [
                    SeriesDef(
                        fit_func=lambda x, p0: x + p0,
                    )
                ]
                __fixed_parameters__ = ["not_existing"]


class CurveAnalysisTestCase(QiskitExperimentsTestCase):
    """A baseclass for CurveAnalysis unittest."""

    seeds = 123

    @classmethod
    def single_sampler(cls, xvalues, yvalues, shots=10000, **metadata):
        """A helper function to generate experiment data."""
        rng = np.random.default_rng(seed=cls.seeds)
        counts = rng.binomial(shots, yvalues)
        data = [
            {
                "counts": {"0": shots - c, "1": c},
                "metadata": {"xval": xi, "qubits": (0,), **metadata},
            }
            for xi, c in zip(xvalues, counts)
        ]

        return data

    @classmethod
    def parallel_sampler(cls, xvalues, yvalues1, yvalues2, shots=10000):
        """A helper function to generate fake parallel experiment data."""
        rng = np.random.default_rng(seed=cls.seeds)

        data = []
        for xi, p1, p2 in zip(xvalues, yvalues1, yvalues2):
            cs = rng.multinomial(
                shots, [(1 - p1) * (1 - p2), p1 * (1 - p2), (1 - p1) * p2, p1 * p2]
            )
            circ_data = {
                "counts": {"00": cs[0], "01": cs[1], "10": cs[2], "11": cs[3]},
                "metadata": {
                    "composite_index": [0, 1],
                    "composite_metadata": [{"xval": xi}, {"xval": xi}],
                    "composite_qubits": [[0], [1]],
                    "composite_clbits": [[0], [1]],
                },
            }
            data.append(circ_data)

        return data


class TestCurveAnalysisUnit(CurveAnalysisTestCase):
    """Unittest of CurveAnalysis functionality."""

    def setUp(self):
        super().setUp()

        # Description of test setting
        #
        # - This model contains three curves, namely, curve1, curve2, curve3
        # - Each curve can be represented by the same function
        # - Parameter amp and baseline are shared among all curves
        # - Each curve has unique lamb
        # - In total 5 parameters in the fit, namely, p0, p1, p2, p3
        #
        class MyAnalysis(CurveAnalysis):
            """Test analysis"""

            # Note that series def function can take different argument now.
            # The signature of composite function is generated on the fly.
            __series__ = [
                SeriesDef(
                    name="curve1",
                    fit_func=lambda x, p0, p1, p4: fit_function.exponential_decay(
                        x, amp=p0, lamb=p1, baseline=p4
                    ),
                    filter_kwargs={"type": 1, "valid": True},
                    model_description=r"p_0 * \exp(p_1 x) + p4",
                ),
                SeriesDef(
                    name="curve2",
                    fit_func=lambda x, p0, p2, p4: fit_function.exponential_decay(
                        x, amp=p0, lamb=p2, baseline=p4
                    ),
                    filter_kwargs={"type": 2, "valid": True},
                    model_description=r"p_0 * \exp(p_2 x) + p4",
                ),
                SeriesDef(
                    name="curve3",
                    fit_func=lambda x, p0, p3, p4: fit_function.exponential_decay(
                        x, amp=p0, lamb=p3, baseline=p4
                    ),
                    filter_kwargs={"type": 3, "valid": True},
                    model_description=r"p_0 * \exp(p_3 x) + p4",
                ),
            ]

        self.analysis_cls = MyAnalysis

    def test_parsed_fit_params(self):
        """Test parsed fit params."""
        instance = self.analysis_cls()

        # Note that parameters are ordered according to the following rule.
        # 1. Take series[0] and add its fittting parameters
        # 2. Take next series and its fitting parameters if not exist in the list
        # 3. Repeat until the last series
        self.assertListEqual(instance.parameters, ["p0", "p1", "p4", "p2", "p3"])

    def test_parsed_init_guess(self):
        """Test parsed initial guess and boundaries."""
        instance = self.analysis_cls()

        ref = {"p0": None, "p1": None, "p2": None, "p3": None, "p4": None}
        self.assertDictEqual(instance.options.p0, ref)
        self.assertDictEqual(instance.options.bounds, ref)

    def test_data_extraction(self):
        """Test data extraction method."""
        shots = 5000000  # something big for data generation unittest

        instance = self.analysis_cls()
        instance.set_options(x_key="xval")

        def data_processor(datum):
            count = datum["counts"].get("1", 0)
            pmean = count / shots
            return pmean, pmean * (1 - pmean) / shots

        x = np.linspace(1.0, 5.0, 10)
        y1 = fit_function.exponential_decay(x, amp=1.0)
        y2 = fit_function.exponential_decay(x, amp=0.5)

        test_data_y1 = self.single_sampler(xvalues=x, yvalues=y1, shots=shots, type=1, valid=True)
        test_data_y2 = self.single_sampler(xvalues=x, yvalues=y2, shots=shots, type=2, valid=False)

        expdata = ExperimentData()
        expdata.add_data(test_data_y1 + test_data_y2)

        instance._extract_curves(experiment_data=expdata, data_processor=data_processor)
        raw_data = instance._data(label="raw_data")

        # check x value
        xdata = raw_data.x
        ref_x = np.r_[x, x]
        np.testing.assert_array_equal(xdata, ref_x)

        # check y value
        ydata = raw_data.y
        ref_y = np.r_[y1, y2]
        np.testing.assert_array_almost_equal(ydata, ref_y, decimal=3)

        # check sigma
        sigma = raw_data.y_err
        ref_sigma = np.r_[y1 * (1 - y1) / shots, y2 * (1 - y2) / shots]
        np.testing.assert_array_almost_equal(sigma, ref_sigma, decimal=3)

        # check data index
        index = raw_data.data_index
        ref_index = np.r_[np.full(10, 0), np.full(10, -1)]  # second value doesn't match; -1
        np.testing.assert_array_equal(index, ref_index)

    def test_get_subset(self):
        """Test that get subset data from full data array."""
        instance = self.analysis_cls()
        instance.set_options(x_key="xval")

        fake_data = [
            {"data": 1, "metadata": {"xval": 1, "type": 1, "valid": True}},
            {"data": 2, "metadata": {"xval": 2, "type": 2, "valid": True}},
            {"data": 3, "metadata": {"xval": 3, "type": 1, "valid": True}},
            {"data": 4, "metadata": {"xval": 4, "type": 3, "valid": True}},
            {"data": 5, "metadata": {"xval": 5, "type": 3, "valid": True}},
            {"data": 6, "metadata": {"xval": 6, "type": 4, "valid": True}},  # this if fake
        ]
        expdata = ExperimentData()
        expdata.add_data(fake_data)

        def data_processor(datum):
            return datum["data"], datum["data"] * 2

        instance._extract_curves(expdata, data_processor=data_processor)

        filt_data = instance._data(series_name="curve1")
        np.testing.assert_array_equal(filt_data.x, np.asarray([1, 3], dtype=float))
        np.testing.assert_array_equal(filt_data.y, np.asarray([1, 3], dtype=float))
        np.testing.assert_array_equal(filt_data.y_err, np.asarray([2, 6], dtype=float))

        filt_data = instance._data(series_name="curve2")
        np.testing.assert_array_equal(filt_data.x, np.asarray([2], dtype=float))
        np.testing.assert_array_equal(filt_data.y, np.asarray([2], dtype=float))
        np.testing.assert_array_equal(filt_data.y_err, np.asarray([4], dtype=float))

        filt_data = instance._data(series_name="curve3")
        np.testing.assert_array_equal(filt_data.x, np.asarray([4, 5], dtype=float))
        np.testing.assert_array_equal(filt_data.y, np.asarray([4, 5], dtype=float))
        np.testing.assert_array_equal(filt_data.y_err, np.asarray([8, 10], dtype=float))


class TestCurveAnalysisIntegration(CurveAnalysisTestCase):
    """Unittest of CurveAnalysis full functionality.

    Because parameter mapping and fitting feature is already tested in
    TestCompositeFunction and TestCurveFit,
    this test suite focuses on the entire workflow of curve analysis.
    """

    def test_single_function(self):
        """Simple test case with a single curve."""
        p0 = 0.5
        p1 = 3

        data_processor = DataProcessor(input_key="counts", data_actions=[Probability("1")])
        xvalues = np.linspace(0, 1, 100)
        yvalues = fit_function.exponential_decay(xvalues, amp=p0, lamb=p1)

        class MyAnalysis(CurveAnalysis):
            __series__ = [
                SeriesDef(
                    fit_func=lambda x, p0, p1: fit_function.exponential_decay(x, amp=p0, lamb=p1)
                )
            ]

        test_data = self.single_sampler(xvalues, yvalues)
        expdata = ExperimentData()
        expdata.add_data(test_data)

        init_guess = {"p0": 0.4, "p1": 2.9}
        instance = MyAnalysis()
        instance.set_options(
            x_key="xval",
            p0=init_guess,
            result_parameters=[ParameterRepr("p0", "amp"), ParameterRepr("p1", "lamb")],
            data_processor=data_processor,
            plot=False,
        )

        run_expdata = instance.run(expdata, replace_results=False)

        all_parameters = run_expdata.analysis_results("@Parameters_MyAnalysis")
        p0_analyzed = run_expdata.analysis_results("amp")
        p1_analyzed = run_expdata.analysis_results("lamb")

        np.testing.assert_array_almost_equal(all_parameters.value, [p0, p1], decimal=2)
        self.assertAlmostEqual(p0_analyzed.value.n, p0, delta=0.05)
        self.assertAlmostEqual(p1_analyzed.value.n, p1, delta=0.05)

    def test_extra_entry(self):
        """Simple test case analysis add new entry."""
        p0 = 0.5
        p1 = 3

        data_processor = DataProcessor(input_key="counts", data_actions=[Probability("1")])
        xvalues = np.linspace(0, 1, 100)
        yvalues = fit_function.exponential_decay(xvalues, amp=p0, lamb=p1)

        class MyAnalysis(CurveAnalysis):
            __series__ = [
                SeriesDef(
                    fit_func=lambda x, p0, p1: fit_function.exponential_decay(x, amp=p0, lamb=p1)
                )
            ]

            def _extra_database_entry(self, fit_data):
                return [
                    AnalysisResultData(
                        name="new_value",
                        value=fit_data.fitval("p0") + fit_data.fitval("p1"),
                    )
                ]

        test_data = self.single_sampler(xvalues, yvalues)
        expdata = ExperimentData()
        expdata.add_data(test_data)

        init_guess = {"p0": 0.4, "p1": 2.9}
        instance = MyAnalysis()
        instance.set_options(
            x_key="xval",
            p0=init_guess,
            data_processor=data_processor,
            plot=False,
        )

        run_expdata = instance.run(expdata, replace_results=False)

        new_entry = run_expdata.analysis_results("new_value")

        self.assertAlmostEqual(new_entry.value.n, p0 + p1, delta=0.05)

    def test_evaluate_quality(self):
        """Simple test case evaluating quality."""
        p0 = 0.5
        p1 = 3

        data_processor = DataProcessor(input_key="counts", data_actions=[Probability("1")])
        xvalues = np.linspace(0, 1, 100)
        yvalues = fit_function.exponential_decay(xvalues, amp=p0, lamb=p1)

        class MyAnalysis(CurveAnalysis):
            __series__ = [
                SeriesDef(
                    fit_func=lambda x, p0, p1: fit_function.exponential_decay(x, amp=p0, lamb=p1)
                )
            ]

            def _evaluate_quality(self, fit_data):
                return "evaluated!"

        test_data = self.single_sampler(xvalues, yvalues)
        expdata = ExperimentData()
        expdata.add_data(test_data)

        init_guess = {"p0": 0.4, "p1": 2.9}
        instance = MyAnalysis()
        instance.set_options(
            x_key="xval",
            p0=init_guess,
            data_processor=data_processor,
            plot=False,
        )

        run_expdata = instance.run(expdata, replace_results=False)

        entry = run_expdata.analysis_results(0)
        self.assertEqual(entry.quality, "evaluated!")

    def test_curve_analysis_multi_thread(self):
        """Test case for composite analyis.

        Check if analysis works properly when two instances are simultaneously operated
        in the multiple threads. Note that composite function is a class attribute
        thus it should not be modified during the fit.
        """
        p00 = 0.5
        p10 = 3

        p01 = 0.5
        p11 = 4

        data_processor = DataProcessor(input_key="counts", data_actions=[Probability("1")])
        xvalues = np.linspace(0, 1, 100)
        yvalues_a = fit_function.exponential_decay(xvalues, amp=p00, lamb=p10)
        yvalues_b = fit_function.exponential_decay(xvalues, amp=p01, lamb=p11)

        comp_data = self.parallel_sampler(xvalues, yvalues_a, yvalues_b)

        subdata1 = ExperimentData()
        subdata2 = ExperimentData()

        composite_expdata = ExperimentData()
        composite_expdata.metadata["component_child_index"] = [0, 1]
        composite_expdata.add_child_data(subdata1)
        composite_expdata.add_child_data(subdata2)
        composite_expdata.add_data(comp_data)

        class MyAnalysis(CurveAnalysis):
            __series__ = [
                SeriesDef(
                    fit_func=lambda x, p0, p1: fit_function.exponential_decay(x, amp=p0, lamb=p1)
                )
            ]
            __fixed_parameters__ = ["p1"]

            @classmethod
            def _default_options(cls):
                options = super()._default_options()
                options.data_processor = data_processor
                options.plot = False
                options.result_parameters = ["p0"]
                options.p0 = {"p0": 0.49}
                options.bounds = {"p0": (0.4, 0.6)}
                options.p1 = None

                return options

        # Override CompositeFitFunction with different fixed parameters
        sub_analysis1 = MyAnalysis()
        sub_analysis1.set_options(p1=p10)
        sub_analysis2 = MyAnalysis()
        sub_analysis2.set_options(p1=p11)

        instance = CompositeAnalysis([sub_analysis1, sub_analysis2])
        run_expdata = instance.run(composite_expdata, replace_results=False).block_for_results()

        p0_sub1 = run_expdata.child_data(0).analysis_results("p0")
        self.assertAlmostEqual(p0_sub1.value.n, p00, delta=0.05)

        p0_sub2 = run_expdata.child_data(1).analysis_results("p0")
        self.assertAlmostEqual(p0_sub2.value.n, p01, delta=0.05)
