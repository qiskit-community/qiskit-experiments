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

# pylint: disable=invalid-name

"""Test curve fitting base class."""
from test.base import QiskitExperimentsTestCase
from test.fake_experiment import FakeExperiment

import pickle

import numpy as np
import uncertainties
from ddt import ddt, data, unpack
from qiskit.qobj.utils import MeasLevel
from uncertainties import unumpy

from qiskit_experiments.curve_analysis import CurveAnalysis, fit_function
from qiskit_experiments.curve_analysis.curve_data import (
    SeriesDef,
    SolverResult,
    ParameterRepr,
    FitOptions,
)
from qiskit_experiments.curve_analysis.models import CurveModel, CurveSolver
from qiskit_experiments.data_processing import DataProcessor, Probability
from qiskit_experiments.exceptions import AnalysisError
from qiskit_experiments.framework import ExperimentData, AnalysisResultData, CompositeAnalysis


@ddt
class TestCurveModel(QiskitExperimentsTestCase):
    """A collection of test for CurveModel object."""

    def test_single_function(self):
        """A simple testcase for having only single fit function."""
        model = CurveModel(
            name="test",
            series_defs=[SeriesDef(fit_func="par0 * exp(-x / par1) + par2")],
        )

        self.assertListEqual(model.param_names, ["par0", "par1", "par2"])
        self.assertEqual(model.name, "Model(test)")

        x = np.linspace(0, 1, 10)
        par0 = 1.2
        par1 = 2.3
        par2 = 4.5
        ref_y = par0 * np.exp(-x / par1) + par2
        test_y = model.eval(par0=1.2, par1=2.3, par2=4.5, x=x, allocation=np.full(x.size, 0))
        np.testing.assert_array_almost_equal(ref_y, test_y)

    def test_multiple_functions(self):
        """A simple testcase for having two fit functions."""
        model = CurveModel(
            name="test",
            series_defs=[
                SeriesDef(fit_func="par0 * cos(par1 * x) + par2"),
                SeriesDef(fit_func="par0 * sin(par1 * x) + par2"),
            ],
        )

        self.assertListEqual(model.param_names, ["par0", "par1", "par2"])

        x1 = np.linspace(0, 1, 10)
        x2 = np.linspace(0, 1, 15)
        x_composed = np.concatenate((x1, x2))
        par0 = 1.2
        par1 = 2.3
        par2 = 4.5

        ref_y1 = par0 * np.cos(par1 * x1) + par2
        ref_y2 = par0 * np.sin(par1 * x2) + par2
        ref_y_composed = np.concatenate((ref_y1, ref_y2))
        allocation = np.concatenate((np.full(x1.size, 0), np.full(x2.size, 1)))
        test_y = model.eval(par0=1.2, par1=2.3, par2=4.5, x=x_composed, allocation=allocation)
        np.testing.assert_array_almost_equal(ref_y_composed, test_y)

    def test_fit_func_serializable(self):
        """A testcase for computing values with pickled model."""
        model = CurveModel(
            name="test",
            series_defs=[SeriesDef(fit_func="par0 * exp(-x / par1) + par2")],
        )

        x = np.linspace(0, 1, 10)
        y_raw = model.eval(par0=1.2, par1=2.3, par2=4.5, x=x, allocation=np.full(x.size, 0))

        model_roundtrip = pickle.loads(pickle.dumps(model))
        y_rt = model_roundtrip.eval(
            par0=1.2, par1=2.3, par2=4.5, x=x, allocation=np.full(x.size, 0)
        )

        np.testing.assert_array_almost_equal(y_raw, y_rt)

    def test_error_propagation(self):
        """A test case for computing values with parameters with uncertainties."""
        model = CurveModel(
            name="test",
            series_defs=[SeriesDef(fit_func="par0 * exp(-x / par1) + par2")],
        )

        x = np.linspace(0, 1, 10)
        par0 = uncertainties.ufloat(nominal_value=1.2, std_dev=0.3)
        par1 = uncertainties.ufloat(nominal_value=2.3, std_dev=0.4)
        par2 = uncertainties.ufloat(nominal_value=4.5, std_dev=0.5)
        ref_y = par0 * unumpy.exp(-x / par1) + par2  # pylint: disable=no-member

        test_y = model.eval_with_uncertainties(
            x=x,
            params={"par0": par0, "par1": par1, "par2": par2},
            model_index=0,
        )

        np.testing.assert_array_almost_equal(
            unumpy.nominal_values(ref_y), unumpy.nominal_values(test_y)
        )
        np.testing.assert_array_almost_equal(unumpy.std_devs(ref_y), unumpy.std_devs(test_y))

    @data(
        [1.2, 0.5, 0.1],
        [-0.7, 0.1, -0.1],
        [2.1, 0.8, 1.2],
    )
    @unpack
    def test_fit_model_of_single_function(self, par0, par1, par2):
        """A testcase for performing a fitting with single objective function."""
        model = CurveModel(
            name="test",
            series_defs=[SeriesDef(fit_func="par0 * exp(-x / par1) + par2", name="s1")],
        )
        rng = np.random.default_rng(123)

        x = np.linspace(0, 1, 100)
        ref_y = par0 * np.exp(-x / par1) + par2 + rng.normal(scale=0.1, size=x.size)
        sigma = np.full(x.size, np.nan)

        params = model.make_params(par0=par0, par1=par1, par2=par2)
        solver = CurveSolver(model, params=params)

        result = solver.fit(
            x=x,
            y=ref_y,
            sigma=sigma,
            allocation=np.full(x.size, 0),
        )
        self.assertTrue(result.success)
        self.assertAlmostEqual(result.params["par0"], par0, delta=0.03)
        self.assertAlmostEqual(result.params["par1"], par1, delta=0.03)
        self.assertAlmostEqual(result.params["par2"], par2, delta=0.03)
        self.assertDictEqual(result.model_repr, {"s1": "par0 * exp(-x / par1) + par2"})
        self.assertEqual(result.dof, 97)
        self.assertListEqual(result.var_names, ["par0", "par1", "par2"])
        self.assertTrue(np.all(np.isfinite(result.covar)))
        self.assertGreater(result.ufloat_params["par0"].std_dev, 0.0)
        self.assertGreater(result.ufloat_params["par1"].std_dev, 0.0)
        self.assertGreater(result.ufloat_params["par2"].std_dev, 0.0)

    @data(
        [1.2, 0.5, 0.1],
        [-0.7, 0.1, -0.1],
        [2.1, 0.8, 1.2],
    )
    @unpack
    def test_fit_model_of_single_function_with_error(self, par0, par1, par2):
        """A testcase for performing a fitting with single objective function with uncertainty."""
        model = CurveModel(
            name="test",
            series_defs=[SeriesDef(fit_func="par0 * exp(-x / par1) + par2", name="s1")],
        )
        rng = np.random.default_rng(123)

        x = np.linspace(0, 1, 100)
        ref_y = par0 * np.exp(-x / par1) + par2 + rng.normal(scale=0.1, size=x.size)
        sigma = np.full(x.size, 0.1)

        params = model.make_params(par0=par0, par1=par1, par2=par2)
        solver = CurveSolver(model, params=params)

        result = solver.fit(
            x=x,
            y=ref_y,
            sigma=sigma,
            allocation=np.full(x.size, 0),
        )
        self.assertTrue(result.success)
        self.assertAlmostEqual(result.params["par0"], par0, delta=0.03)
        self.assertAlmostEqual(result.params["par1"], par1, delta=0.03)
        self.assertAlmostEqual(result.params["par2"], par2, delta=0.03)
        self.assertTrue(np.all(np.isfinite(result.covar)))
        self.assertGreater(result.ufloat_params["par0"].std_dev, 0.0)
        self.assertGreater(result.ufloat_params["par1"].std_dev, 0.0)
        self.assertGreater(result.ufloat_params["par2"].std_dev, 0.0)

    @data(
        [1.5, 1.6, 0.2],
        [-0.5, 3.2, -0.1],
        [1.0, 0.5, 0.2],
    )
    @unpack
    def test_fit_model_of_multi_function(self, par0, par1, par2):
        """A testcase for performing a fitting with multiple objective function."""
        model = CurveModel(
            name="test",
            series_defs=[
                SeriesDef(fit_func="par0 * cos(par1 * x) + par2", name="s1"),
                SeriesDef(fit_func="par0 * sin(par1 * x) + par2", name="s2"),
            ],
        )
        rng = np.random.default_rng(123)

        x1 = np.linspace(0, 1, 100)
        x2 = np.linspace(0, 2, 100)
        ref_y1 = par0 * np.cos(par1 * x1) + par2 + rng.normal(scale=0.1, size=x1.size)
        ref_y2 = par0 * np.sin(par1 * x2) + par2 + rng.normal(scale=0.1, size=x2.size)
        x_composed = np.concatenate((x1, x2))
        y_composed = np.concatenate((ref_y1, ref_y2))
        allocation = np.concatenate((np.full(x1.size, 0), np.full(x2.size, 1)))
        sigma = np.full(y_composed.size, 0.1)

        params = model.make_params(par0=par0, par1=par1, par2=par2)
        solver = CurveSolver(model, params=params)

        result = solver.fit(
            x=x_composed,
            y=y_composed,
            sigma=sigma,
            allocation=allocation,
        )
        self.assertTrue(result.success)
        self.assertAlmostEqual(result.params["par0"], par0, delta=0.03)
        self.assertAlmostEqual(result.params["par1"], par1, delta=0.03)
        self.assertAlmostEqual(result.params["par2"], par2, delta=0.03)
        self.assertDictEqual(
            result.model_repr,
            {"s1": "par0 * cos(par1 * x) + par2", "s2": "par0 * sin(par1 * x) + par2"},
        )
        self.assertEqual(result.dof, 197)
        self.assertListEqual(result.var_names, ["par0", "par1", "par2"])
        self.assertTrue(np.all(np.isfinite(result.covar)))
        self.assertGreater(result.ufloat_params["par0"].std_dev, 0.0)
        self.assertGreater(result.ufloat_params["par1"].std_dev, 0.0)
        self.assertGreater(result.ufloat_params["par2"].std_dev, 0.0)

    @data(
        [1.2, 0.5, 0.1],
        [-0.7, 0.1, -0.1],
        [2.1, 0.8, 1.2],
    )
    @unpack
    def test_fit_model_of_single_function_fixed_param(self, par0, par1, par2):
        """A testcase for performing a fitting with fixed parameter."""
        model = CurveModel(
            name="test",
            series_defs=[SeriesDef(fit_func="par0 * exp(-x / par1) + par2", name="s1")],
        )
        rng = np.random.default_rng(123)

        x = np.linspace(0, 1, 100)
        ref_y = par0 * np.exp(-x / par1) + par2 + rng.normal(scale=0.1, size=x.size)
        sigma = np.full(x.size, 0.1)

        params = model.make_params(par0=par0, par1=par1, par2=par2)

        # Fix parameter of par0
        params["par0"].vary = False
        solver = CurveSolver(model, params=params)

        result = solver.fit(
            x=x,
            y=ref_y,
            sigma=sigma,
            allocation=np.full(x.size, 0),
        )
        self.assertTrue(result.success)
        self.assertEqual(result.params["par0"], par0)
        self.assertAlmostEqual(result.params["par1"], par1, delta=0.03)
        self.assertAlmostEqual(result.params["par2"], par2, delta=0.03)
        self.assertEqual(result.dof, 98)
        self.assertListEqual(result.var_names, ["par1", "par2"])
        self.assertEqual(result.covar.shape, (2, 2))
        self.assertEqual(result.ufloat_params["par0"].std_dev, 0.0)
        self.assertGreater(result.ufloat_params["par1"].std_dev, 0.0)
        self.assertGreater(result.ufloat_params["par2"].std_dev, 0.0)

    def test_fit_with_invalid_bound(self):
        """A testcase for failing fitting with invalid bounds."""
        model = CurveModel(
            name="test",
            series_defs=[SeriesDef(fit_func="par0 * exp(-x / par1) + par2", name="s1")],
        )
        rng = np.random.default_rng(123)

        par0 = 1.3
        par1 = 0.5
        par2 = 1.2

        x = np.linspace(0, 1, 100)
        ref_y = par0 * np.exp(-x / par1) + par2 + rng.normal(scale=0.1, size=x.size)
        sigma = np.full(x.size, 0.1)

        params = model.make_params(par0=par0, par1=par1, par2=par2)

        # Impossible to fit
        params["par1"].min = -2
        params["par1"].max = -1
        solver = CurveSolver(model, params=params)

        # No crash with raises
        result = solver.fit(
            x=x,
            y=ref_y,
            sigma=sigma,
            allocation=np.full(x.size, 0),
        )
        self.assertFalse(result.success)

    def test_injecting_invalid_equation(self):
        """A testcase for failing evaluation with invalid input equation."""
        model = CurveModel(
            name="test",
            series_defs=[SeriesDef(fit_func="import os; x")],
        )
        with self.assertRaises(NotImplementedError):
            model.eval(x=np.array([0]), allocation=np.array([0]))


class CurveAnalysisTestCase(QiskitExperimentsTestCase):
    """Base class for testing Curve Analysis subclasses."""

    @staticmethod
    def single_sampler(x, y, shots=10000, seed=123, **metadata):
        """Prepare fake experiment data."""
        rng = np.random.default_rng(seed=seed)
        counts = rng.binomial(shots, y)

        circuit_results = [
            {"counts": {"0": shots - count, "1": count}, "metadata": {"xval": xi, **metadata}}
            for xi, count in zip(x, counts)
        ]
        expdata = ExperimentData(experiment=FakeExperiment())
        expdata.add_data(circuit_results)
        expdata.metadata["meas_level"] = MeasLevel.CLASSIFIED

        return expdata

    @staticmethod
    def parallel_sampler(x, y1, y2, shots=10000, seed=123, **metadata):
        """Prepare fake parallel experiment data."""
        rng = np.random.default_rng(seed=seed)

        circuit_results = []
        for xi, p1, p2 in zip(x, y1, y2):
            cs = rng.multinomial(
                shots, [(1 - p1) * (1 - p2), p1 * (1 - p2), (1 - p1) * p2, p1 * p2]
            )
            circ_data = {
                "counts": {"00": cs[0], "01": cs[1], "10": cs[2], "11": cs[3]},
                "metadata": {
                    "composite_index": [0, 1],
                    "composite_metadata": [{"xval": xi, **metadata}, {"xval": xi, **metadata}],
                    "composite_qubits": [[0], [1]],
                    "composite_clbits": [[0], [1]],
                },
            }
            circuit_results.append(circ_data)

        expdata = ExperimentData(experiment=FakeExperiment())
        expdata.add_data(circuit_results)
        expdata.metadata["meas_level"] = MeasLevel.CLASSIFIED

        return expdata


class TestCurveAnalysis(CurveAnalysisTestCase):
    """A collection of CurveAnalysis unit tests and integration tests."""

    def test_roundtrip_serialize(self):
        """A testcase for serializing analysis instance."""
        analysis = CurveAnalysis(series_defs=[SeriesDef(fit_func="par0 * x + par1")])
        self.assertRoundTripSerializable(analysis, check_func=self.json_equiv)

    def test_parameters(self):
        """A testcase for getting fit parameters with attribute."""
        analysis = CurveAnalysis(series_defs=[SeriesDef(fit_func="par0 * x + par1")])
        self.assertListEqual(analysis.parameters, ["par0", "par1"])

        analysis.set_options(fixed_parameters={"par0": 1.0})
        self.assertListEqual(analysis.parameters, ["par1"])

    def test_combine_funcs_with_different_parameters(self):
        """A testcase for composing two objectives with different signature."""
        analysis = CurveAnalysis(
            series_defs=[
                SeriesDef(fit_func="par0 * x + par1"),
                SeriesDef(fit_func="par2 * x + par1"),
            ]
        )
        self.assertListEqual(analysis.parameters, ["par0", "par1", "par2"])

    def test_data_extraction(self):
        """A testcase for extracting data."""
        x = np.linspace(0, 1, 10)
        y1 = 0.1 * x + 0.3
        y2 = 0.2 * x + 0.4
        expdata1 = self.single_sampler(x, y1, shots=1000000, series=1)
        expdata2 = self.single_sampler(x, y2, shots=1000000, series=2)

        analysis = CurveAnalysis(
            series_defs=[
                SeriesDef(fit_func="par0 * x + par1", filter_kwargs={"series": 1}, name="s1"),
                SeriesDef(fit_func="par2 * x + par3", filter_kwargs={"series": 2}, name="s2"),
            ]
        )
        analysis.set_options(data_processor=DataProcessor("counts", [Probability("1")]))

        curve_data = analysis._run_data_processing(
            raw_data=expdata1.data() + expdata2.data(),
            model=analysis._model,
        )
        self.assertListEqual(curve_data.labels, ["s1", "s2"])

        # check data of series1
        sub1 = curve_data.get_subset_of("s1")
        self.assertListEqual(sub1.labels, ["s1"])
        np.testing.assert_array_equal(sub1.x, x)
        np.testing.assert_array_almost_equal(sub1.y, y1, decimal=3)
        np.testing.assert_array_equal(sub1.data_allocation, np.full(x.size, 0))

        # check data of series2
        sub2 = curve_data.get_subset_of("s2")
        self.assertListEqual(sub2.labels, ["s2"])
        np.testing.assert_array_equal(sub2.x, x)
        np.testing.assert_array_almost_equal(sub2.y, y2, decimal=3)
        np.testing.assert_array_equal(sub2.data_allocation, np.full(x.size, 1))

    def test_create_result(self):
        """A testcase for creating analysis result data from fit data."""
        analysis = CurveAnalysis(series_defs=[SeriesDef(fit_func="par0 * x + par1", name="s1")])
        analysis.set_options(
            result_parameters=["par0", ParameterRepr("par1", "Param1", "SomeUnit")]
        )

        covar = np.diag([0.1**2, 0.2**2])

        fit_data = SolverResult(
            method="some_method",
            model_repr={"s1": "par0 * x + par1"},
            success=True,
            params={"par0": 0.3, "par1": 0.4},
            var_names=["par0", "par1"],
            covar=covar,
            reduced_chisq=1.5,
        )

        result_data = analysis._create_analysis_results(fit_data, quality="good", test="hoge")

        # entry name
        self.assertEqual(result_data[0].name, "par0")
        self.assertEqual(result_data[1].name, "Param1")

        # entry value
        self.assertEqual(result_data[0].value.nominal_value, 0.3)
        self.assertEqual(result_data[0].value.std_dev, 0.1)
        self.assertEqual(result_data[1].value.nominal_value, 0.4)
        self.assertEqual(result_data[1].value.std_dev, 0.2)

        # other metadata
        self.assertEqual(result_data[1].quality, "good")
        self.assertEqual(result_data[1].chisq, 1.5)
        ref_meta = {
            "test": "hoge",
            "unit": "SomeUnit",
        }
        self.assertDictEqual(result_data[1].extra, ref_meta)

    def test_invalid_type_options(self):
        """A testcase for failing with invalid options."""
        analysis = CurveAnalysis()

        class InvalidClass:
            """Dummy class."""

            pass

        with self.assertRaises(TypeError):
            analysis.set_options(data_processor=InvalidClass())

        with self.assertRaises(TypeError):
            analysis.set_options(curve_drawer=InvalidClass())

    def test_end_to_end_single_function(self):
        """Integration test for single function."""
        analysis = CurveAnalysis(series_defs=[SeriesDef(fit_func="amp * exp(-x/tau)")])
        analysis.set_options(
            data_processor=DataProcessor(input_key="counts", data_actions=[Probability("1")]),
            p0={"amp": 0.5, "tau": 0.3},
            result_parameters=["amp", "tau"],
            plot=False,
        )
        amp = 0.5
        tau = 0.3

        x = np.linspace(0, 1, 100)
        y = amp * np.exp(-x / tau)

        test_data = self.single_sampler(x, y)
        result = analysis.run(test_data).block_for_results()

        self.assertAlmostEqual(result.analysis_results("amp").value.nominal_value, 0.5, delta=0.1)
        self.assertAlmostEqual(result.analysis_results("tau").value.nominal_value, 0.3, delta=0.1)

    def test_end_to_end_single_function_with_fixed_parameter(self):
        """Integration test for fitting with fixed parameter."""
        analysis = CurveAnalysis(series_defs=[SeriesDef(fit_func="amp * exp(-x/tau)")])
        analysis.set_options(
            data_processor=DataProcessor(input_key="counts", data_actions=[Probability("1")]),
            p0={"tau": 0.3},
            result_parameters=["amp", "tau"],
            fixed_parameters={"amp": 0.5},
            plot=False,
        )
        amp = 0.5
        tau = 0.3

        x = np.linspace(0, 1, 100)
        y = amp * np.exp(-x / tau)

        test_data = self.single_sampler(x, y)
        result = analysis.run(test_data).block_for_results()

        self.assertEqual(result.analysis_results("amp").value.nominal_value, 0.5)
        self.assertEqual(result.analysis_results("amp").value.std_dev, 0.0)
        self.assertAlmostEqual(result.analysis_results("tau").value.nominal_value, 0.3, delta=0.1)

    def test_end_to_end_compute_new_entry(self):
        """Integration test for computing new parameter with error propagation."""

        class CustomAnalysis(CurveAnalysis):
            """Custom analysis class to override result generation."""

            def __init__(self):
                super().__init__(series_defs=[SeriesDef(fit_func="amp * exp(-x/tau)")])

            def _create_analysis_results(self, fit_data, quality, **metadata):
                results = super()._create_analysis_results(fit_data, quality, **metadata)
                u_amp = fit_data.ufloat_params["amp"]
                u_tau = fit_data.ufloat_params["tau"]
                results.append(
                    AnalysisResultData(
                        name="new_value",
                        value=u_amp + u_tau,
                    )
                )
                return results

        analysis = CustomAnalysis()
        analysis.set_options(
            data_processor=DataProcessor(input_key="counts", data_actions=[Probability("1")]),
            p0={"amp": 0.5, "tau": 0.3},
            plot=False,
        )
        amp = 0.5
        tau = 0.3

        x = np.linspace(0, 1, 100)
        y = amp * np.exp(-x / tau)

        test_data = self.single_sampler(x, y)
        result = analysis.run(test_data).block_for_results()

        new_value = result.analysis_results("new_value").value

        # Use ufloat_params in @Parameters dataclass.
        # This dataclass stores UFloat values with correlation.
        fit_amp = result.analysis_results(0).value.ufloat_params["amp"]
        fit_tau = result.analysis_results(0).value.ufloat_params["tau"]

        self.assertEqual(new_value.n, fit_amp.n + fit_tau.n)

        # This is not equal because of fit parameter correlation
        self.assertNotEqual(new_value.s, np.sqrt(fit_amp.s**2 + fit_tau.s**2))
        self.assertEqual(new_value.s, (fit_amp + fit_tau).s)

    def test_end_to_end_create_model_at_run(self):
        """Integration test for dynamically generate model at run time."""

        class CustomAnalysis(CurveAnalysis):
            """Custom analysis class to override model generation."""

            @classmethod
            def _default_options(cls):
                options = super()._default_options()
                options.model_var = None

                return options

            def _initialize(self, experiment_data):
                super()._initialize(experiment_data)

                # Generate model with `model_var` option
                self._model = CurveModel(
                    name="test",
                    series_defs=[
                        SeriesDef(fit_func=f"{self.options.model_var} * amp * exp(-x/tau)")
                    ],
                )

        analysis = CustomAnalysis()
        analysis.set_options(
            data_processor=DataProcessor(input_key="counts", data_actions=[Probability("1")]),
            p0={"amp": 0.5, "tau": 0.3},
            result_parameters=["amp", "tau"],
            plot=False,
            model_var=0.5,
        )
        amp = 0.5
        tau = 0.3

        x = np.linspace(0, 1, 100)
        y = 0.5 * amp * np.exp(-x / tau)

        test_data = self.single_sampler(x, y)
        result = analysis.run(test_data).block_for_results()

        self.assertAlmostEqual(result.analysis_results("amp").value.nominal_value, 0.5, delta=0.1)
        self.assertAlmostEqual(result.analysis_results("tau").value.nominal_value, 0.3, delta=0.1)

    def test_end_to_end_parallel_analysis(self):
        """Integration test for running two curve analyses in parallel."""

        analysis1 = CurveAnalysis(series_defs=[SeriesDef(fit_func="amp * exp(-x/tau)")])
        analysis1.set_options(
            data_processor=DataProcessor(input_key="counts", data_actions=[Probability("1")]),
            p0={"amp": 0.5, "tau": 0.3},
            result_parameters=["amp", "tau"],
            plot=False,
        )

        analysis2 = CurveAnalysis(series_defs=[SeriesDef(fit_func="amp * exp(-x/tau)")])
        analysis2.set_options(
            data_processor=DataProcessor(input_key="counts", data_actions=[Probability("1")]),
            p0={"amp": 0.7, "tau": 0.5},
            result_parameters=["amp", "tau"],
            plot=False,
        )

        composite = CompositeAnalysis([analysis1, analysis2], flatten_results=True)
        amp1 = 0.5
        tau1 = 0.3
        amp2 = 0.7
        tau2 = 0.5

        x = np.linspace(0, 1, 100)
        y1 = amp1 * np.exp(-x / tau1)
        y2 = amp2 * np.exp(-x / tau2)

        test_data = self.parallel_sampler(x, y1, y2)
        result = composite.run(test_data).block_for_results()

        amps = result.analysis_results("amp")
        taus = result.analysis_results("tau")

        self.assertAlmostEqual(amps[0].value.nominal_value, amp1, delta=0.1)
        self.assertAlmostEqual(amps[1].value.nominal_value, amp2, delta=0.1)

        self.assertAlmostEqual(taus[0].value.nominal_value, tau1, delta=0.1)
        self.assertAlmostEqual(taus[1].value.nominal_value, tau2, delta=0.1)

    def test_end_to_end_fit_fail(self):
        """Integration test for failing in fitting with status reported."""

        analysis = CurveAnalysis(series_defs=[SeriesDef(fit_func="amp * exp(-x/tau)")])
        analysis.set_options(
            data_processor=DataProcessor(input_key="counts", data_actions=[Probability("1")]),
            p0={"tau": 0.3},
            bounds={"amp": (-0.5, 0.0)},  # infeasible parameter bound
            result_parameters=["amp", "tau"],
            plot=False,
        )
        amp = 0.5
        tau = 0.3

        x = np.linspace(0, 1, 100)
        y = amp * np.exp(-x / tau)

        test_data = self.single_sampler(x, y)
        result = analysis.run(test_data).block_for_results()

        status = result.analysis_results(0).value
        self.assertFalse(status.success)
        self.assertEqual(status.message, "`x0` is infeasible.")


class TestFitOptions(QiskitExperimentsTestCase):
    """Unittest for fit option object."""

    def test_empty(self):
        """Test if default value is automatically filled."""
        opt = FitOptions(["par0", "par1", "par2"])

        # bounds should be default to inf tuple. otherwise crashes the scipy fitter.
        ref_opts = {
            "p0": {"par0": None, "par1": None, "par2": None},
            "bounds": {
                "par0": (-np.inf, np.inf),
                "par1": (-np.inf, np.inf),
                "par2": (-np.inf, np.inf),
            },
        }

        self.assertDictEqual(opt.options, ref_opts)

    def test_create_option_with_dict(self):
        """Create option and fill with dictionary."""
        opt = FitOptions(
            ["par0", "par1", "par2"],
            default_p0={"par0": 0, "par1": 1, "par2": 2},
            default_bounds={"par0": (0, 1), "par1": (1, 2), "par2": (2, 3)},
        )

        ref_opts = {
            "p0": {"par0": 0.0, "par1": 1.0, "par2": 2.0},
            "bounds": {"par0": (0.0, 1.0), "par1": (1.0, 2.0), "par2": (2.0, 3.0)},
        }

        self.assertDictEqual(opt.options, ref_opts)

    def test_create_option_with_array(self):
        """Create option and fill with array."""
        opt = FitOptions(
            ["par0", "par1", "par2"],
            default_p0=[0, 1, 2],
            default_bounds=[(0, 1), (1, 2), (2, 3)],
        )

        ref_opts = {
            "p0": {"par0": 0.0, "par1": 1.0, "par2": 2.0},
            "bounds": {"par0": (0.0, 1.0), "par1": (1.0, 2.0), "par2": (2.0, 3.0)},
        }

        self.assertDictEqual(opt.options, ref_opts)

    def test_override_partial_dict(self):
        """Create option and override value with partial dictionary."""
        opt = FitOptions(["par0", "par1", "par2"])
        opt.p0.set_if_empty(par1=3)

        ref_opts = {
            "p0": {"par0": None, "par1": 3.0, "par2": None},
            "bounds": {
                "par0": (-np.inf, np.inf),
                "par1": (-np.inf, np.inf),
                "par2": (-np.inf, np.inf),
            },
        }

        self.assertDictEqual(opt.options, ref_opts)

    def test_cannot_override_assigned_value(self):
        """Test cannot override already assigned value."""
        opt = FitOptions(["par0", "par1", "par2"])
        opt.p0.set_if_empty(par1=3)
        opt.p0.set_if_empty(par1=5)

        ref_opts = {
            "p0": {"par0": None, "par1": 3.0, "par2": None},
            "bounds": {
                "par0": (-np.inf, np.inf),
                "par1": (-np.inf, np.inf),
                "par2": (-np.inf, np.inf),
            },
        }

        self.assertDictEqual(opt.options, ref_opts)

    def test_can_override_assigned_value_with_dict_access(self):
        """Test override already assigned value with direct dict access."""
        opt = FitOptions(["par0", "par1", "par2"])
        opt.p0["par1"] = 3
        opt.p0["par1"] = 5

        ref_opts = {
            "p0": {"par0": None, "par1": 5.0, "par2": None},
            "bounds": {
                "par0": (-np.inf, np.inf),
                "par1": (-np.inf, np.inf),
                "par2": (-np.inf, np.inf),
            },
        }

        self.assertDictEqual(opt.options, ref_opts)

    def test_cannot_override_user_option(self):
        """Test cannot override already assigned value."""
        opt = FitOptions(["par0", "par1", "par2"], default_p0={"par1": 3})
        opt.p0.set_if_empty(par1=5)

        ref_opts = {
            "p0": {"par0": None, "par1": 3, "par2": None},
            "bounds": {
                "par0": (-np.inf, np.inf),
                "par1": (-np.inf, np.inf),
                "par2": (-np.inf, np.inf),
            },
        }

        self.assertDictEqual(opt.options, ref_opts)

    def test_set_operation(self):
        """Test if set works and duplicated entry is removed."""
        opt1 = FitOptions(["par0", "par1"], default_p0=[0, 1])
        opt2 = FitOptions(["par0", "par1"], default_p0=[0, 1])
        opt3 = FitOptions(["par0", "par1"], default_p0=[0, 2])

        opts = set()
        opts.add(opt1)
        opts.add(opt2)
        opts.add(opt3)

        self.assertEqual(len(opts), 2)

    def test_detect_invalid_p0(self):
        """Test if invalid p0 raises Error."""
        with self.assertRaises(AnalysisError):
            # less element
            FitOptions(["par0", "par1", "par2"], default_p0=[0, 1])

    def test_detect_invalid_bounds(self):
        """Test if invalid bounds raises Error."""
        with self.assertRaises(AnalysisError):
            # less element
            FitOptions(["par0", "par1", "par2"], default_bounds=[(0, 1), (1, 2)])

        with self.assertRaises(AnalysisError):
            # not min-max tuple
            FitOptions(["par0", "par1", "par2"], default_bounds=[0, 1, 2])

        with self.assertRaises(AnalysisError):
            # max-min tuple
            FitOptions(["par0", "par1", "par2"], default_bounds=[(1, 0), (2, 1), (3, 2)])

    def test_detect_invalid_key(self):
        """Test if invalid key raises Error."""
        opt = FitOptions(["par0", "par1", "par2"])

        with self.assertRaises(AnalysisError):
            opt.p0.set_if_empty(par3=3)

    def test_set_extra_options(self):
        """Add extra fitter options."""
        opt = FitOptions(
            ["par0", "par1", "par2"], default_p0=[0, 1, 2], default_bounds=[(0, 1), (1, 2), (2, 3)]
        )
        opt.add_extra_options(ex1=0, ex2=1)

        ref_opts = {
            "p0": {"par0": 0.0, "par1": 1.0, "par2": 2.0},
            "bounds": {"par0": (0.0, 1.0), "par1": (1.0, 2.0), "par2": (2.0, 3.0)},
            "ex1": 0,
            "ex2": 1,
        }

        self.assertDictEqual(opt.options, ref_opts)

    def test_complicated(self):
        """Test for realistic operations for algorithmic guess with user options."""
        user_p0 = {"par0": 1, "par1": None}
        user_bounds = {"par0": None, "par1": (-100, 100)}

        opt = FitOptions(
            ["par0", "par1", "par2"],
            default_p0=user_p0,
            default_bounds=user_bounds,
        )

        # similar computation in algorithmic guess

        opt.p0.set_if_empty(par0=5)  # this is ignored because user already provided initial guess
        opt.p0.set_if_empty(par1=opt.p0["par0"] * 2 + 3)  # user provided guess propagates

        opt.bounds.set_if_empty(par0=(0, 10))  # this will be set
        opt.add_extra_options(fitter="algo1")

        opt1 = opt.copy()  # copy options while keeping previous values
        opt1.p0.set_if_empty(par2=opt1.p0["par0"] + opt1.p0["par1"])

        opt2 = opt.copy()
        opt2.p0.set_if_empty(par2=opt2.p0["par0"] * 2)  # add another p2 value

        ref_opt1 = {
            "p0": {"par0": 1.0, "par1": 5.0, "par2": 6.0},
            "bounds": {"par0": (0.0, 10.0), "par1": (-100.0, 100.0), "par2": (-np.inf, np.inf)},
            "fitter": "algo1",
        }

        ref_opt2 = {
            "p0": {"par0": 1.0, "par1": 5.0, "par2": 2.0},
            "bounds": {"par0": (0.0, 10.0), "par1": (-100.0, 100.0), "par2": (-np.inf, np.inf)},
            "fitter": "algo1",
        }

        self.assertDictEqual(opt1.options, ref_opt1)
        self.assertDictEqual(opt2.options, ref_opt2)


class TestBackwardCompatibility(QiskitExperimentsTestCase):
    """Test case for backward compatibility."""

    def test_old_fixed_param_attributes(self):
        """Test if old class structure for fixed param is still supported."""

        class _DeprecatedAnalysis(CurveAnalysis):
            __series__ = [
                SeriesDef(
                    fit_func=lambda x, par0, par1, par2, par3: fit_function.exponential_decay(
                        x, amp=par0, lamb=par1, x0=par2, baseline=par3
                    ),
                )
            ]

            __fixed_parameters__ = ["par1"]

            @classmethod
            def _default_options(cls):
                opts = super()._default_options()
                opts.par1 = 2

                return opts

        with self.assertWarns(DeprecationWarning):
            instance = _DeprecatedAnalysis()

        self.assertDictEqual(instance.options.fixed_parameters, {"par1": 2})

    def test_loading_data_with_deprecated_fixed_param(self):
        """Test loading old data with fixed parameters as standalone options."""

        class _DeprecatedAnalysis(CurveAnalysis):
            __series__ = [
                SeriesDef(
                    fit_func=lambda x, par0, par1, par2, par3: fit_function.exponential_decay(
                        x, amp=par0, lamb=par1, x0=par2, baseline=par3
                    ),
                )
            ]

        with self.assertWarns(DeprecationWarning):
            # old option data structure, i.e. fixed param as a standalone option
            # the analysis instance fixed parameters might be set via the experiment instance
            instance = _DeprecatedAnalysis.from_config({"options": {"par1": 2}})

        self.assertDictEqual(instance.options.fixed_parameters, {"par1": 2})

    def test_instantiating_series_def_in_old_format(self):
        """Test instantiating curve analysis with old series def format."""

        class _DeprecatedAnalysis(CurveAnalysis):
            __series__ = [
                SeriesDef(fit_func=lambda x, par0: fit_function.exponential_decay(x, amp=par0))
            ]

        with self.assertWarns(DeprecationWarning):
            instance = _DeprecatedAnalysis()

        # Still works.
        self.assertListEqual(instance.parameters, ["par0"])
