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
import warnings

from test.base import QiskitExperimentsTestCase
from test.fake_experiment import FakeExperiment

import numpy as np
from ddt import data, ddt, unpack

from lmfit.models import ExpressionModel
from qiskit.qobj.utils import MeasLevel

from qiskit_experiments.curve_analysis import CurveAnalysis, CompositeCurveAnalysis
from qiskit_experiments.curve_analysis.curve_data import (
    CurveFitResult,
    ParameterRepr,
    FitOptions,
)
from qiskit_experiments.data_processing import DataProcessor, Probability
from qiskit_experiments.exceptions import AnalysisError
from qiskit_experiments.framework import ExperimentData, AnalysisResultData, CompositeAnalysis


class CurveAnalysisTestCase(QiskitExperimentsTestCase):
    """Base class for testing Curve Analysis subclasses."""

    @staticmethod
    def single_sampler(x, y, shots=10000, seed=123, **metadata):
        """Prepare fake experiment data."""
        rng = np.random.default_rng(seed=seed)
        counts = rng.binomial(shots, y)

        circuit_results = [
            {
                "counts": {"0": shots - count, "1": count},
                "metadata": {"xval": xi, **metadata},
                "shots": 1024,
            }
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


@ddt
class TestCurveAnalysis(CurveAnalysisTestCase):
    """A collection of CurveAnalysis unit tests and integration tests."""

    def test_roundtrip_serialize(self):
        """A testcase for serializing analysis instance."""
        analysis = CurveAnalysis(models=[ExpressionModel(expr="par0 * x + par1", name="test")])
        self.assertRoundTripSerializable(analysis)

    def test_parameters(self):
        """A testcase for getting fit parameters with attribute."""
        analysis = CurveAnalysis(models=[ExpressionModel(expr="par0 * x + par1", name="test")])
        self.assertListEqual(analysis.parameters, ["par0", "par1"])

        analysis.set_options(fixed_parameters={"par0": 1.0})
        self.assertListEqual(analysis.parameters, ["par1"])

    def test_combine_funcs_with_different_parameters(self):
        """A testcase for composing two objectives with different signature."""
        analysis = CurveAnalysis(
            models=[
                ExpressionModel(expr="par0 * x + par1", name="test1"),
                ExpressionModel(expr="par2 * x + par1", name="test2"),
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
            models=[
                ExpressionModel(
                    expr="par0 * x + par1",
                    name="s1",
                ),
                ExpressionModel(
                    expr="par2 * x + par3",
                    name="s2",
                ),
            ]
        )
        analysis.set_options(
            data_processor=DataProcessor("counts", [Probability("1")]),
            data_subfit_map={
                "s1": {"series": 1},
                "s2": {"series": 2},
            },
        )

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*ScatterTable.labels.*")
            warnings.filterwarnings("ignore", message=".*ScatterTable.get_subset_of.*")
            warnings.filterwarnings("ignore", message=".*ScatterTable.data_allocation.*")
            curve_data = analysis._run_data_processing(raw_data=expdata1.data() + expdata2.data())
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
        analysis = CurveAnalysis(models=[ExpressionModel(expr="par0 * x + par1", name="s1")])
        analysis.set_options(
            result_parameters=["par0", ParameterRepr("par1", "Param1", "SomeUnit")]
        )

        covar = np.diag([0.1**2, 0.2**2])

        fit_data = CurveFitResult(
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
            analysis.set_options(plotter=InvalidClass())

    def test_end_to_end_single_function(self):
        """Integration test for single function."""
        init_params = {"amp": 0.5, "tau": 0.3}
        analysis = CurveAnalysis(models=[ExpressionModel(expr="amp * exp(-x/tau)", name="test")])
        analysis.set_options(
            data_processor=DataProcessor(input_key="counts", data_actions=[Probability("1")]),
            p0=init_params,
            result_parameters=["amp", "tau"],
            plot=False,
        )
        amp = 0.5
        tau = 0.3

        x = np.linspace(0, 1, 100)
        y = amp * np.exp(-x / tau)

        test_data = self.single_sampler(x, y)
        result = analysis.run(test_data)
        self.assertExperimentDone(result)

        curve_data = result.artifacts("curve_data").data
        np.testing.assert_array_equal(curve_data.series_name, "test")
        np.testing.assert_array_equal(curve_data.analysis, "CurveAnalysis")
        self.assertEqual(len(curve_data.filter(category="raw")), 100)
        self.assertEqual(len(curve_data.filter(category="formatted")), 100)
        self.assertEqual(len(curve_data.filter(category="fitted")), 100)
        np.testing.assert_array_equal(curve_data.filter(category="raw").x, np.linspace(0, 1, 100))
        np.testing.assert_array_equal(curve_data.filter(category="raw").shots, 1024)
        np.testing.assert_array_equal(curve_data.filter(category="formatted").shots, 1024)
        self.assertTrue(
            np.isnan(np.array(curve_data.filter(category="fitted").shots, dtype=float)).all()
        )
        np.testing.assert_array_equal(
            curve_data.filter(category="fitted").x, np.linspace(0, 1, 100)
        )
        np.testing.assert_array_equal(
            curve_data.filter(category="formatted").x, np.linspace(0, 1, 100)
        )

        fit_data = result.artifacts("fit_summary").data
        self.assertEqual(
            fit_data.model_repr,
            {"test": "amp * exp(-x/tau)"},
        )
        self.assertEqual(fit_data.dof, 98)
        self.assertEqual(fit_data.init_params, init_params)
        self.assertEqual(fit_data.success, True)
        self.assertAlmostEqual(fit_data.params["amp"], 0.5, delta=0.1)
        self.assertAlmostEqual(fit_data.params["tau"], 0.3, delta=0.1)

        self.assertAlmostEqual(result.analysis_results("amp").value.nominal_value, 0.5, delta=0.1)
        self.assertAlmostEqual(result.analysis_results("tau").value.nominal_value, 0.3, delta=0.1)
        self.assertEqual(len(result._figures), 0)

    def test_end_to_end_multi_objective(self):
        """Integration test for multi objective function."""
        init_params = {"amp": 0.5, "freq": 2.1, "phi": 0.3, "base": 0.1}
        analysis = CurveAnalysis(
            models=[
                ExpressionModel(
                    expr="amp * cos(2 * pi * freq * x + phi) + base",
                    name="m1",
                ),
                ExpressionModel(
                    expr="amp * sin(2 * pi * freq * x + phi) + base",
                    name="m2",
                ),
            ]
        )
        analysis.set_options(
            data_processor=DataProcessor(input_key="counts", data_actions=[Probability("1")]),
            data_subfit_map={
                "m1": {"series": "cos"},
                "m2": {"series": "sin"},
            },
            p0=init_params,
            result_parameters=["amp", "freq", "phi", "base"],
            plot=False,
        )
        amp = 0.3
        freq = 2.1
        phi = 0.3
        base = 0.4

        x = np.linspace(0, 1, 100)
        y1 = amp * np.cos(2 * np.pi * freq * x + phi) + base
        y2 = amp * np.sin(2 * np.pi * freq * x + phi) + base

        test_data1 = self.single_sampler(x, y1, series="cos")
        test_data2 = self.single_sampler(x, y2, series="sin")

        expdata = ExperimentData(experiment=FakeExperiment())
        expdata.add_data(test_data1.data())
        expdata.add_data(test_data2.data())
        expdata.metadata["meas_level"] = MeasLevel.CLASSIFIED

        result = analysis.run(expdata)
        self.assertExperimentDone(result)

        fit_data = result.artifacts("fit_summary").data
        self.assertEqual(
            fit_data.model_repr,
            {
                "m1": "amp * cos(2 * pi * freq * x + phi) + base",
                "m2": "amp * sin(2 * pi * freq * x + phi) + base",
            },
        )
        self.assertEqual(fit_data.dof, 196)
        self.assertEqual(fit_data.init_params, init_params)
        self.assertEqual(fit_data.success, True)
        self.assertAlmostEqual(fit_data.params["amp"], amp, delta=0.1)
        self.assertAlmostEqual(fit_data.params["freq"], freq, delta=0.1)
        self.assertAlmostEqual(fit_data.params["phi"], phi, delta=0.1)
        self.assertAlmostEqual(fit_data.params["base"], base, delta=0.1)

        self.assertAlmostEqual(result.analysis_results("amp").value.nominal_value, amp, delta=0.1)
        self.assertAlmostEqual(result.analysis_results("freq").value.nominal_value, freq, delta=0.1)
        self.assertAlmostEqual(result.analysis_results("phi").value.nominal_value, phi, delta=0.1)
        self.assertAlmostEqual(result.analysis_results("base").value.nominal_value, base, delta=0.1)

    def test_end_to_end_single_function_with_fixed_parameter(self):
        """Integration test for fitting with fixed parameter."""
        analysis = CurveAnalysis(models=[ExpressionModel(expr="amp * exp(-x/tau)", name="test")])
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
        result = analysis.run(test_data)
        self.assertExperimentDone(result)

        fit_data = result.artifacts("fit_summary").data
        self.assertEqual(fit_data.init_params, {"amp": 0.5, "tau": 0.3})
        self.assertEqual(fit_data.success, True)
        self.assertEqual(fit_data.params["amp"], 0.5)

        self.assertEqual(result.analysis_results("amp").value.nominal_value, 0.5)
        self.assertEqual(result.analysis_results("amp").value.std_dev, 0.0)
        self.assertAlmostEqual(result.analysis_results("tau").value.nominal_value, 0.3, delta=0.1)

    def test_end_to_end_compute_new_entry(self):
        """Integration test for computing new parameter with error propagation."""

        class CustomAnalysis(CurveAnalysis):
            """Custom analysis class to override result generation."""

            def __init__(self):
                super().__init__(models=[ExpressionModel(expr="amp * exp(-x/tau)", name="test")])

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
        result = analysis.run(test_data)
        self.assertExperimentDone(result)

        new_value = result.analysis_results("new_value").value

        # Use ufloat_params in @Parameters dataclass.
        # This dataclass stores UFloat values with correlation.
        fit_amp = result.artifacts("fit_summary").data.ufloat_params["amp"]
        fit_tau = result.artifacts("fit_summary").data.ufloat_params["tau"]

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
                self._models = [
                    ExpressionModel(
                        expr=f"{self.options.model_var} * amp * exp(-x/tau)",
                        name="test",
                    )
                ]

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
        result = analysis.run(test_data)
        self.assertExperimentDone(result)

        self.assertAlmostEqual(result.analysis_results("amp").value.nominal_value, 0.5, delta=0.1)
        self.assertAlmostEqual(result.analysis_results("tau").value.nominal_value, 0.3, delta=0.1)

    @data((False, "always", 0), (True, "never", 2), (None, "always", 2), (None, "never", 0))
    @unpack
    def test_end_to_end_parallel_analysis(self, plot_flag, figure_flag, n_figures):
        """Integration test for running two curve analyses in parallel, including
        selective figure generation."""

        fit1_p0 = {"amp": 0.5, "tau": 0.3}
        fit2_p0 = {"amp": 0.7, "tau": 0.5}

        analysis1 = CurveAnalysis(models=[ExpressionModel(expr="amp * exp(-x/tau)", name="test")])
        analysis1.set_options(
            data_processor=DataProcessor(input_key="counts", data_actions=[Probability("1")]),
            p0=fit1_p0,
            result_parameters=["amp", "tau"],
            plot=plot_flag,
        )

        analysis2 = CurveAnalysis(models=[ExpressionModel(expr="amp * exp(-x/tau)", name="test")])
        analysis2.set_options(
            data_processor=DataProcessor(input_key="counts", data_actions=[Probability("1")]),
            p0=fit2_p0,
            result_parameters=["amp", "tau"],
            plot=plot_flag,
        )

        composite = CompositeAnalysis(
            [analysis1, analysis2], flatten_results=True, generate_figures=figure_flag
        )
        amp1 = 0.5
        tau1 = 0.3
        amp2 = 0.7
        tau2 = 0.5

        x = np.linspace(0, 1, 100)
        y1 = amp1 * np.exp(-x / tau1)
        y2 = amp2 * np.exp(-x / tau2)

        test_data = self.parallel_sampler(x, y1, y2)
        result = composite.run(test_data)
        self.assertExperimentDone(result)

        self.assertEqual(len(result.artifacts()), 4)
        fit1 = result.artifacts("fit_summary")[0].data
        self.assertEqual(fit1.model_repr, {"test": "amp * exp(-x/tau)"})
        self.assertEqual(fit1.init_params, fit1_p0)
        self.assertAlmostEqual(fit1.params["amp"], amp1, delta=0.1)
        self.assertAlmostEqual(fit1.params["tau"], tau1, delta=0.1)

        fit2 = result.artifacts("fit_summary")[1].data
        self.assertEqual(fit2.model_repr, {"test": "amp * exp(-x/tau)"})
        self.assertEqual(fit2.init_params, fit2_p0)
        self.assertAlmostEqual(fit2.params["amp"], amp2, delta=0.1)
        self.assertAlmostEqual(fit2.params["tau"], tau2, delta=0.1)

        data1 = result.artifacts("curve_data")[0].data
        data2 = result.artifacts("curve_data")[1].data

        identical_cols = ["xval", "series_name", "series_id", "category", "shots", "analysis"]
        self.assertTrue(data1.dataframe[identical_cols].equals(data2.dataframe[identical_cols]))
        self.assertEqual(len(data1), 300)

        np.testing.assert_array_equal(data1.category[:100], "raw")
        np.testing.assert_array_equal(data1.category[100:200], "formatted")
        np.testing.assert_array_equal(data1.category[-100:], "fitted")
        np.testing.assert_array_equal(data1.series_name, "test")
        np.testing.assert_array_equal(data1.series_id, 0)
        np.testing.assert_array_equal(data1.analysis, "CurveAnalysis")
        np.testing.assert_array_equal(data1.x[:100], np.linspace(0, 1, 100))
        np.testing.assert_array_equal(data1.x[100:200], np.linspace(0, 1, 100))
        np.testing.assert_array_equal(data1.x[-100:], np.linspace(0, 1, 100))

        amps = result.analysis_results("amp")
        taus = result.analysis_results("tau")

        self.assertAlmostEqual(amps[0].value.nominal_value, amp1, delta=0.1)
        self.assertAlmostEqual(amps[1].value.nominal_value, amp2, delta=0.1)

        self.assertAlmostEqual(taus[0].value.nominal_value, tau1, delta=0.1)
        self.assertAlmostEqual(taus[1].value.nominal_value, tau2, delta=0.1)

        self.assertEqual(len(result._figures), n_figures)

    def test_selective_figure_generation(self):
        """Test that selective figure generation based on quality works as expected."""

        # analysis with intentionally bad fit
        analysis1 = CurveAnalysis(models=[ExpressionModel(expr="amp * exp(-x)", name="test")])
        analysis1.set_options(
            data_processor=DataProcessor(input_key="counts", data_actions=[Probability("1")]),
            p0={"amp": 0.7},
            result_parameters=["amp"],
        )
        analysis2 = CurveAnalysis(models=[ExpressionModel(expr="amp * exp(-x/tau)", name="test")])
        analysis2.set_options(
            data_processor=DataProcessor(input_key="counts", data_actions=[Probability("1")]),
            p0={"amp": 0.7, "tau": 0.5},
            result_parameters=["amp", "tau"],
        )
        composite = CompositeAnalysis(
            [analysis1, analysis2], flatten_results=False, generate_figures="selective"
        )
        amp1 = 0.7
        tau1 = 0.5
        amp2 = 0.7
        tau2 = 0.5

        x = np.linspace(0, 1, 100)
        y1 = amp1 * np.exp(-x / tau1)
        y2 = amp2 * np.exp(-x / tau2)

        test_data = self.parallel_sampler(x, y1, y2)
        result = composite.run(test_data)
        self.assertExperimentDone(result)

        for res in result.child_data():
            # only generate a figure if the quality is bad
            if res.analysis_results("amp").quality == "bad":
                self.assertEqual(len(res._figures), 1)
            else:
                self.assertEqual(len(res._figures), 0)

    def test_end_to_end_zero_yerr(self):
        """Integration test for an edge case of having zero y error.

        When the error bar is zero, the fit weights to compute residual tend to become larger.
        When the weight is too much significant, the result locally overfits to
        certain data points with smaller or zero y error.
        """
        analysis = CurveAnalysis(models=[ExpressionModel(expr="amp * x**2", name="test")])
        analysis.set_options(
            data_processor=DataProcessor(input_key="counts", data_actions=[Probability("1")]),
            result_parameters=["amp"],
            average_method="sample",  # Use sample average to make some yerr = 0
            plot=False,
            p0={"amp": 0.2},
        )

        amp = 0.3
        x = np.linspace(0, 1, 100)
        y = amp * x**2

        # Replace small y values with zero.
        # Since mock function samples count dictionary from binomial distribution,
        # y=0 (or 1) yield always the same count dictionary
        # and hence y error becomes zero with sample averaging.
        # In this case, amp = 0 may yield the best result.
        y[0] = 0
        y[1] = 0
        y[2] = 0

        test_data1 = self.single_sampler(x, y, seed=123)
        test_data2 = self.single_sampler(x, y, seed=124)
        test_data3 = self.single_sampler(x, y, seed=125)

        expdata = ExperimentData(experiment=FakeExperiment())
        expdata.add_data(test_data1.data())
        expdata.add_data(test_data2.data())
        expdata.add_data(test_data3.data())

        result = analysis.run(expdata)
        self.assertExperimentDone(result)

        for i in range(3):
            self.assertEqual(
                result.data(i),
                {"counts": {"0": 10000, "1": 0}, "metadata": {"xval": i / 99}, "shots": 1024},
            )
            self.assertEqual(
                result.artifacts("curve_data").data.y[i], 0.5 / 10001
            )  # from Beta distribution estimate

        self.assertAlmostEqual(result.analysis_results("amp").value.nominal_value, amp, delta=0.1)

    def test_get_init_params(self):
        """Integration test for getting initial parameter from overview entry."""

        analysis = CurveAnalysis(models=[ExpressionModel(expr="amp * exp(-x/tau)", name="test")])
        analysis.set_options(
            data_processor=DataProcessor(input_key="counts", data_actions=[Probability("1")]),
            p0={"amp": 0.45, "tau": 0.25},
            plot=False,
        )
        amp = 0.5
        tau = 0.3

        x = np.linspace(0, 1, 100)
        y_true = amp * np.exp(-x / tau)

        test_data = self.single_sampler(x, y_true)
        result = analysis.run(test_data)
        self.assertExperimentDone(result)

        overview = result.artifacts("fit_summary").data

        self.assertDictEqual(overview.init_params, {"amp": 0.45, "tau": 0.25})

        y_ref = 0.45 * np.exp(-x / 0.25)
        y_reproduced = analysis.models[0].eval(x=x, **overview.init_params)
        np.testing.assert_array_almost_equal(y_ref, y_reproduced)

    @data(
        (False, "never", 0, "m1", "raw"),
        (True, "never", 1, "m2", "raw"),
        (None, "never", 0, 0, "fitted"),
        (None, "always", 1, 1, "fitted"),
    )
    @unpack
    def test_multi_composite_curve_analysis(self, plot, gen_figures, n_figures, series, category):
        """Integration test for composite curve analysis.

        This analysis consists of two curve fittings for cos and sin series.
        This experiment is executed twice for different setups of "A" and "B".
        """
        analyses = []

        group_names = ["group_A", "group_B"]
        setups = ["setup_A", "setup_B"]
        for group_name, setup in zip(group_names, setups):
            analysis = CurveAnalysis(
                models=[
                    ExpressionModel(
                        expr="amp * cos(2 * pi * freq * x) + b",
                        name="m1",
                    ),
                    ExpressionModel(
                        expr="amp * sin(2 * pi * freq * x) + b",
                        name="m2",
                    ),
                ],
                name=group_name,
            )
            analysis.set_options(
                filter_data={"setup": setup},
                data_subfit_map={
                    "m1": {"type": "cos"},
                    "m2": {"type": "sin"},
                },
                result_parameters=["amp"],
                data_processor=DataProcessor(input_key="counts", data_actions=[Probability("1")]),
            )
            analyses.append(analysis)

        group_analysis = CompositeCurveAnalysis(analyses)
        group_A_p0 = {"amp": 0.3, "freq": 2.1, "b": 0.5}
        group_B_p0 = {"amp": 0.5, "freq": 3.2, "b": 0.5}
        group_analysis.analyses("group_A").set_options(p0=group_A_p0)
        group_analysis.analyses("group_B").set_options(p0=group_B_p0)
        group_analysis.set_options(plot=plot)
        group_analysis._generate_figures = gen_figures

        amp1 = 0.2
        amp2 = 0.4
        b1 = 0.5
        b2 = 0.5
        freq1 = 2.1
        freq2 = 3.2

        x = np.linspace(0, 1, 100)
        y1a = amp1 * np.cos(2 * np.pi * freq1 * x) + b1
        y2a = amp1 * np.sin(2 * np.pi * freq1 * x) + b1
        y1b = amp2 * np.cos(2 * np.pi * freq2 * x) + b2
        y2b = amp2 * np.sin(2 * np.pi * freq2 * x) + b2

        # metadata must contain key for filtering, specified in filter_data option.
        test_data1a = self.single_sampler(x, y1a, type="cos", setup="setup_A")
        test_data2a = self.single_sampler(x, y2a, type="sin", setup="setup_A")
        test_data1b = self.single_sampler(x, y1b, type="cos", setup="setup_B")
        test_data2b = self.single_sampler(x, y2b, type="sin", setup="setup_B")

        expdata = ExperimentData(experiment=FakeExperiment())
        expdata.add_data(test_data1a.data())
        expdata.add_data(test_data2a.data())
        expdata.add_data(test_data1b.data())
        expdata.add_data(test_data2b.data())
        expdata.metadata["meas_level"] = MeasLevel.CLASSIFIED

        result = group_analysis.run(expdata)
        self.assertExperimentDone(result)
        amps = result.analysis_results("amp")

        fit_A = expdata.artifacts("fit_summary").data["group_A"]
        self.assertEqual(
            fit_A.model_repr,
            {"m1": "amp * cos(2 * pi * freq * x) + b", "m2": "amp * sin(2 * pi * freq * x) + b"},
        )
        self.assertEqual(fit_A.init_params, group_A_p0)
        self.assertAlmostEqual(fit_A.params["amp"], amp1, delta=0.1)
        self.assertAlmostEqual(fit_A.params["freq"], freq1, delta=0.1)
        self.assertAlmostEqual(fit_A.params["b"], b1, delta=0.1)

        fit_B = expdata.artifacts("fit_summary").data["group_B"]
        self.assertEqual(
            fit_B.model_repr,
            {"m1": "amp * cos(2 * pi * freq * x) + b", "m2": "amp * sin(2 * pi * freq * x) + b"},
        )
        self.assertEqual(fit_B.init_params, group_B_p0)
        self.assertAlmostEqual(fit_B.params["amp"], amp2, delta=0.1)
        self.assertAlmostEqual(fit_B.params["freq"], freq2, delta=0.1)
        self.assertAlmostEqual(fit_B.params["b"], b2, delta=0.1)

        table_subset = expdata.artifacts("curve_data").data.filter(series=series, category=category)
        self.assertEqual(len(table_subset), 200)
        if isinstance(series, int):
            np.testing.assert_array_equal(table_subset.series_id, series)
        else:
            np.testing.assert_array_equal(table_subset.series_name, series)
        if category == "raw":
            np.testing.assert_array_equal(table_subset.shots, 1024)
        else:
            self.assertTrue(np.isnan(np.array(table_subset.shots, dtype=float)).all())
        np.testing.assert_array_equal(table_subset.category, category)
        np.testing.assert_array_equal(table_subset.analysis[:100], "group_A")
        np.testing.assert_array_equal(table_subset.analysis[-100:], "group_B")
        np.testing.assert_array_equal(
            table_subset.filter(analysis="group_A").x, np.linspace(0, 1, 100)
        )
        np.testing.assert_array_equal(
            table_subset.filter(analysis="group_B").x, np.linspace(0, 1, 100)
        )

        # two entries are generated for group A and group B
        self.assertEqual(len(amps), 2)
        self.assertEqual(amps[0].extra["group"], "group_A")
        self.assertEqual(amps[1].extra["group"], "group_B")
        self.assertAlmostEqual(amps[0].value.n, amp1, delta=0.1)
        self.assertAlmostEqual(amps[1].value.n, amp2, delta=0.1)
        self.assertEqual(len(result._figures), n_figures)


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
