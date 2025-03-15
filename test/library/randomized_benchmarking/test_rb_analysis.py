# This code is part of Qiskit.
#
# (C) Copyright IBM 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Test for randomized benchmarking experiments with running."""

from test.base import QiskitExperimentsTestCase
import numpy as np
import pandas as pd

from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error
from qiskit_experiments.database_service.exceptions import ExperimentEntryNotFound
from qiskit_experiments.library import randomized_benchmarking as rb


class TestEPGAnalysis(QiskitExperimentsTestCase):
    """Test case for EPG calculation from EPC.

    EPG and depolarizing probability p are assumed to have following relationship

        EPG = (2^n - 1) / 2^n · p

    This p is provided to the Aer noise model, thus we verify EPG computation
    by comparing the value with the depolarizing probability.
    """

    @classmethod
    def setUpClass(cls):
        """Run experiments without analysis for test data preparation."""
        super().setUpClass()

        # Setup noise model, including more gate for complicated EPG computation
        # Note that 1Q channel error is amplified to check 1q channel correction mechanism
        cls.p_x = 0.04
        cls.p_h = 0.02
        cls.p_s = 0.0
        cls.p_cx = 0.09
        x_error = depolarizing_error(cls.p_x, 1)
        h_error = depolarizing_error(cls.p_h, 1)
        s_error = depolarizing_error(cls.p_s, 1)
        cx_error = depolarizing_error(cls.p_cx, 2)

        noise_model = NoiseModel()
        noise_model.add_all_qubit_quantum_error(x_error, "x")
        noise_model.add_all_qubit_quantum_error(h_error, "h")
        noise_model.add_all_qubit_quantum_error(s_error, "s")
        noise_model.add_all_qubit_quantum_error(cx_error, "cx")

        # Need level1 for consecutive gate cancellation for reference EPC value calculation
        transpiler_options = {
            "basis_gates": ["x", "h", "s", "cx"],
            "optimization_level": 1,
        }

        # Aer simulator
        backend = AerSimulator(noise_model=noise_model, seed_simulator=123)

        # Prepare experiment data and cache for analysis
        exp_1qrb_q0 = rb.StandardRB(
            physical_qubits=(0,),
            lengths=[1, 10, 30, 50, 80, 120, 150, 200],
            seed=123,
            backend=backend,
        )
        exp_1qrb_q0.set_transpile_options(**transpiler_options)
        expdata_1qrb_q0 = exp_1qrb_q0.run(analysis=None).block_for_results()

        exp_1qrb_q1 = rb.StandardRB(
            physical_qubits=(1,),
            lengths=[1, 10, 30, 50, 80, 120, 150, 200],
            seed=123,
            backend=backend,
        )
        exp_1qrb_q1.set_transpile_options(**transpiler_options)
        expdata_1qrb_q1 = exp_1qrb_q1.run(analysis=None).block_for_results()

        exp_2qrb = rb.StandardRB(
            physical_qubits=(0, 1),
            lengths=[1, 3, 5, 10, 15, 20, 30, 50],
            seed=123,
            backend=backend,
        )
        exp_2qrb.set_transpile_options(**transpiler_options)
        expdata_2qrb = exp_2qrb.run(analysis=None).block_for_results()

        cls.expdata_1qrb_q0 = expdata_1qrb_q0
        cls.expdata_1qrb_q1 = expdata_1qrb_q1
        cls.expdata_2qrb = expdata_2qrb

    def setUp(self):
        """Setup the tests."""
        super().setUp()
        self.assertExperimentDone(self.expdata_1qrb_q0)
        self.assertExperimentDone(self.expdata_1qrb_q1)
        self.assertExperimentDone(self.expdata_2qrb)

    def test_default_epg_ratio(self):
        """Calculate EPG with default ratio dictionary. H and X have the same ratio."""
        analysis = rb.RBAnalysis()
        analysis.set_options(outcome="0")
        result = analysis.run(self.expdata_1qrb_q0, replace_results=False)
        self.assertExperimentDone(result)

        s_epg = result.analysis_results("EPG_s", dataframe=True).iloc[0]
        h_epg = result.analysis_results("EPG_h", dataframe=True).iloc[0]
        x_epg = result.analysis_results("EPG_x", dataframe=True).iloc[0]

        self.assertEqual(s_epg.value.n, 0.0)

        # H and X gate EPG are assumed to be the same, so this underestimate X and overestimate H
        self.assertEqual(h_epg.value.n, x_epg.value.n)
        self.assertLess(x_epg.value.n, self.p_x * 0.5)
        self.assertGreater(h_epg.value.n, self.p_h * 0.5)

    def test_no_epg(self):
        """Calculate no EPGs."""
        analysis = rb.RBAnalysis()
        analysis.set_options(outcome="0", gate_error_ratio=None)
        result = analysis.run(self.expdata_1qrb_q0, replace_results=False)
        self.assertExperimentDone(result)

        with self.assertRaises(ExperimentEntryNotFound):
            result.analysis_results("EPG_s", dataframe=True)

        with self.assertRaises(ExperimentEntryNotFound):
            result.analysis_results("EPG_h", dataframe=True)

        with self.assertRaises(ExperimentEntryNotFound):
            result.analysis_results("EPG_x", dataframe=True)

    def test_with_custom_epg_ratio(self):
        """Calculate no EPGs with custom EPG ratio dictionary."""
        analysis = rb.RBAnalysis()
        analysis.set_options(outcome="0", gate_error_ratio={"x": 2, "h": 1, "s": 0})
        result = analysis.run(self.expdata_1qrb_q0, replace_results=False)
        self.assertExperimentDone(result)

        h_epg = result.analysis_results("EPG_h", dataframe=True).iloc[0]
        x_epg = result.analysis_results("EPG_x", dataframe=True).iloc[0]

        self.assertAlmostEqual(x_epg.value.n, self.p_x * 0.5, delta=0.005)
        self.assertAlmostEqual(h_epg.value.n, self.p_h * 0.5, delta=0.005)

    def test_2q_epg(self):
        """Compute 2Q EPG without correction.

        Since 1Q gates are designed to have comparable EPG with CX gate,
        this will overestimate the error of CX gate.
        """
        analysis = rb.RBAnalysis()
        analysis.set_options(outcome="00")
        result = analysis.run(self.expdata_2qrb, replace_results=False)
        self.assertExperimentDone(result)

        cx_epg = result.analysis_results("EPG_cx", dataframe=True).iloc[0]

        self.assertGreater(cx_epg.value.n, self.p_cx * 0.75)

    def test_2q_epg_with_correction(self):
        """Check that 2Q EPG with 1Q depolarization correction gives a better (smaller) result than
        without the correction."""
        analysis_1qrb_q0 = rb.RBAnalysis()
        analysis_1qrb_q0.set_options(outcome="0", gate_error_ratio={"x": 2, "h": 1, "s": 0})
        result_q0 = analysis_1qrb_q0.run(self.expdata_1qrb_q0, replace_results=False)
        self.assertExperimentDone(result_q0)

        analysis_1qrb_q1 = rb.RBAnalysis()
        analysis_1qrb_q1.set_options(outcome="0", gate_error_ratio={"x": 2, "h": 1, "s": 0})
        result_q1 = analysis_1qrb_q1.run(self.expdata_1qrb_q1, replace_results=False)
        self.assertExperimentDone(result_q1)

        analysis_2qrb = rb.RBAnalysis()
        analysis_2qrb.set_options(
            outcome="00",
        )
        result_2qrb = analysis_2qrb.run(self.expdata_2qrb)
        self.assertExperimentDone(result_2qrb)
        cx_epg_raw = result_2qrb.analysis_results("EPG_cx", dataframe=True).iloc[0]

        analysis_2qrb = rb.RBAnalysis()
        analysis_2qrb.set_options(
            outcome="00",
            epg_1_qubit=pd.concat(
                [
                    result_q0.analysis_results(dataframe=True),
                    result_q1.analysis_results(dataframe=True),
                ]
            ),
        )
        result_2qrb = analysis_2qrb.run(self.expdata_2qrb)
        analysis_2qrb._run_analysis(self.expdata_2qrb)
        self.assertExperimentDone(result_2qrb)
        cx_epg_corrected = result_2qrb.analysis_results("EPG_cx", dataframe=True).iloc[0]
        self.assertLess(
            np.abs(cx_epg_corrected.value.n - self.p_cx * 0.75),
            np.abs(cx_epg_raw.value.n - self.p_cx * 0.75),
        )
