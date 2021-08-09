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
A test for RB analysis.
"""
from qiskit.circuit.library import (
    XGate,
    CXGate,
)
from qiskit.providers.aer import QasmSimulator
from qiskit.providers.aer.noise import NoiseModel
from qiskit.providers.aer.noise.errors.standard_errors import depolarizing_error
from qiskit.test import QiskitTestCase

from qiskit_experiments.library import StandardRB, InterleavedRB

ATOL_DEFAULT = 1e-2
RTOL_DEFAULT = 1e-5


def create_noise_model_1q():
    """Create noise model of depolarizing error for 1q RB.

    Notes:
        Depolarizing parameters are engineered to shorten RB circuit under
        the test for faster completion.

    Returns:
        NoiseModel: depolarizing error noise model
    """
    p1q = 0.12
    noise_model = NoiseModel()
    noise_model.add_all_qubit_quantum_error(depolarizing_error(p1q, 1), "x")
    noise_model.add_all_qubit_quantum_error(depolarizing_error(p1q, 1), "sx")
    return noise_model


def create_noise_model_2q():
    """Create noise model of depolarizing error for 2q RB.

    Notes:
        Depolarizing parameters are engineered to shorten RB circuit under
        the test for faster completion.

    Returns:
        NoiseModel: depolarizing error noise model
    """
    p1q = 0.001
    p2q = 0.02
    noise_model = NoiseModel()
    noise_model.add_all_qubit_quantum_error(depolarizing_error(p1q, 1), "x")
    noise_model.add_all_qubit_quantum_error(depolarizing_error(p1q, 1), "sx")
    noise_model.add_all_qubit_quantum_error(depolarizing_error(p2q, 2), "cx")

    return noise_model


class TestStandardRBAnalysis(QiskitTestCase):
    """
    A test for the analysis of the standard RB experiment
    """

    def setUp(self):
        self.gate_error_ratio = {
            ((0,), "id"): 1,
            ((0,), "rz"): 0,
            ((0,), "sx"): 1,
            ((0,), "x"): 1,
            ((0, 1), "cx"): 1,
        }
        self.transpiled_base_gate = ["cx", "sx", "x"]
        super().setUp()

    def _check_fit_val(self, entry, ref_value=None, ref_stderr=None):
        """A helper method to check fit value entry with tolerance."""
        if ref_value is not None:
            self.assertAlmostEqual(
                entry.value.value,
                ref_value,
                delta=ATOL_DEFAULT,
            )
        if ref_stderr is not None:
            self.assertAlmostEqual(
                entry.value.stderr,
                ref_stderr,
                delta=ATOL_DEFAULT,
            )

    def test_1qubit_standard_rb(self):
        """Executing standard RB experiment and analyze with 1 qubit."""
        noise_model = create_noise_model_1q()
        backend = QasmSimulator()
        rb_exp = StandardRB(
            qubits=[0],
            lengths=list(range(1, 100, 10)),
            num_samples=3,
            seed=100,
        )
        rb_exp.set_analysis_options(gate_error_ratio=self.gate_error_ratio)
        rb_data = rb_exp.run(
            backend,
            noise_model=noise_model,
            seed_simulator=123,
            basis_gates=self.transpiled_base_gate,
        )
        rb_data.block_for_results()

        # alpha entry
        alpha = rb_data.analysis_results("alpha")
        self._check_fit_val(alpha, 0.9735872703302408, 0.00517212083711343)

        # epc entry
        epc = rb_data.analysis_results("EPC")
        self._check_fit_val(epc, 0.01320636483487958, 0.002586060418556715)

        # epg entry
        epg_x = rb_data.analysis_results("EPG_x")
        self._check_fit_val(epg_x, 0.055314196785605056)

    def test_2qubit_standard_rb(self):
        """Executing standard RB experiment and analyze with 2 qubit."""
        noise_model = create_noise_model_2q()
        backend = QasmSimulator()
        rb_exp = StandardRB(
            qubits=[0, 1],
            lengths=list(range(1, 100, 10)),
            num_samples=3,
            seed=100,
        )
        rb_exp.set_analysis_options(gate_error_ratio=self.gate_error_ratio)
        rb_data = rb_exp.run(
            backend,
            noise_model=noise_model,
            seed_simulator=123,
            basis_gates=self.transpiled_base_gate,
        )
        rb_data.block_for_results()

        # alpha entry
        alpha = rb_data.analysis_results("alpha")
        self._check_fit_val(alpha, 0.965548680131692, 0.0008621494403591618)

        # epc entry
        epc = rb_data.analysis_results("EPC")
        self._check_fit_val(epc, 0.02583848990123097, 0.0006466120802693713)

        # epg entry
        epg_x = rb_data.analysis_results("EPG_cx")
        self._check_fit_val(epg_x, 0.01571091672064036)


class TestInterleavedRBAnalysis(QiskitTestCase):
    """
    A test for the analysis of the standard RB experiment
    """

    def setUp(self):
        self.gate_error_ratio = {
            ((0,), "id"): 1,
            ((0,), "rz"): 0,
            ((0,), "sx"): 1,
            ((0,), "x"): 1,
            ((0, 1), "cx"): 1,
        }
        self.interleaved_gates = {"x": XGate(), "cx": CXGate()}
        self.transpiled_base_gate = ["cx", "sx", "x"]
        super().setUp()

    def _check_fit_val(self, entry, ref_value=None, ref_stderr=None):
        """A helper method to check fit value entry with tolerance."""
        if ref_value is not None:
            self.assertAlmostEqual(
                entry.value.value,
                ref_value,
                delta=ATOL_DEFAULT,
            )
        if ref_stderr is not None:
            self.assertAlmostEqual(
                entry.value.stderr,
                ref_stderr,
                delta=ATOL_DEFAULT,
            )

    def test_1qubit_interleaved_rb(self):
        """Executing interleaved RB experiment and analyze with 1 qubit."""
        noise_model = create_noise_model_1q()
        backend = QasmSimulator()
        rb_exp = InterleavedRB(
            interleaved_element=self.interleaved_gates["x"],
            qubits=[0],
            lengths=list(range(1, 100, 10)),
            num_samples=3,
            seed=100,
        )
        rb_exp.set_analysis_options(gate_error_ratio=self.gate_error_ratio)
        rb_data = rb_exp.run(
            backend,
            noise_model=noise_model,
            seed_simulator=123,
            basis_gates=self.transpiled_base_gate,
        )
        rb_data.block_for_results()

        # alpha entry
        alpha = rb_data.analysis_results("alpha")
        self._check_fit_val(alpha, 0.9717858243931459, 0.0020830984944994763)

        # alpha_c entry
        alpha_c = rb_data.analysis_results("alpha_c")
        self._check_fit_val(alpha_c, 0.8585131810858377, 0.014708414678760393)

        # epc entry
        epc = rb_data.analysis_results("EPC")
        ref_epc_systematic_err = 0.07074340945708113
        ref_epc_systematic_bounds = [0.0, 0.14148681891416226]

        self._check_fit_val(epc, 0.07074340945708113, 0.0073542073393801964)

        self.assertAlmostEqual(
            epc.extra["EPC_systematic_err"],
            ref_epc_systematic_err,
            delta=ATOL_DEFAULT,
        )
        self.assertAlmostEqual(
            epc.extra["EPC_systematic_bounds"][0],
            ref_epc_systematic_bounds[0],
            delta=ATOL_DEFAULT,
        )
        self.assertAlmostEqual(
            epc.extra["EPC_systematic_bounds"][1],
            ref_epc_systematic_bounds[1],
            delta=ATOL_DEFAULT,
        )

    def test_2qubit_interleaved_rb(self):
        """Executing interleaved RB experiment and analyze with 2 qubit."""
        noise_model = create_noise_model_2q()
        backend = QasmSimulator()
        rb_exp = InterleavedRB(
            interleaved_element=self.interleaved_gates["cx"],
            qubits=[0, 1],
            lengths=list(range(1, 100, 10)),
            num_samples=3,
            seed=100,
        )
        rb_exp.set_analysis_options(gate_error_ratio=self.gate_error_ratio)
        rb_data = rb_exp.run(
            backend,
            noise_model=noise_model,
            seed_simulator=123,
            basis_gates=self.transpiled_base_gate,
        )
        rb_data.block_for_results()

        # alpha entry
        alpha = rb_data.analysis_results("alpha")
        self._check_fit_val(alpha, 0.9689125020789088, 0.0008008435286539923)

        # alpha_c entry
        alpha_c = rb_data.analysis_results("alpha_c")
        self._check_fit_val(alpha_c, 0.9780926279539126, 0.001798318260044234)

        # epc entry
        epc = rb_data.analysis_results("EPC")
        ref_epc_systematic_err = 0.03020071784707129
        ref_epc_systematic_bounds = [0.0, 0.046631246881636834]

        self._check_fit_val(epc, 0.016430529034565544, 0.0013487386950331755)

        self.assertAlmostEqual(
            epc.extra["EPC_systematic_err"],
            ref_epc_systematic_err,
            delta=ATOL_DEFAULT,
        )
        self.assertAlmostEqual(
            epc.extra["EPC_systematic_bounds"][0],
            ref_epc_systematic_bounds[0],
            delta=ATOL_DEFAULT,
        )
        self.assertAlmostEqual(
            epc.extra["EPC_systematic_bounds"][1],
            ref_epc_systematic_bounds[1],
            delta=ATOL_DEFAULT,
        )
