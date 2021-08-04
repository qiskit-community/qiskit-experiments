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


def create_depolarizing_noise_model():
    """
    create noise model of depolarizing error
    Returns:
        NoiseModel: depolarizing error noise model
    """
    # the error parameters were taken from ibmq_manila on 17 june 2021
    p1q = 0.002257 * 10
    p2q = 0.006827 * 10
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
        noise_model = create_depolarizing_noise_model()
        backend = QasmSimulator()
        rb_exp = StandardRB(
            qubits=[0],
            lengths=list(range(1, 100, 10)),
            num_samples=3,
            seed=100,
        )
        rb_exp.set_analysis_options(gate_error_ratio=self.gate_error_ratio)
        rb_data = rb_exp.run(
            backend, noise_model=noise_model, basis_gates=self.transpiled_base_gate
        )
        rb_data.block_for_results()

        # alpha entry
        alpha = rb_data.analysis_results("alpha")
        self._check_fit_val(alpha, 0.9997606942463159, 0.00041321430695411904)

        # epc entry
        epc = rb_data.analysis_results("EPC")
        self._check_fit_val(epc, 0.00011965287684206904, 0.00020660715347705952)

        # epg entry
        epg_x = rb_data.analysis_results("EPG_x")
        self._check_fit_val(epg_x, 0.0005083368294731997)

    def test_2qubit_standard_rb(self):
        """Executing standard RB experiment and analyze with 2 qubit."""
        noise_model = create_depolarizing_noise_model()
        backend = QasmSimulator()
        rb_exp = StandardRB(
            qubits=[0, 1],
            lengths=list(range(1, 20, 2)),
            num_samples=3,
            seed=100,
        )
        rb_exp.set_analysis_options(gate_error_ratio=self.gate_error_ratio)
        rb_data = rb_exp.run(
            backend, noise_model=noise_model, basis_gates=self.transpiled_base_gate
        )
        rb_data.block_for_results()

        # alpha entry
        alpha = rb_data.analysis_results("alpha")
        self._check_fit_val(alpha, 0.988240664113885, 0.0005254263962064677)

        # epc entry
        epc = rb_data.analysis_results("EPC")
        self._check_fit_val(epc, 0.008819501914586247, 0.00039406979715485074)

        # epg entry
        epg_x = rb_data.analysis_results("EPG_cx")
        self._check_fit_val(epg_x, 0.005560563562776086)


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
        noise_model = create_depolarizing_noise_model()
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
            backend, noise_model=noise_model, basis_gates=self.transpiled_base_gate
        )
        rb_data.block_for_results()

        # alpha entry
        alpha = rb_data.analysis_results("alpha")
        self._check_fit_val(alpha, 0.9994768276408555, 1.4847358402650982e-05)

        # alpha_c entry
        alpha_c = rb_data.analysis_results("alpha_c")
        self._check_fit_val(alpha_c, 0.9979725591597532, 6.882915444481367e-05)

        # epc entry
        epc = rb_data.analysis_results("EPC")
        ref_epc_systematic_err = 0.0010137204201233763
        ref_epc_systematic_bounds = [0.0, 0.0020274408402467525]

        self._check_fit_val(epc, 0.0010137204201233763, 3.4414577222406835e-05)

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
        noise_model = create_depolarizing_noise_model()
        backend = QasmSimulator()
        rb_exp = InterleavedRB(
            interleaved_element=self.interleaved_gates["cx"],
            qubits=[0, 1],
            lengths=list(range(1, 20, 2)),
            num_samples=3,
            seed=100,
        )
        rb_exp.set_analysis_options(gate_error_ratio=self.gate_error_ratio)
        rb_data = rb_exp.run(
            backend, noise_model=noise_model, basis_gates=self.transpiled_base_gate
        )
        rb_data.block_for_results()

        # alpha entry
        alpha = rb_data.analysis_results("alpha")
        self._check_fit_val(alpha, 0.9884063217136179, 0.00020370367690368603)

        # alpha_c entry
        alpha_c = rb_data.analysis_results("alpha_c")
        self._check_fit_val(alpha_c, 0.9927651616883224, 0.00044396411042263065)

        # epc entry
        epc = rb_data.analysis_results("EPC")
        ref_epc_systematic_err = 0.01196438869581501
        ref_epc_systematic_bounds = [0.0, 0.017390517429573205]

        self._check_fit_val(epc, 0.005426128733758195, 0.000332973082816973)

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
