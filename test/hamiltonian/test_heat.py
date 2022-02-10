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

"""
Test for the HEAT experiment
"""
from test.base import QiskitExperimentsTestCase

import scipy.linalg as la
import numpy as np

from ddt import ddt, data, unpack

from qiskit import circuit, quantum_info as qi
from qiskit.providers.aer import AerSimulator
from qiskit_experiments.library.hamiltonian import HeatElement, HeatAnalysis
from qiskit_experiments.library import ZX90HeatXError, ZX90HeatYError, ZX90HeatZError
from qiskit_experiments.framework import BatchExperiment


class HeatExperimentsTestCase:
    """Base class for HEAT experiment test."""

    backend = AerSimulator()

    @staticmethod
    def create_heat_gate(generator):
        """Helper function to create HEAT gate for Aer simulator."""
        unitary = la.expm(-1j * generator)

        gate_decomp = circuit.QuantumCircuit(2)
        gate_decomp.unitary(unitary, [0, 1])

        heat_gate = circuit.Gate(f"heat_{hash(unitary.tobytes())}", 2, [])
        heat_gate.add_decomposition(gate_decomp)

        return heat_gate


class TestHeatBase(QiskitExperimentsTestCase):
    """Test for base classes."""

    @staticmethod
    def _create_fake_amplifier(prep_seed, echo_seed, meas_seed):
        """Helper method to generate fake experiment."""
        prep = circuit.QuantumCircuit(2)
        prep.compose(qi.random_unitary(4, seed=prep_seed).to_instruction(), inplace=True)

        echo = circuit.QuantumCircuit(2)
        echo.compose(qi.random_unitary(4, seed=echo_seed).to_instruction(), inplace=True)

        meas = circuit.QuantumCircuit(2)
        meas.compose(qi.random_unitary(4, seed=meas_seed).to_instruction(), inplace=True)

        exp = HeatElement(
            qubits=(0, 1),
            prep_circ=prep,
            echo_circ=echo,
            meas_circ=meas,
        )

        return exp

    def test_element_experiment_config(self):
        """Test converting to and from config works"""
        exp = self._create_fake_amplifier(123, 456, 789)

        loaded_exp = HeatElement.from_config(exp.config())
        self.assertNotEqual(exp, loaded_exp)
        self.assertTrue(self.json_equiv(exp, loaded_exp))

    def test_element_roundtrip_serializable(self):
        """Test round trip JSON serialization"""
        exp = self._create_fake_amplifier(123, 456, 789)

        self.assertRoundTripSerializable(exp, self.json_equiv)

    def test_analysis_config(self):
        """Test converting analysis to and from config works"""
        analysis = HeatAnalysis(fit_params=("i1", "i2"), out_params=("o1", "o2"))
        loaded = HeatAnalysis.from_config(analysis.config())
        self.assertNotEqual(analysis, loaded)
        self.assertEqual(analysis.config(), loaded.config())

    def test_create_circuit(self):
        """Test HEAT circuit generation."""
        prep = circuit.QuantumCircuit(2)
        prep.x(0)
        prep.ry(np.pi / 2, 1)

        echo = circuit.QuantumCircuit(2)
        echo.z(1)

        meas = circuit.QuantumCircuit(2)
        meas.rx(np.pi / 2, 1)

        exp = HeatElement(
            qubits=(0, 1),
            prep_circ=prep,
            echo_circ=echo,
            meas_circ=meas,
        )
        exp.set_experiment_options(repetitions=[2])

        heat_circ = exp.circuits()[0]

        ref_circ = circuit.QuantumCircuit(2, 1)
        ref_circ.x(0)
        ref_circ.ry(np.pi / 2, 1)
        ref_circ.barrier()
        ref_circ.append(exp.experiment_options.heat_gate, [0, 1])
        ref_circ.z(1)
        ref_circ.barrier()
        ref_circ.append(exp.experiment_options.heat_gate, [0, 1])
        ref_circ.z(1)
        ref_circ.barrier()
        ref_circ.rx(np.pi / 2, 1)
        ref_circ.measure(1, 0)

        self.assertEqual(heat_circ, ref_circ)


@ddt
class TestZXHeat(QiskitExperimentsTestCase, HeatExperimentsTestCase):
    """Test ZX Heat experiment."""

    @staticmethod
    def create_generator(
        angle=np.pi / 2,
        e_zx=0.0,
        e_zy=0.0,
        e_zz=0.0,
        e_ix=0.0,
        e_iy=0.0,
        e_iz=0.0,
    ):
        """Create generator Hamiltonian represented by numpy array."""
        generator_ham = (
            0.5
            * (
                (angle + e_zx) * qi.Operator.from_label("XZ")
                + e_zy * qi.Operator.from_label("YZ")
                + e_zz * qi.Operator.from_label("ZZ")
                + e_ix * qi.Operator.from_label("XI")
                + e_iy * qi.Operator.from_label("YI")
                + e_iz * qi.Operator.from_label("ZI")
            ).data
        )

        return generator_ham

    def test_transpile_options_sync(self):
        """Test if transpile option set to composite can update all component experiments."""
        exp = ZX90HeatXError(qubits=(0, 1), backend=self.backend)
        basis_exp0 = exp.component_experiment(0).transpile_options.basis_gates
        basis_exp1 = exp.component_experiment(1).transpile_options.basis_gates
        self.assertListEqual(basis_exp0, ["sx", "x", "rz", "heat"])
        self.assertListEqual(basis_exp1, ["sx", "x", "rz", "heat"])

        # override from composite
        exp.set_transpile_options(basis_gates=["sx", "x", "rz", "my_heat"])
        new_basis_exp0 = exp.component_experiment(0).transpile_options.basis_gates
        new_basis_exp1 = exp.component_experiment(1).transpile_options.basis_gates
        self.assertListEqual(new_basis_exp0, ["sx", "x", "rz", "my_heat"])
        self.assertListEqual(new_basis_exp1, ["sx", "x", "rz", "my_heat"])

    def test_experiment_options_sync(self):
        """Test if experiment option set to composite can update all component experiments."""
        exp = ZX90HeatXError(qubits=(0, 1), backend=self.backend)
        reps_exp0 = exp.component_experiment(0).experiment_options.repetitions
        reps_exp1 = exp.component_experiment(1).experiment_options.repetitions
        self.assertListEqual(reps_exp0, list(range(21)))
        self.assertListEqual(reps_exp1, list(range(21)))

        # override from composite
        exp.set_experiment_options(repetitions=[1, 2, 3])
        new_reps_exp0 = exp.component_experiment(0).experiment_options.repetitions
        new_reps_exp1 = exp.component_experiment(1).experiment_options.repetitions
        self.assertListEqual(new_reps_exp0, [1, 2, 3])
        self.assertListEqual(new_reps_exp1, [1, 2, 3])

    @data(
        [0.08, -0.01],
        [-0.05, 0.13],
        [0.15, 0.02],
        [-0.04, -0.02],
        [0.0, 0.12],
        [0.12, 0.0],
    )
    @unpack
    def test_x_error_amplification(self, e_zx, e_ix):
        """Test for X error amplification."""
        exp = ZX90HeatXError(qubits=(0, 1), backend=self.backend)
        generator = self.create_generator(e_zx=e_zx, e_ix=e_ix)
        gate = self.create_heat_gate(generator)
        exp.set_experiment_options(heat_gate=gate)
        exp.set_transpile_options(basis_gates=["x", "sx", "rz", "unitary"])

        exp_data = exp.run()
        self.assertExperimentDone(exp_data)

        self.assertAlmostEqual(
            exp_data.analysis_results("A_IX").value.nominal_value, e_ix, delta=0.01
        )
        self.assertAlmostEqual(
            exp_data.analysis_results("A_ZX").value.nominal_value, e_zx, delta=0.01
        )

    @data(
        [0.02, -0.01],
        [-0.05, 0.03],
        [0.03, 0.02],
        [-0.04, -0.01],
        [0.0, 0.01],
        [0.01, 0.0],
    )
    @unpack
    def test_y_error_amplification(self, e_zy, e_iy):
        """Test for Y error amplification."""
        exp = ZX90HeatYError(qubits=(0, 1), backend=self.backend)
        generator = self.create_generator(e_zy=e_zy, e_iy=e_iy)
        gate = self.create_heat_gate(generator)
        exp.set_experiment_options(heat_gate=gate)
        exp.set_transpile_options(basis_gates=["x", "sx", "rz", "unitary"])

        exp_data = exp.run()
        self.assertExperimentDone(exp_data)

        # The factor 0.7 is estimated from numerical analysis, which comes from ZX commutator term.
        # Note that this number may depend on magnitude of coefficients.
        self.assertAlmostEqual(
            exp_data.analysis_results("A_IY").value.nominal_value, 0.7 * e_iy, delta=0.01
        )
        self.assertAlmostEqual(
            exp_data.analysis_results("A_ZY").value.nominal_value, 0.7 * e_zy, delta=0.01
        )

    @data(
        [0.02, -0.01],
        [-0.05, 0.03],
        [0.03, 0.02],
        [-0.04, -0.01],
        [0.0, 0.01],
        [0.01, 0.0],
    )
    @unpack
    def test_z_error_amplification(self, e_zz, e_iz):
        """Test for Z error amplification."""
        exp = ZX90HeatZError(qubits=(0, 1), backend=self.backend)
        generator = self.create_generator(e_zz=e_zz, e_iz=e_iz)
        gate = self.create_heat_gate(generator)
        exp.set_experiment_options(heat_gate=gate)
        exp.set_transpile_options(basis_gates=["x", "sx", "rz", "unitary"])

        exp_data = exp.run()
        self.assertExperimentDone(exp_data)

        # The factor 0.7 is estimated from numerical analysis, which comes from ZX commutator term.
        # Note that this number may depend on magnitude of coefficients.
        self.assertAlmostEqual(
            exp_data.analysis_results("A_IZ").value.nominal_value, 0.7 * e_iz, delta=0.01
        )
        self.assertAlmostEqual(
            exp_data.analysis_results("A_ZZ").value.nominal_value, 0.7 * e_zz, delta=0.01
        )

    @data(123, 456)
    def test_pseudo_calibration(self, seed):
        """Test calibration with HEAT.

        This is somewhat of an integration test that covers multiple aspects of the experiment.

        The protocol of this test is as follows:

            First, this generates random Hamiltonian with multiple finite error terms.
            Then errors in every axis is measured by three HEAT experiments as
            a batch experiment, then inferred error values are subtracted from the
            actual errors randomly determined. Repeating this eventually converges into
            zero-ish errors in all axes if HEAT experiments work correctly.

        This checks if experiment sequence is designed correctly, and also checks
        if HEAT experiment can be batched.
        Note that HEAT itself is a batch experiment of amplifications.
        """
        np.random.seed(seed)
        coeffs = np.random.normal(0, 0.03, 6)
        terms = ["e_zx", "e_zy", "e_zz", "e_ix", "e_iy", "e_iz"]

        errors_dict = dict(zip(terms, coeffs))

        for _ in range(10):
            generator = self.create_generator(**errors_dict)
            gate = self.create_heat_gate(generator)

            exp_x = ZX90HeatXError(qubits=(0, 1))
            exp_x.set_experiment_options(heat_gate=gate)
            exp_x.set_transpile_options(basis_gates=["x", "sx", "rz", "unitary"])

            exp_y = ZX90HeatYError(qubits=(0, 1))
            exp_y.set_experiment_options(heat_gate=gate)
            exp_y.set_transpile_options(basis_gates=["x", "sx", "rz", "unitary"])

            exp_z = ZX90HeatZError(qubits=(0, 1))
            exp_z.set_experiment_options(heat_gate=gate)
            exp_z.set_transpile_options(basis_gates=["x", "sx", "rz", "unitary"])

            exp = BatchExperiment([exp_x, exp_y, exp_z], backend=self.backend)
            exp_data = exp.run()
            self.assertExperimentDone(exp_data)

            for n, tp in enumerate(["x", "y", "z"]):
                a_zp = exp_data.child_data(n).analysis_results(f"A_Z{tp.upper()}")
                a_ip = exp_data.child_data(n).analysis_results(f"A_I{tp.upper()}")
                errors_dict[f"e_z{tp}"] -= a_zp.value.nominal_value
                errors_dict[f"e_i{tp}"] -= a_ip.value.nominal_value

        for v in errors_dict.values():
            self.assertAlmostEqual(v, 0.0, delta=0.005)
