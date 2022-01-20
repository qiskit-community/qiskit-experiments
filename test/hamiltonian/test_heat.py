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

from qiskit import circuit, quantum_info as qi
from qiskit_experiments.library.hamiltonian import HeatElement, BatchHeatHelper, HeatAnalysis
from qiskit_experiments.library import ZXHeat


class TestHeatBase(QiskitExperimentsTestCase):
    """Test for base classes."""

    @staticmethod
    def _create_fake_amplifier(prep_seed, echo_seed, meas_seed, pname):
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
            parameter_name=pname,
        )

        return exp

    def test_element_experiment_config(self):
        """Test converting to and from config works"""
        exp = self._create_fake_amplifier(123, 456, 789, "test")

        loaded_exp = HeatElement.from_config(exp.config())
        self.assertNotEqual(exp, loaded_exp)
        self.assertTrue(self.experiments_equiv(exp, loaded_exp))

    def test_element_roundtrip_serializable(self):
        """Test round trip JSON serialization"""
        exp = self._create_fake_amplifier(123, 456, 789, "test")

        self.assertRoundTripSerializable(exp, self.experiments_equiv)

    def test_experiment_config(self):
        """Test converting to and from config works"""
        ampl1 = self._create_fake_amplifier(123, 456, 789, "i1")
        ampl2 = self._create_fake_amplifier(987, 654, 321, "i2")
        analysis = HeatAnalysis(fit_params=["i1", "i2"], out_params=["o1", "o2"])
        exp = BatchHeatHelper(heat_experiments=[ampl1, ampl2], heat_analysis=analysis)

        loaded_exp = BatchHeatHelper.from_config(exp.config())
        self.assertNotEqual(exp, loaded_exp)
        self.assertTrue(self.experiments_equiv(exp, loaded_exp))

    def test_roundtrip_serializable(self):
        """Test round trip JSON serialization"""
        ampl1 = self._create_fake_amplifier(123, 456, 789, "i1")
        ampl2 = self._create_fake_amplifier(987, 654, 321, "i2")
        analysis = HeatAnalysis(fit_params=["i1", "i2"], out_params=["o1", "o2"])
        exp = BatchHeatHelper(heat_experiments=[ampl1, ampl2], heat_analysis=analysis)

        self.assertRoundTripSerializable(exp, self.experiments_equiv)

    def test_analysis_config(self):
        """Test converting analysis to and from config works"""
        analysis = HeatAnalysis(fit_params=["i1", "i2"], out_params=["o1", "o2"])
        loaded = HeatAnalysis.from_config(analysis.config())
        self.assertNotEqual(analysis, loaded)
        self.assertEqual(analysis.config(), loaded.config())






