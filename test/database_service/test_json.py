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

"""Tests for base experiment framework."""

from test.base import QiskitExperimentsTestCase
from test.fake_experiment import FakeExperiment

import ddt
import numpy as np
from qiskit.circuit import Instruction
from qiskit.circuit.library import SXGate, RZXGate, Barrier, Measure, quantum_volume
import qiskit.quantum_info as qi
from qiskit_experiments.curve_analysis import CurveFitResult


class CustomClass:
    """Custom class for serialization tests"""

    def __init__(self, value=None):
        self._value = value

    def __eq__(self, other):
        return other.__class__ == self.__class__ and other._value == self._value

    @property
    def settings(self):
        """Return settings for class"""
        return {"value": self._value}

    @staticmethod
    def static_method(arg):
        """A static method"""
        return arg

    @classmethod
    def class_method(cls, arg):
        """A static method"""
        return arg


def custom_function(*args, **kwargs):
    """Test function for serialization"""
    return args, kwargs


@ddt.ddt
class TestJSON(QiskitExperimentsTestCase):
    """Test JSON encoder and decoder"""

    def test_roundtrip_experiment(self):
        """Test serializing an experiment"""

        obj = FakeExperiment([0])
        obj.set_transpile_options(optimization_level=3, basis_gates=["rx", "ry", "cz"])
        obj.set_run_options(shots=2000)
        self.assertRoundTripSerializable(obj)

    @ddt.data(SXGate(), RZXGate(0.4), Barrier(5), Measure())
    def test_roundtrip_gate(self, instruction):
        """Test round-trip serialization of a gate."""
        self.assertRoundTripSerializable(instruction)

    def test_custom_instruction(self):
        """Test the serialisation of a custom instruction."""

        class CustomInstruction(Instruction):
            """A custom instruction for testing."""

            def __init__(self, param: float):
                """Initialize the instruction."""
                super().__init__("test_inst", 2, 2, [param, 0.6])

        def compare_instructions(inst1, inst2):
            """Soft comparison of two instructions."""
            return inst1.soft_compare(inst2)

        self.assertRoundTripSerializable(CustomInstruction(0.123), check_func=compare_instructions)

    def test_roundtrip_quantum_circuit(self):
        """Test round-trip serialization of a circuits"""
        obj = quantum_volume(4)
        self.assertRoundTripSerializable(obj)

    def test_roundtrip_operator(self):
        """Test round-trip serialization of an Operator"""
        obj = qi.random_unitary(4, seed=10)
        self.assertRoundTripSerializable(obj)

    def test_roundtrip_statevector(self):
        """Test round-trip serialization of a Statevector"""
        obj = qi.random_statevector(4, seed=10)
        self.assertRoundTripSerializable(obj)

    def test_roundtrip_density_matrix(self):
        """Test round-trip serialization of a DensityMatrix"""
        obj = qi.random_density_matrix(4, seed=10)
        self.assertRoundTripSerializable(obj)

    @ddt.data("Choi", "SuperOp", "Kraus", "Stinespring", "PTM", "Chi")
    def test_roundtrip_quantum_channel(self, rep):
        """Test round-trip serialization of a DensityMatrix"""
        chan_cls = {
            "Choi": qi.Choi,
            "SuperOp": qi.SuperOp,
            "Kraus": qi.Kraus,
            "Stinespring": qi.Stinespring,
            "PTM": qi.PTM,
            "Chi": qi.Chi,
        }
        obj = chan_cls[rep](qi.random_quantum_channel(2, seed=10))
        self.assertRoundTripSerializable(obj)

    def test_roundtrip_function(self):
        """Test roundtrip serialization of custom class object"""
        obj = custom_function
        self.assertRoundTripSerializable(obj)

    def test_roundtrip_curvefitresult(self):
        """Test roundtrip serialization of the ScatterTable class"""
        obj = CurveFitResult(
            method="some_method",
            model_repr={"s1": "par0 * x + par1"},
            success=True,
            params={"par0": 0.3, "par1": 0.4},
            var_names=["par0", "par1"],
            covar=np.array([[2.19188077e-03, 2.19906808e-01], [2.19906808e-01, 2.62351788e01]]),
            reduced_chisq=1.5,
        )
        self.assertRoundTripSerializable(obj)

    def test_roundtrip_class_type(self):
        """Test roundtrip serialization of custom class"""
        obj = CustomClass
        self.assertRoundTripSerializable(obj)

    def test_roundtrip_class_object(self):
        """Test roundtrip serialization of custom class object"""
        obj = CustomClass(123)
        self.assertRoundTripSerializable(obj)

    def test_roundtrip_class_method(self):
        """Test roundtrip serialization of custom class object"""
        obj = CustomClass.class_method
        self.assertRoundTripSerializable(obj)

    def test_roundtrip_custom_static_method(self):
        """Test roundtrip serialization of custom class object"""
        obj = CustomClass.static_method
        self.assertRoundTripSerializable(obj)

    def test_roundtrip_main_function(self):
        """Test roundtrip serialization of __main__ custom class object"""
        import __main__ as main_mod

        main_mod.custom_function = custom_function
        main_mod.custom_function.__module__ = "__main__"
        obj = main_mod.custom_function
        self.assertRoundTripSerializable(obj)

    def test_roundtrip_main_class_type(self):
        """Test roundtrip serialization of __main__ custom class"""
        import __main__ as main_mod

        main_mod.CustomClass = CustomClass
        main_mod.CustomClass.__module__ = "__main__"
        obj = main_mod.CustomClass
        self.assertRoundTripSerializable(obj)

    def test_roundtrip_main_class_object(self):
        """Test roundtrip serialization of __main__ custom class object"""
        import __main__ as main_mod

        main_mod.CustomClass = CustomClass
        main_mod.CustomClass.__module__ = "__main__"
        obj = main_mod.CustomClass(123)
        self.assertRoundTripSerializable(obj)

    def test_roundtrip_main_class_method(self):
        """Test roundtrip serialization of __main__ custom class object"""
        import __main__ as main_mod

        main_mod.CustomClass = CustomClass
        main_mod.CustomClass.__module__ = "__main__"
        obj = main_mod.CustomClass.class_method
        self.assertRoundTripSerializable(obj)

    def test_roundtrip_main_custom_static_method(self):
        """Test roundtrip serialization of __main__ custom class object"""
        import __main__ as main_mod

        main_mod.CustomClass = CustomClass
        main_mod.CustomClass.__module__ = "__main__"
        obj = main_mod.CustomClass.static_method
        self.assertRoundTripSerializable(obj)
