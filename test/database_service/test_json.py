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
from __future__ import annotations

import json
from test.base import QiskitExperimentsTestCase
from test.fake_experiment import FakeExperiment
from typing import TYPE_CHECKING

import ddt
import numpy as np
import uncertainties
from qiskit.circuit import Instruction
from qiskit.circuit.library import SXGate, RZXGate, Barrier, Measure, quantum_volume
import qiskit.quantum_info as qi

import qiskit_experiments.framework.json as qe_json
from qiskit_experiments.curve_analysis import CurveFitResult

if TYPE_CHECKING:
    from typing import Self


class CustomClass:
    """Custom class for serialization tests"""

    def __init__(self, value=None):
        self._value = value

    def __eq__(self, other):
        return other.__class__ == self.__class__ and other._value == self._value

    def __json_encode__(self) -> dict:
        return self.settings

    @classmethod
    def __json_decode__(cls, value) -> Self:
        return cls(**value)

    @property
    def settings(self):
        """Return settings for class"""
        return {"value": self._value}


@ddt.ddt
class TestJSON(QiskitExperimentsTestCase):
    """Test JSON encoder and decoder"""

    _orig_json_safe_modules = frozenset()

    @classmethod
    def setUpClass(cls):
        """Class-level test setup"""
        super().setUpClass()
        qe_json._load_allowed_packages()
        cls._orig_json_safe_modules = qe_json._allowed_packages
        qe_json._allowed_packages = cls._orig_json_safe_modules.union(["test"])

    @classmethod
    def tearDownClass(cls):
        """Class-level tear down"""
        super().tearDownClass()
        qe_json._allowed_packages = cls._orig_json_safe_modules

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

    def test_legacy_quantum_info(self):
        """Test deserialization of old quantum_info data"""
        data = {
            "__type__": "object",
            "__value__": {
                "class": {
                    "__type__": "type",
                    "__value__": {
                        "name": "Statevector",
                        "module": "qiskit.quantum_info.states.statevector",
                        "version": "2.4.0",
                    },
                },
                "settings": {"data": [0, 1], "dims": [2]},
                "version": "2.4.0",
            },
        }
        # Dump data to json with default serializer and then deserialize with custom decoder
        deserialized = json.loads(json.dumps(data), cls=qe_json.ExperimentDecoder)
        self.assertEqual(deserialized, qi.Statevector([0, 1]))

    def test_legacy_ufloat(self):
        """Test deserialization of old uncertainties data"""
        data = {
            "__type__": "object",
            "__value__": {
                "class": {
                    "__type__": "type",
                    "__value__": {
                        "name": "Variable",
                        "module": "uncertainties.core",
                        "version": "3.2.3",
                    },
                },
                "settings": {"value": 2.0, "std_dev": 0.1, "tag": None},
                "version": "3.2.3",
            },
        }
        # Dump data to json with default serializer and then deserialize with custom decoder
        deserialized = json.loads(json.dumps(data), cls=qe_json.ExperimentDecoder)
        self.assertIsInstance(deserialized, uncertainties.UFloat)
        self.assertEqual(deserialized.nominal_value, 2)
        self.assertEqual(deserialized.std_dev, 0.1)

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
