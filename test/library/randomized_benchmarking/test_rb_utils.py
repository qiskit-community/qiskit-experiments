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
A Tester for the RB utils module
"""
from test.base import QiskitExperimentsTestCase
import numpy as np
from uncertainties import ufloat
from ddt import ddt, data, unpack

from qiskit import QuantumCircuit
from qiskit.circuit.library import (
    IGate,
    XGate,
    YGate,
    ZGate,
    HGate,
    SGate,
    SdgGate,
    CXGate,
    CZGate,
    SwapGate,
)
import qiskit_experiments.library.randomized_benchmarking as rb
from qiskit_experiments.framework import AnalysisResultData


@ddt
class TestRBUtilities(QiskitExperimentsTestCase):
    """
    A test class for additional functionality provided by the StandardRB
    class.
    """

    instructions = {
        "i": IGate(),
        "x": XGate(),
        "y": YGate(),
        "z": ZGate(),
        "h": HGate(),
        "s": SGate(),
        "sdg": SdgGate(),
        "cx": CXGate(),
        "cz": CZGate(),
        "swap": SwapGate(),
    }
    seed = 42

    @data(
        [1, {((0,), "x"): 3, ((0,), "y"): 2, ((0,), "h"): 1}],
        [5, {((1,), "x"): 3, ((4,), "y"): 2, ((1,), "h"): 1, ((1, 4), "cx"): 7}],
    )
    @unpack
    def test_count_ops(self, num_qubits, expected_counts):
        """Testing the count_ops utility function
        this function receives a circuit and counts the number of gates
        in it, counting gates for different qubits separately"""
        circuit = QuantumCircuit(num_qubits)
        gates_to_add = []
        for gate, count in expected_counts.items():
            gates_to_add += [gate for _ in range(count)]
        rng = np.random.default_rng(self.seed)
        rng.shuffle(gates_to_add)
        for qubits, gate in gates_to_add:
            circuit.append(self.instructions[gate], qubits)
        with self.assertWarns(DeprecationWarning):
            counts = rb.RBUtils.count_ops(circuit)
        self.assertDictEqual(expected_counts, counts)

    def test_calculate_1q_epg(self):
        """Testing the calculation of 1 qubit error per gate
        The EPG is computed based on the error per clifford determined
        in the RB experiment, the gate counts, and an estimate about the
        relations between the errors of different gate types
        """
        epc_1_qubit = ufloat(0.0037, 0)
        qubits = [0]
        gate_error_ratio = {((0,), "id"): 1, ((0,), "rz"): 0, ((0,), "sx"): 1, ((0,), "x"): 1}
        gates_per_clifford = {((0,), "rz"): 10.5, ((0,), "sx"): 8.15, ((0,), "x"): 0.25}
        with self.assertWarns(DeprecationWarning):
            epg = rb.RBUtils.calculate_1q_epg(
                epc_1_qubit, qubits, gate_error_ratio, gates_per_clifford
            )
        error_dict = {
            ((0,), "rz"): ufloat(0, 0),
            ((0,), "sx"): ufloat(0.0004432101747785104, 0),
            ((0,), "x"): ufloat(0.0004432101747785104, 0),
        }

        for gate in ["x", "sx", "rz"]:
            expected_epg = error_dict[((0,), gate)]
            actual_epg = epg[(0,)][gate]
            self.assertAlmostEqual(
                expected_epg.nominal_value, actual_epg.nominal_value, delta=0.001
            )
            self.assertAlmostEqual(expected_epg.std_dev, actual_epg.std_dev, delta=0.001)

    def test_calculate_2q_epg(self):
        """Testing the calculation of 2 qubit error per gate
        The EPG is computed based on the error per clifford determined
        in the RB experiment, the gate counts, and an estimate about the
        relations between the errors of different gate types
        """
        epc_2_qubit = ufloat(0.034184849962675984, 0)
        qubits = [1, 4]
        gate_error_ratio = {
            ((1,), "id"): 1,
            ((4,), "id"): 1,
            ((1,), "rz"): 0,
            ((4,), "rz"): 0,
            ((1,), "sx"): 1,
            ((4,), "sx"): 1,
            ((1,), "x"): 1,
            ((4,), "x"): 1,
            ((4, 1), "cx"): 1,
            ((1, 4), "cx"): 1,
        }
        gates_per_clifford = {
            ((1, 4), "barrier"): 1.032967032967033,
            ((1,), "rz"): 15.932967032967033,
            ((1,), "sx"): 12.382417582417583,
            ((4,), "rz"): 18.681946624803768,
            ((4,), "sx"): 14.522605965463109,
            ((1, 4), "cx"): 1.0246506515936569,
            ((4, 1), "cx"): 0.5212064090480678,
            ((4,), "x"): 0.24237661112857592,
            ((1,), "measure"): 0.01098901098901099,
            ((4,), "measure"): 0.01098901098901099,
            ((1,), "x"): 0.2525918944392083,
        }
        epg_1_qubit = [
            AnalysisResultData("EPG_rz", 0.0, device_components=[1]),
            AnalysisResultData("EPG_rz", 0.0, device_components=[4]),
            AnalysisResultData("EPG_sx", 0.00036207066403884814, device_components=[1]),
            AnalysisResultData("EPG_sx", 0.0005429962529239195, device_components=[4]),
            AnalysisResultData("EPG_x", 0.00036207066403884814, device_components=[1]),
            AnalysisResultData("EPG_x", 0.0005429962529239195, device_components=[4]),
        ]
        with self.assertWarns(DeprecationWarning):
            epg = rb.RBUtils.calculate_2q_epg(
                epc_2_qubit, qubits, gate_error_ratio, gates_per_clifford, epg_1_qubit
            )

        error_dict = {
            ((1, 4), "cx"): ufloat(0.012438847900902494, 0),
        }

        expected_epg = error_dict[((1, 4), "cx")]
        actual_epg = epg[(1, 4)]["cx"]
        self.assertAlmostEqual(expected_epg.nominal_value, actual_epg.nominal_value, delta=0.001)
        self.assertAlmostEqual(expected_epg.std_dev, actual_epg.std_dev, delta=0.001)

    def test_coherence_limit(self):
        """Test coherence_limit."""
        t1 = 100.0
        t2 = 100.0
        gate_2_qubits = 0.5
        gate_1_qubit = 0.1
        twoq_coherence_err = rb.RBUtils.coherence_limit(2, [t1, t1], [t2, t2], gate_2_qubits)

        oneq_coherence_err = rb.RBUtils.coherence_limit(1, [t1], [t2], gate_1_qubit)

        self.assertAlmostEqual(oneq_coherence_err, 0.00049975, 6, "Error: 1Q Coherence Limit")

        self.assertAlmostEqual(twoq_coherence_err, 0.00597, 5, "Error: 2Q Coherence Limit")
