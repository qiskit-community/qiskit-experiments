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
A Tester for the Clifford utilities
"""
from test.base import QiskitExperimentsTestCase
import os
import numpy as np
from numpy.random import default_rng
from ddt import ddt

from qiskit import QuantumCircuit, QuantumRegister, QiskitError
from qiskit.compiler import transpile
from qiskit.quantum_info import Operator, Clifford
from qiskit import qpy
from qiskit.circuit.library import (
    IGate,
    XGate,
    YGate,
    ZGate,
    HGate,
    SGate,
    SdgGate,
    SXGate,
    RZGate,
)
from qiskit_experiments.library.randomized_benchmarking.clifford_utils import CliffordUtils
from qiskit_experiments.library.randomized_benchmarking.clifford_data import (
    CLIFF_SINGLE_GATE_MAP_1Q,
    CLIFF_SINGLE_GATE_MAP_2Q,
    CLIFF_COMPOSE_DATA_1Q,
    CLIFF_COMPOSE_DATA_2Q,
    CLIFF_INVERSE_DATA_1Q,
    CLIFF_INVERSE_DATA_2Q,
    CLIFF_NUM_TO_LAYERS_2Q,
    CLIFF_LAYERS_TO_NUM_2Q
)

@ddt
class TestCliffordUtils(QiskitExperimentsTestCase):
    """A test for the Clifford manipulations, including number to and from Clifford mapping"""

    basis_gates = ["rz", "sx", "cx"]
    seed = 123

    def test_clifford_1_qubit_generation(self):
        """Verify 1-qubit clifford indeed generates the correct group"""
        clifford_dicts = [
            {"stabilizer": ["+Z"], "destabilizer": ["+X"]},
            {"stabilizer": ["+X"], "destabilizer": ["+Z"]},
            {"stabilizer": ["+Y"], "destabilizer": ["+X"]},
            {"stabilizer": ["+X"], "destabilizer": ["+Y"]},
            {"stabilizer": ["+Z"], "destabilizer": ["+Y"]},
            {"stabilizer": ["+Y"], "destabilizer": ["+Z"]},
            {"stabilizer": ["-Z"], "destabilizer": ["+X"]},
            {"stabilizer": ["+X"], "destabilizer": ["-Z"]},
            {"stabilizer": ["-Y"], "destabilizer": ["+X"]},
            {"stabilizer": ["+X"], "destabilizer": ["-Y"]},
            {"stabilizer": ["-Z"], "destabilizer": ["-Y"]},
            {"stabilizer": ["-Y"], "destabilizer": ["-Z"]},
            {"stabilizer": ["-Z"], "destabilizer": ["-X"]},
            {"stabilizer": ["-X"], "destabilizer": ["-Z"]},
            {"stabilizer": ["+Y"], "destabilizer": ["-X"]},
            {"stabilizer": ["-X"], "destabilizer": ["+Y"]},
            {"stabilizer": ["-Z"], "destabilizer": ["+Y"]},
            {"stabilizer": ["+Y"], "destabilizer": ["-Z"]},
            {"stabilizer": ["+Z"], "destabilizer": ["-X"]},
            {"stabilizer": ["-X"], "destabilizer": ["+Z"]},
            {"stabilizer": ["-Y"], "destabilizer": ["-X"]},
            {"stabilizer": ["-X"], "destabilizer": ["-Y"]},
            {"stabilizer": ["+Z"], "destabilizer": ["-Y"]},
            {"stabilizer": ["-Y"], "destabilizer": ["+Z"]},
        ]
        cliffords = [Clifford.from_dict(i) for i in clifford_dicts]
        cliff_utils = CliffordUtils(num_qubits=1, basis_gates=self.basis_gates)
        for n in range(24):
            clifford = cliff_utils.clifford_1_qubit(n)
            self.assertEqual(clifford, cliffords[n])

    def test_clifford_2_qubit_generation(self):
        """Verify 2-qubit clifford indeed generates the correct group"""
        pauli_free_elements = [
            0,
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            8,
            9,
            10,
            11,
            12,
            13,
            14,
            15,
            16,
            17,
            18,
            19,
            20,
            21,
            22,
            23,
            24,
            25,
            26,
            27,
            28,
            29,
            30,
            31,
            32,
            33,
            34,
            35,
            576,
            577,
            578,
            579,
            580,
            581,
            582,
            583,
            584,
            585,
            586,
            587,
            588,
            589,
            590,
            591,
            592,
            593,
            594,
            595,
            596,
            597,
            598,
            599,
            600,
            601,
            602,
            603,
            604,
            605,
            606,
            607,
            608,
            609,
            610,
            611,
            612,
            613,
            614,
            615,
            616,
            617,
            618,
            619,
            620,
            621,
            622,
            623,
            624,
            625,
            626,
            627,
            628,
            629,
            630,
            631,
            632,
            633,
            634,
            635,
            636,
            637,
            638,
            639,
            640,
            641,
            642,
            643,
            644,
            645,
            646,
            647,
            648,
            649,
            650,
            651,
            652,
            653,
            654,
            655,
            656,
            657,
            658,
            659,
            660,
            661,
            662,
            663,
            664,
            665,
            666,
            667,
            668,
            669,
            670,
            671,
            672,
            673,
            674,
            675,
            676,
            677,
            678,
            679,
            680,
            681,
            682,
            683,
            684,
            685,
            686,
            687,
            688,
            689,
            690,
            691,
            692,
            693,
            694,
            695,
            696,
            697,
            698,
            699,
            700,
            701,
            702,
            703,
            704,
            705,
            706,
            707,
            708,
            709,
            710,
            711,
            712,
            713,
            714,
            715,
            716,
            717,
            718,
            719,
            720,
            721,
            722,
            723,
            724,
            725,
            726,
            727,
            728,
            729,
            730,
            731,
            732,
            733,
            734,
            735,
            736,
            737,
            738,
            739,
            740,
            741,
            742,
            743,
            744,
            745,
            746,
            747,
            748,
            749,
            750,
            751,
            752,
            753,
            754,
            755,
            756,
            757,
            758,
            759,
            760,
            761,
            762,
            763,
            764,
            765,
            766,
            767,
            768,
            769,
            770,
            771,
            772,
            773,
            774,
            775,
            776,
            777,
            778,
            779,
            780,
            781,
            782,
            783,
            784,
            785,
            786,
            787,
            788,
            789,
            790,
            791,
            792,
            793,
            794,
            795,
            796,
            797,
            798,
            799,
            800,
            801,
            802,
            803,
            804,
            805,
            806,
            807,
            808,
            809,
            810,
            811,
            812,
            813,
            814,
            815,
            816,
            817,
            818,
            819,
            820,
            821,
            822,
            823,
            824,
            825,
            826,
            827,
            828,
            829,
            830,
            831,
            832,
            833,
            834,
            835,
            836,
            837,
            838,
            839,
            840,
            841,
            842,
            843,
            844,
            845,
            846,
            847,
            848,
            849,
            850,
            851,
            852,
            853,
            854,
            855,
            856,
            857,
            858,
            859,
            860,
            861,
            862,
            863,
            864,
            865,
            866,
            867,
            868,
            869,
            870,
            871,
            872,
            873,
            874,
            875,
            876,
            877,
            878,
            879,
            880,
            881,
            882,
            883,
            884,
            885,
            886,
            887,
            888,
            889,
            890,
            891,
            892,
            893,
            894,
            895,
            896,
            897,
            898,
            899,
            5760,
            5761,
            5762,
            5763,
            5764,
            5765,
            5766,
            5767,
            5768,
            5769,
            5770,
            5771,
            5772,
            5773,
            5774,
            5775,
            5776,
            5777,
            5778,
            5779,
            5780,
            5781,
            5782,
            5783,
            5784,
            5785,
            5786,
            5787,
            5788,
            5789,
            5790,
            5791,
            5792,
            5793,
            5794,
            5795,
            5796,
            5797,
            5798,
            5799,
            5800,
            5801,
            5802,
            5803,
            5804,
            5805,
            5806,
            5807,
            5808,
            5809,
            5810,
            5811,
            5812,
            5813,
            5814,
            5815,
            5816,
            5817,
            5818,
            5819,
            5820,
            5821,
            5822,
            5823,
            5824,
            5825,
            5826,
            5827,
            5828,
            5829,
            5830,
            5831,
            5832,
            5833,
            5834,
            5835,
            5836,
            5837,
            5838,
            5839,
            5840,
            5841,
            5842,
            5843,
            5844,
            5845,
            5846,
            5847,
            5848,
            5849,
            5850,
            5851,
            5852,
            5853,
            5854,
            5855,
            5856,
            5857,
            5858,
            5859,
            5860,
            5861,
            5862,
            5863,
            5864,
            5865,
            5866,
            5867,
            5868,
            5869,
            5870,
            5871,
            5872,
            5873,
            5874,
            5875,
            5876,
            5877,
            5878,
            5879,
            5880,
            5881,
            5882,
            5883,
            5884,
            5885,
            5886,
            5887,
            5888,
            5889,
            5890,
            5891,
            5892,
            5893,
            5894,
            5895,
            5896,
            5897,
            5898,
            5899,
            5900,
            5901,
            5902,
            5903,
            5904,
            5905,
            5906,
            5907,
            5908,
            5909,
            5910,
            5911,
            5912,
            5913,
            5914,
            5915,
            5916,
            5917,
            5918,
            5919,
            5920,
            5921,
            5922,
            5923,
            5924,
            5925,
            5926,
            5927,
            5928,
            5929,
            5930,
            5931,
            5932,
            5933,
            5934,
            5935,
            5936,
            5937,
            5938,
            5939,
            5940,
            5941,
            5942,
            5943,
            5944,
            5945,
            5946,
            5947,
            5948,
            5949,
            5950,
            5951,
            5952,
            5953,
            5954,
            5955,
            5956,
            5957,
            5958,
            5959,
            5960,
            5961,
            5962,
            5963,
            5964,
            5965,
            5966,
            5967,
            5968,
            5969,
            5970,
            5971,
            5972,
            5973,
            5974,
            5975,
            5976,
            5977,
            5978,
            5979,
            5980,
            5981,
            5982,
            5983,
            5984,
            5985,
            5986,
            5987,
            5988,
            5989,
            5990,
            5991,
            5992,
            5993,
            5994,
            5995,
            5996,
            5997,
            5998,
            5999,
            6000,
            6001,
            6002,
            6003,
            6004,
            6005,
            6006,
            6007,
            6008,
            6009,
            6010,
            6011,
            6012,
            6013,
            6014,
            6015,
            6016,
            6017,
            6018,
            6019,
            6020,
            6021,
            6022,
            6023,
            6024,
            6025,
            6026,
            6027,
            6028,
            6029,
            6030,
            6031,
            6032,
            6033,
            6034,
            6035,
            6036,
            6037,
            6038,
            6039,
            6040,
            6041,
            6042,
            6043,
            6044,
            6045,
            6046,
            6047,
            6048,
            6049,
            6050,
            6051,
            6052,
            6053,
            6054,
            6055,
            6056,
            6057,
            6058,
            6059,
            6060,
            6061,
            6062,
            6063,
            6064,
            6065,
            6066,
            6067,
            6068,
            6069,
            6070,
            6071,
            6072,
            6073,
            6074,
            6075,
            6076,
            6077,
            6078,
            6079,
            6080,
            6081,
            6082,
            6083,
            10944,
            10945,
            10946,
            10947,
            10948,
            10949,
            10950,
            10951,
            10952,
            10953,
            10954,
            10955,
            10956,
            10957,
            10958,
            10959,
            10960,
            10961,
            10962,
            10963,
            10964,
            10965,
            10966,
            10967,
            10968,
            10969,
            10970,
            10971,
            10972,
            10973,
            10974,
            10975,
            10976,
            10977,
            10978,
            10979,
        ]
        cliffords = []
        cliff_utils = CliffordUtils(num_qubits=2, basis_gates=self.basis_gates)
        for n in pauli_free_elements:
            clifford = cliff_utils.clifford_2_qubit(n)
            phase = clifford.table.phase
            for i in range(4):
                self.assertFalse(phase[i])
            for other_clifford in cliffords:
                self.assertNotEqual(clifford, other_clifford)
            cliffords.append(clifford)

        pauli_check_elements_list = [
            [0, 36, 72, 108, 144, 180, 216, 252, 288, 324, 360, 396, 432, 468, 504, 540],
            [
                576,
                900,
                1224,
                1548,
                1872,
                2196,
                2520,
                2844,
                3168,
                3492,
                3816,
                4140,
                4464,
                4788,
                5112,
                5436,
            ],
            [
                5760,
                6084,
                6408,
                6732,
                7056,
                7380,
                7704,
                8028,
                8352,
                8676,
                9000,
                9324,
                9648,
                9972,
                10296,
                10620,
            ],
            [
                10944,
                10980,
                11016,
                11052,
                11088,
                11124,
                11160,
                11196,
                11232,
                11268,
                11304,
                11340,
                11376,
                11412,
                11448,
                11484,
            ],
        ]
        cliff_utils = CliffordUtils(num_qubits=2, basis_gates=self.basis_gates)
        for pauli_check_elements in pauli_check_elements_list:
            phases = []
            table = None
            for n in pauli_check_elements:
                clifford = cliff_utils.clifford_2_qubit(n)
                if table is None:
                    table = clifford.table.array
                else:
                    self.assertTrue(np.all(table == clifford.table.array))
                phase = tuple(clifford.table.phase)
                for other_phase in phases:
                    self.assertNotEqual(phase, other_phase)
                phases.append(phase)

    def test_number_to_clifford_mapping_single_gate(self):
        """Test that the methods num_from_1q_clifford_single_gate and
        clifford_1_qubit_circuit perform the reverse operations from
        each other"""
        transpiled_cliff_list = [
            SXGate(),
            RZGate(np.pi),
            RZGate(-np.pi),
            RZGate(np.pi / 2),
            RZGate(-np.pi / 2),
        ]
        transpiled_cliff_names = [gate.name for gate in transpiled_cliff_list]
        qubits = [0]
        cliff_utils = CliffordUtils(num_qubits=1, basis_gates=transpiled_cliff_names)
        for inst in transpiled_cliff_list:
            num = cliff_utils.num_from_clifford_single_gate(
                inst, qubits, rb_num_qubits=1
            )
            qc_from_num = cliff_utils.clifford_1_qubit_circuit(num=num)
            qr = QuantumRegister(1)
            qc_from_inst = QuantumCircuit(qr)
            qc_from_inst._append(inst, [qr[0]], [])
            assert Operator(qc_from_num).equiv(Operator(qc_from_inst))

        general_cliff_list = [
            IGate(),
            HGate(),
            SdgGate(),
            SGate(),
            XGate(),
            SXGate(),
            YGate(),
            ZGate(),
        ]
        general_cliff_names = [gate.name for gate in general_cliff_list]
        cliff_utils = CliffordUtils(num_qubits=1, basis_gates=general_cliff_names)
        for inst in general_cliff_list:
            num = cliff_utils.num_from_clifford_single_gate(
                inst, qubits, rb_num_qubits=1
            )
            qc_from_num = cliff_utils.clifford_1_qubit_circuit(num=num)
            qr = QuantumRegister(1)
            qc_from_inst = QuantumCircuit(qr)
            qc_from_inst._append(inst, [qr[0]], [])
            assert Operator(qc_from_num).equiv(Operator(qc_from_inst))

    def test_number_to_clifford_mapping_1q(self):
        """Test that the number generated by compose_num_with_clifford on qc
        corresponds to the index of the circuit qc.
        """

        cliff_utils = CliffordUtils(num_qubits=1, basis_gates=self.basis_gates)

        for index, qc in enumerate(cliff_utils._transpiled_cliffords_1q):
            num = cliff_utils.compose_num_with_clifford(
                composed_num=0, qc=qc,
            )
            assert num == index

    def test_number_to_clifford_mapping_2q(self):
        """Test that the number generated by compose_num_with_clifford on qc
        corresponds to the index of the circuit qc.
        """
        cliff_utils = CliffordUtils(num_qubits=2, basis_gates=self.basis_gates)
        for index in range(cliff_utils.NUM_CLIFFORD_2_QUBIT):
            qc = cliff_utils.create_cliff_from_num(index)
            num = cliff_utils.compose_num_with_clifford(
                composed_num=0, qc=qc,
            )
            assert num == index

    def test_compose_by_num_1q(self):
        """Compare compose using num and Clifford to compose using two Cliffords, for
        a single qubit"""
        num_tests = 50
        rng = default_rng(seed=self.seed)
        cliff_utils = CliffordUtils(num_qubits=1, basis_gates=self.basis_gates)
        for _ in range(num_tests):
            num1 = rng.integers(0, cliff_utils.NUM_CLIFFORD_1_QUBIT - 1)
            cliff1_qc = cliff_utils.clifford_1_qubit_circuit(num1)
            num2 = rng.integers(0, cliff_utils.NUM_CLIFFORD_1_QUBIT - 1)
            cliff2_qc = cliff_utils.clifford_1_qubit_circuit(num2)
            transpiled_qc2 = transpile(
                cliff2_qc, optimization_level=1, basis_gates=self.basis_gates
            )
            result_by_num = cliff_utils.compose_num_with_clifford(
                num1, transpiled_qc2
            )
            clifford_from_num = cliff_utils.clifford_1_qubit_circuit(result_by_num)
            clifford_from_compose = cliff1_qc.compose(cliff2_qc)
            assert Operator(clifford_from_num).equiv(Operator(clifford_from_compose))

    def test_compose_by_num_2q(self):
        """Compare compose using num and Clifford to compose using two Cliffords, for
        two qubits"""
        num_tests = 100
        rng = default_rng(seed=self.seed)
        cliff_utils = CliffordUtils(num_qubits=2, basis_gates=self.basis_gates)
        for _ in range(num_tests):
            num1 = rng.integers(0, cliff_utils.NUM_CLIFFORD_2_QUBIT - 1)
            cliff1_qc = cliff_utils.clifford_2_qubit_circuit(num1)
            num2 = rng.integers(0, cliff_utils.NUM_CLIFFORD_2_QUBIT - 1)
            cliff2_qc = cliff_utils.clifford_2_qubit_circuit(num2)
            transpiled_qc2 = transpile(
                cliff2_qc, optimization_level=1, basis_gates=self.basis_gates
            )
            result_by_num = cliff_utils.compose_num_with_clifford(
                num1, transpiled_qc2
            )
            clifford_from_num = cliff_utils.clifford_2_qubit_circuit(result_by_num)
            clifford_from_compose = cliff1_qc.compose(cliff2_qc)
            assert Operator(clifford_from_num).equiv(Operator(clifford_from_compose))

    def test_inverse_by_num_1q(self):
        """Compare inverse using num to inverse using Clifford"""
        num_tests = 10
        rng = default_rng(seed=self.seed)
        cliff_utils = CliffordUtils(num_qubits=1, basis_gates=self.basis_gates)
        for _ in range(num_tests):
            num = rng.integers(0, cliff_utils.NUM_CLIFFORD_1_QUBIT - 1)
            cliff_qc = cliff_utils.clifford_1_qubit_circuit(num)
            inverse_num = cliff_utils.clifford_inverse_by_num(num)
            inverse_by_num = cliff_utils.clifford_1_qubit_circuit(inverse_num)
            inverse_cliff = Clifford(cliff_qc).adjoint()
            assert (Operator(inverse_by_num)).equiv(Operator(inverse_cliff))

    def test_inverse_by_num_2q(self):
        """Compare inverse using num to inverse using Clifford"""
        num_tests = 100
        rng = default_rng(seed=self.seed)
        cliff_utils = CliffordUtils(num_qubits=2, basis_gates=self.basis_gates)
        for _ in range(num_tests):
            num = rng.integers(0, cliff_utils.NUM_CLIFFORD_2_QUBIT - 1)
            cliff_qc = cliff_utils.clifford_2_qubit_circuit(num)
            inverse_num = cliff_utils.clifford_inverse_by_num(num)
            inverse_by_num = cliff_utils.clifford_2_qubit_circuit(inverse_num)
            inverse_cliff = Clifford(cliff_qc).adjoint()
            assert (Operator(inverse_by_num)).equiv(Operator(inverse_cliff))

    def is_permutation(self, num_elements, perm):
        all_nums = {i: False for i in range(num_elements)}
        for n in all_nums:
            if all_nums[perm[n]]:
                raise QiskitError("Not a permutation")
            else:
                all_nums[perm[n]] = True
        assert all(x == True for x in all_nums.values())

    def test_layers_to_num(self):
        """ Check that all 2 clifford numbers form a permutation over [0, 11519]"""
        cliff_utils = CliffordUtils(num_qubits=2, basis_gates=self.basis_gates)
        cliff_utils.transpile_2q_cliff_layers()
        self.is_permutation(cliff_utils.NUM_CLIFFORD_2_QUBIT, CLIFF_LAYERS_TO_NUM_2Q)

    def test_mapping_layers_to_num(self):
        """ Test the mapping from numbers to layer indices"""
        cliff_utils = CliffordUtils(num_qubits=2, basis_gates=self.basis_gates)
        for i in range(cliff_utils.NUM_CLIFFORD_2_QUBIT):
            indices = cliff_utils.layer_indices_from_num(i)
            reverse_i = cliff_utils.num_from_layer_indices(indices)
            assert(i == reverse_i)
