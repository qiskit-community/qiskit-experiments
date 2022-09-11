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
Utilities for using the Clifford group in randomized benchmarking
"""

from typing import List, Tuple, Optional, Union
from functools import lru_cache
from math import isclose
import itertools
import numpy as np
from numpy.random import Generator, default_rng

from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit import Gate, Instruction
from qiskit.circuit.library import SdgGate, HGate, SGate
from qiskit.compiler import transpile
from qiskit.providers.backend import Backend
from qiskit.exceptions import QiskitError
from qiskit.quantum_info import Clifford, random_clifford

from .clifford_data import (
    CLIFF_SINGLE_GATE_MAP_1Q,
    CLIFF_SINGLE_GATE_MAP_2Q,
    CLIFF_COMPOSE_DATA_1Q,
    CLIFF_COMPOSE_DATA_2Q,
    CLIFF_INVERSE_DATA_1Q,
    CLIFF_INVERSE_DATA_2Q,
    CLIFF_NUM_TO_LAYERS_2Q,
    CLIFF_LAYERS_TO_NUM_2Q,
)


# The classes VGate and WGate are no actually used in the code - we leave them here to give
# a better understanding of the composition of the layers for 2-qubit Cliffords.
class VGate(Gate):
    """V Gate used in Clifford synthesis."""

    def __init__(self):
        """Create new V Gate."""
        super().__init__("v", 1, [])

    def _define(self):
        """V Gate definition."""
        q = QuantumRegister(1, "q")
        qc = QuantumCircuit(q)
        qc.data = [(SdgGate(), [q[0]], []), (HGate(), [q[0]], [])]
        self.definition = qc


class WGate(Gate):
    """W Gate used in Clifford synthesis."""

    def __init__(self):
        """Create new W Gate."""
        super().__init__("w", 1, [])

    def _define(self):
        """W Gate definition."""
        q = QuantumRegister(1, "q")
        qc = QuantumCircuit(q)
        qc.data = [(HGate(), [q[0]], []), (SGate(), [q[0]], [])]
        self.definition = qc


class CliffordUtils:
    """Utilities for generating 1 and 2 qubit clifford circuits and elements"""

    NUM_CLIFFORD_1_QUBIT = 24
    NUM_CLIFFORD_2_QUBIT = 11520
    CLIFFORD_1_QUBIT_SIG = (2, 3, 4)
    CLIFFORD_2_QUBIT_SIGS = [
        (2, 2, 3, 3, 4, 4),
        (2, 2, 3, 3, 3, 3, 4, 4),
        (2, 2, 3, 3, 3, 3, 4, 4),
        (2, 2, 3, 3, 4, 4),
    ]
    CLIFF_SINGLE_GATE_MAP = {1: CLIFF_SINGLE_GATE_MAP_1Q, 2: CLIFF_SINGLE_GATE_MAP_2Q}
    CLIFF_COMPOSE_DATA = {1: CLIFF_COMPOSE_DATA_1Q, 2: CLIFF_COMPOSE_DATA_2Q}
    CLIFF_INVERSE_DATA = {1: CLIFF_INVERSE_DATA_1Q, 2: CLIFF_INVERSE_DATA_2Q}

    NUM_LAYER_0 = 36
    NUM_LAYER_1 = 20
    NUM_LAYER_2 = 16

    def __init__(self, num_qubits, basis_gates: List[str], backend: Optional[Backend] = None):
        self.num_qubits = num_qubits
        self.basis_gates = basis_gates
        self._backend = backend
        self._transpiled_cliffords_1q = []
        self._transpiled_cliff_layer = {}
        if self.num_qubits == 1:
            self.transpile_1q_cliffords()
        elif self.num_qubits == 2:
            self.transpile_2q_cliff_layers()

    @classmethod
    def clifford_1_qubit(cls, num):
        """Return the 1-qubit clifford element corresponding to `num`
        where `num` is between 0 and 23.
        """
        return Clifford(cls.clifford_1_qubit_circuit(num), validate=False)

    @classmethod
    def clifford_2_qubit(cls, num):
        """Return the 2-qubit clifford element corresponding to `num`
        where `num` is between 0 and 11519.
        """
        return Clifford(cls.clifford_2_qubit_circuit(num), validate=False)

    @classmethod
    @lru_cache(maxsize=24)
    def clifford_1_qubit_circuit(cls, num):
        """Return the 1-qubit clifford circuit corresponding to `num`
        where `num` is between 0 and 23.
        """
        # pylint: disable=unbalanced-tuple-unpacking
        # This is safe since `_unpack_num` returns list the size of the sig
        (i, j, p) = cls._unpack_num(num, cls.CLIFFORD_1_QUBIT_SIG)
        qr = QuantumRegister(1)
        qc = QuantumCircuit(qr)
        if i == 1:
            qc.h(0)
        if j == 1:
            qc.sxdg(0)
        if j == 2:
            qc.s(0)
        if p == 1:
            qc.x(0)
        if p == 2:
            qc.y(0)
        if p == 3:
            qc.z(0)
        return qc

    @classmethod
    @lru_cache(maxsize=11520)
    def clifford_2_qubit_circuit(cls, num):
        """Return the 2-qubit clifford circuit corresponding to `num`
        where `num` is between 0 and 11519.
        """
        vals = cls._unpack_num_multi_sigs(num, cls.CLIFFORD_2_QUBIT_SIGS)
        qr = QuantumRegister(2)
        qc = QuantumCircuit(qr)
        if vals[0] == 0 or vals[0] == 3:
            (form, i0, i1, j0, j1, p0, p1) = vals
        else:
            (form, i0, i1, j0, j1, k0, k1, p0, p1) = vals
        if i0 == 1:
            qc.h(0)
        if i1 == 1:
            qc.h(1)
        if j0 == 1:
            qc.sxdg(0)
        if j0 == 2:
            qc.s(0)
        if j1 == 1:
            qc.sxdg(1)
        if j1 == 2:
            qc.s(1)
        if form in (1, 2, 3):
            qc.cx(0, 1)
        if form in (2, 3):
            qc.cx(1, 0)
        if form == 3:
            qc.cx(0, 1)
        if form in (1, 2):
            if k0 == 1:
                qc.sdg(0)
                qc.h(0)
            if k0 == 2:
                qc.h(0)
                qc.s(0)
            if k1 == 1:
                qc.sdg(1)
                qc.h(1)
            if k1 == 2:
                qc.h(1)
                qc.s(1)
        if p0 == 1:
            qc.x(0)
        if p0 == 2:
            qc.y(0)
        if p0 == 3:
            qc.z(0)
        if p1 == 1:
            qc.x(1)
        if p1 == 2:
            qc.y(1)
        if p1 == 3:
            qc.z(1)
        return qc

    @staticmethod
    def _unpack_num(num, sig):
        r"""Returns a tuple :math:`(a_1, \ldots, a_n)` where
        :math:`0 \le a_i \le \sigma_i` where
        sig=:math:`(\sigma_1, \ldots, \sigma_n)` and num is the sequential
        number of the tuple
        """
        res = []
        for k in sig:
            res.append(num % k)
            num //= k
        return res

    @classmethod
    def _unpack_num_multi_sigs(cls, num, sigs):
        """Returns the result of `_unpack_num` on one of the
        signatures in `sigs`
        """
        for i, sig in enumerate(sigs):
            sig_size = 1
            for k in sig:
                sig_size *= k
            if num < sig_size:
                return [i] + cls._unpack_num(num, sig)
            num -= sig_size
        return None

    def num_from_clifford_single_gate(
        self, inst: Instruction, qubits: List[int], rb_num_qubits: int
    ) -> int:
        """
        This method does the reverse of clifford_1_qubit_circuit and clifford_2_qubit_circuit -
        given a clifford, it returns the corresponding integer, with the mapping
        defined in the above method.

         Returns:
            An integer representing a Clifford consisting of a single gate.

        Raises:
            QiskitError: if the input instruction is not a Clifford instruction.
            QiskitError: if rz is given with a angle that is not Clifford.
        """
        name = inst.name
        if name == "delay":
            return 0

        clifford_gate_set = {key[0] for key in self.CLIFF_SINGLE_GATE_MAP[rb_num_qubits]}
        if name == "rz":
            # The next two are identical up to a phase, which makes no difference
            # for the associated Cliffords
            if isclose(inst.params[0], np.pi) or isclose(inst.params[0], -np.pi):
                map_index = "z"
            elif isclose(inst.params[0], np.pi / 2):
                map_index = "s"
            elif isclose(inst.params[0], -np.pi / 2):
                map_index = "sdg"
            else:
                raise QiskitError("wrong param {} for rz in clifford".format(inst.params[0]))
        elif name in clifford_gate_set:
            map_index = name
            if name == "cz":
                qubits = [min(qubits), max(qubits)]  # for cz we save only [0, 1] since
                # it is a symmetric operation
        else:
            raise QiskitError("Instruction {} could not be converted to Clifford gate".format(name))
        return self.CLIFF_SINGLE_GATE_MAP[rb_num_qubits][(map_index, str(qubits))]

    def compose_num_with_clifford(self, composed_num: int, qc: QuantumCircuit) -> int:
        """Compose a number that represents a Clifford, with a single-gate Clifford, and return the
        number that represents the resulting Clifford."""

        # The numbers corresponding to single gate Cliffords are not in sequence -
        # see num_from_1q_clifford_single_gate. To compute the index in
        # the array CLIFF_COMPOSE_DATA_1Q, we map the numbers to [0, 8].

        map_clifford_num_to_array_index = {}
        num_single_gate_cliffs = len(self.CLIFF_SINGLE_GATE_MAP[self.num_qubits])
        for k in list(self.CLIFF_SINGLE_GATE_MAP[self.num_qubits]):
            map_clifford_num_to_array_index[self.CLIFF_SINGLE_GATE_MAP[self.num_qubits][k]] = list(
                self.CLIFF_SINGLE_GATE_MAP[self.num_qubits].keys()
            ).index(k)
        if self.num_qubits == 1:
            for inst, qargs, _ in qc:
                num = self.num_from_clifford_single_gate(inst=inst, qubits=[0], rb_num_qubits=1)
                index = num_single_gate_cliffs * composed_num + map_clifford_num_to_array_index[num]
                composed_num = self.CLIFF_COMPOSE_DATA[self.num_qubits][index]
        else:  # num_qubits == 2
            for inst, qargs, _ in qc:
                if inst.num_qubits == 2:
                    qubits = [qc.find_bit(qargs[0]).index, qc.find_bit(qargs[1]).index]
                else:
                    qubits = [qc.find_bit(qargs[0]).index]
                num = self.num_from_clifford_single_gate(inst=inst, qubits=qubits, rb_num_qubits=2)
                index = num_single_gate_cliffs * composed_num + map_clifford_num_to_array_index[num]
                composed_num = self.CLIFF_COMPOSE_DATA[self.num_qubits][index]
        return composed_num

    def clifford_inverse_by_num(self, num: int):
        """Return the number of the inverse Clifford to the input num"""
        return self.CLIFF_INVERSE_DATA[self.num_qubits][num]

    def transpile_1q_cliffords(self):
        """Transpile all the 1-qubit Cliffords and store them in self._transpiled_cliffords_1q."""
        if self._transpiled_cliffords_1q != []:
            return
        for num in range(0, CliffordUtils.NUM_CLIFFORD_1_QUBIT):
            circ = CliffordUtils.clifford_1_qubit_circuit(num=num)
            transpiled_circ = transpile(
                circuits=circ,
                optimization_level=1,
                basis_gates=self.basis_gates,
                backend=self._backend,
            )
            self._transpiled_cliffords_1q.append(transpiled_circ)

    def transpiled_clifford_from_num_1q(self, num):
        """Return the transpiled Clifford whose index is 'num'"""
        return self._transpiled_cliffords_1q[num]

    def transpile_2q_cliff_layers(self):
        """Transpile all the 2-qubit layers and store them in self._transpiled_cliffords_layer."""
        if self._transpiled_cliff_layer != {}:
            return
        self._transpiled_cliff_layer[0] = []
        self._transpiled_cliff_layer[1] = []
        self._transpiled_cliff_layer[2] = []
        self._transpile_cliff_layer_0()
        self._transpile_cliff_layer_1()
        self._transpile_cliff_layer_2()

    def _transpile_cliff_layer_0(self):
        """Transpile layer 0. length == 36."""
        if self._transpiled_cliff_layer[0] != []:
            return
        num_h = [0, 1]
        v_w_gates = ["i", "v", "w"]

        for h0, h1, v0, v1 in itertools.product(num_h, num_h, v_w_gates, v_w_gates):
            qr = QuantumRegister(2)
            qc = QuantumCircuit(qr)
            for _ in range(h0):
                qc.h(0)
            for _ in range(h1):
                qc.h(1)
            if v0 == "v":
                qc.sdg(0)  # VGate
                qc.h(0)
            elif v0 == "w":
                qc.h(0)  # WGate
                qc.s(0)
            if v1 == "v":
                qc.sdg(1)  # VGate
                qc.h(1)
            elif v1 == "w":
                qc.h(1)  # WGate
                qc.s(1)
            transpiled = transpile(qc, optimization_level=1, basis_gates=self.basis_gates)
            self._transpiled_cliff_layer[0].append(transpiled)

    def _transpile_cliff_layer_1(self):
        """Transpile layer 1. length == 20."""
        if self._transpiled_cliff_layer[1] != []:
            return
        v_w_gates = ["i", "v", "w"]
        qr = QuantumRegister(2)
        qc = QuantumCircuit(qr)
        transpiled = transpile(
            qc, optimization_level=1, basis_gates=self.basis_gates, backend=self._backend
        )
        self._transpiled_cliff_layer[1].append(transpiled)

        for v0, v1 in itertools.product(v_w_gates, v_w_gates):
            qc = QuantumCircuit(qr)
            qc.cx(0, 1)
            if v0 == "v":
                qc.sdg(0)  # VGate
                qc.h(0)
            elif v0 == "w":
                qc.h(0)  # WGate
                qc.s(0)
            if v1 == "v":
                qc.sdg(1)  # VGate
                qc.h(1)
            elif v1 == "w":
                qc.h(1)  # WGate
                qc.s(1)
            transpiled = transpile(
                qc, optimization_level=1, basis_gates=self.basis_gates, backend=self._backend
            )
            self._transpiled_cliff_layer[1].append(transpiled)

        for v0, v1 in itertools.product(v_w_gates, v_w_gates):
            qc = QuantumCircuit(qr)
            qc.cx(0, 1)
            qc.cx(1, 0)
            if v0 == "v":
                qc.sdg(0)  # VGate
                qc.h(0)
            elif v0 == "w":
                qc.h(0)  # WGate
                qc.s(0)
            if v1 == "v":
                qc.sdg(1)  # VGate
                qc.h(1)
            elif v1 == "w":
                qc.h(1)  # WGate
                qc.s(1)
            transpiled = transpile(
                qc, optimization_level=1, basis_gates=self.basis_gates, backend=self._backend
            )
            self._transpiled_cliff_layer[1].append(transpiled)

        qc = QuantumCircuit(qr)
        qc.cx(0, 1)
        qc.cx(1, 0)
        qc.cx(0, 1)
        transpiled = transpile(
            qc, optimization_level=1, basis_gates=self.basis_gates, backend=self._backend
        )
        self._transpiled_cliff_layer[1].append(transpiled)

    def _transpile_cliff_layer_2(self):
        """Transpile layer 2. length == 16."""
        if self._transpiled_cliff_layer[2] != []:
            return
        pauli = ["i", "x", "y", "z"]
        for p0, p1 in itertools.product(pauli, pauli):
            qr = QuantumRegister(2)
            qc = QuantumCircuit(qr)
            if p0 != "i":
                qc._append(Gate(p0, 1, []), [qr[0]], [])
            if p1 != "i":
                qc._append(Gate(p1, 1, []), [qr[1]], [])

            transpiled = transpile(
                qc, optimization_level=1, basis_gates=self.basis_gates, backend=self._backend
            )
            self._transpiled_cliff_layer[2].append(transpiled)

    def create_random_clifford(self, rng: Generator) -> QuantumCircuit:
        """For 1-qubit, select a random transpiled Clifford. For 2-qubits, select 3 indices at random in
        _transpiled_cliff_layer, and compose them to create a Clifford.

         Returns:
            A random Clifford represented as a :class:`QuantumCircuit`.

        Raises:
            QiskitError: if the number of qubits is greater than 2.

        """
        if rng is None:
            rng = default_rng()
        if isinstance(rng, int):
            rng = default_rng(rng)
        if self.num_qubits == 1:
            rand = rng.integers(self.NUM_CLIFFORD_1_QUBIT)
            return self._transpiled_cliffords_1q[rand]
        elif self.num_qubits == 2:
            rand = rng.integers(self.NUM_CLIFFORD_2_QUBIT)
            return self.create_cliff_from_num(num=rand)
        else:
            raise QiskitError("create_random_clifford is not supported for more than 2 qubits")

    def create_cliff_from_num(self, num) -> QuantumCircuit:
        """Create a Clifford corresponding to a give number.

         Returns:
            A Clifford represented as a :class:`QuantumCircuit`.

        Raises:
            QiskitError: if the number of qubits is greater than 2.

        """
        if self.num_qubits == 1:
            return self._transpiled_cliffords_1q[num]
        elif self.num_qubits == 2:
            triplet = self.layer_indices_from_num(num)
            return self.transpiled_cliff_from_layer_nums(triplet)
        else:
            raise QiskitError("create_cliff_from_num is not supported for more than 2 qubits")

    @lru_cache(NUM_CLIFFORD_2_QUBIT)
    def transpiled_cliff_from_layer_nums(self, triplet: Tuple):
        q0 = self._transpiled_cliff_layer[0][triplet[0]]
        q1 = self._transpiled_cliff_layer[1][triplet[1]]
        q2 = self._transpiled_cliff_layer[2][triplet[2]]
        qc = q0.copy()
        qc.compose(q1, inplace=True)
        qc.compose(q2, inplace=True)
        return qc

    @classmethod
    def num_from_layer_indices(cls, triplet: Tuple) -> int:
        """Return the clifford number corresponding to the input triplet."""
        num = (
            triplet[0] * cls.NUM_LAYER_1 * cls.NUM_LAYER_2
            + triplet[1] * cls.NUM_LAYER_2
            + triplet[2]
        )
        return CLIFF_LAYERS_TO_NUM_2Q[num]

    @classmethod
    def layer_indices_from_num(cls, num: int) -> Tuple[int]:
        """Return the triplet of layer indices corresponding to the input number."""
        return CLIFF_NUM_TO_LAYERS_2Q[num]

    def inverse_cliff(self, cliff_num: int) -> QuantumCircuit:
        """Return the Clifford that is the inverse of the clifford corresponding to Cliff_num

        Returns:
            A Clifford represented as a :class:`QuantumCircuit`.

        Raises:
            QiskitError: if the number of qubits is greater than 2.

        """
        inverse_clifford_num = self.clifford_inverse_by_num(cliff_num)
        if self.num_qubits == 1:
            return self._transpiled_cliffords_1q[inverse_clifford_num]
        elif self.num_qubits == 2:
            indices = CliffordUtils.layer_indices_from_num(inverse_clifford_num)
            return self.transpiled_cliff_from_layer_nums(indices)
        else:
            raise QiskitError("inverse_cliff is not supported for more than 2 qubits")

    def random_clifford_circuits(
        self, size: int = 1, rng: Optional[Union[int, Generator]] = None
    ) -> List[QuantumCircuit]:
        """Generate a list of random clifford circuits.
        Used for 3 or more qubits"""
        if self.num_qubits > 2:
            return [random_clifford(self.num_qubits, seed=rng).to_circuit() for _ in range(size)]

        if rng is None:
            rng = default_rng()

        if isinstance(rng, int):
            rng = default_rng(rng)

        if self.num_qubits == 1:
            samples = rng.integers(24, size=size)
            return [self.clifford_1_qubit_circuit(i) for i in samples]
        else:
            samples = rng.integers(11520, size=size)
            return [self.clifford_2_qubit_circuit(i) for i in samples]
