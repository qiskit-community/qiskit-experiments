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
import itertools
import os
from functools import lru_cache
from numbers import Integral
from typing import Optional, Union, Tuple

import numpy as np
from numpy.random import Generator, default_rng

from qiskit.circuit import Gate, Instruction
from qiskit.circuit import QuantumCircuit, QuantumRegister
from qiskit.circuit.library import SdgGate, HGate, SGate, XGate, YGate, ZGate
from qiskit.exceptions import QiskitError
from qiskit.quantum_info import Clifford, random_clifford


@lru_cache(maxsize=None)
def _clifford_1q_int_to_instruction(num: Integral) -> Instruction:
    return CliffordUtils.clifford_1_qubit_circuit(num).to_instruction()


@lru_cache(maxsize=11520)
def _clifford_2q_int_to_instruction(num: Integral) -> Instruction:
    return CliffordUtils.clifford_2_qubit_circuit(num).to_instruction()


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
    CLIFFORD_2_QUBIT_SIGS = [  # TODO: depracate
        (2, 2, 3, 3, 4, 4),
        (2, 2, 3, 3, 3, 3, 4, 4),
        (2, 2, 3, 3, 3, 3, 4, 4),
        (2, 2, 3, 3, 4, 4),
    ]

    @classmethod
    @lru_cache(maxsize=24)
    def clifford_1_qubit(cls, num):
        """Return the 1-qubit clifford element corresponding to `num`
        where `num` is between 0 and 23.
        """
        return Clifford(cls.clifford_1_qubit_circuit(num), validate=False)

    @classmethod
    @lru_cache(maxsize=11520)
    def clifford_2_qubit(cls, num):
        """Return the 2-qubit clifford element corresponding to `num`
        where `num` is between 0 and 11519.
        """
        return Clifford(cls.clifford_2_qubit_circuit(num), validate=False)

    def random_cliffords(
        self, num_qubits: int, size: int = 1, rng: Optional[Union[int, Generator]] = None
    ):
        """Generate a list of random clifford elements"""
        if num_qubits > 2:
            return random_clifford(num_qubits, seed=rng)

        if rng is None:
            rng = default_rng()

        if isinstance(rng, int):
            rng = default_rng(rng)

        if num_qubits == 1:
            samples = rng.integers(24, size=size)
            return [Clifford(self.clifford_1_qubit_circuit(i), validate=False) for i in samples]
        else:
            samples = rng.integers(11520, size=size)
            return [Clifford(self.clifford_2_qubit_circuit(i), validate=False) for i in samples]

    def random_clifford_circuits(
        self, num_qubits: int, size: int = 1, rng: Optional[Union[int, Generator]] = None
    ):
        """Generate a list of random clifford circuits"""
        if num_qubits > 2:
            return [random_clifford(num_qubits, seed=rng).to_circuit() for _ in range(size)]

        if rng is None:
            rng = default_rng()

        if isinstance(rng, int):
            rng = default_rng(rng)

        if num_qubits == 1:
            samples = rng.integers(24, size=size)
            return [self.clifford_1_qubit_circuit(i) for i in samples]
        else:
            samples = rng.integers(11520, size=size)
            return [self.clifford_2_qubit_circuit(i) for i in samples]

    @classmethod
    @lru_cache(maxsize=24)
    def clifford_1_qubit_circuit(cls, num):
        """Return the 1-qubit clifford circuit corresponding to `num`
        where `num` is between 0 and 23.
        """
        unpacked = cls._unpack_num(num, (2, 3, 4))
        i, j, p = unpacked[0], unpacked[1], unpacked[2]
        qc = QuantumCircuit(1, name=f"Clifford-1Q({num})")
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
        qc = QuantumCircuit(2, name=f"Clifford-2Q({num})")
        for layer, idx in enumerate(_layer_indices_from_num(num)):
            qc.compose(_CLIFFORD_LAYER[layer][idx], inplace=True)
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


_CLIFF_SINGLE_GATE_MAP_1Q = {
    ("id", (0,)): 0,
    ("h", (0,)): 1,
    ("sxdg", (0,)): 2,
    ("s", (0,)): 4,
    ("x", (0,)): 6,
    ("sx", (0,)): 8,
    ("y", (0,)): 12,
    ("z", (0,)): 18,
    ("sdg", (0,)): 22,
}
_CLIFF_SINGLE_GATE_MAP_2Q = {
    ("id", (0,)): 0,
    ("id", (1,)): 0,
    ("h", (0,)): 5760,
    ("h", (1,)): 2880,
    ("sxdg", (0,)): 6720,
    ("sxdg", (1,)): 3200,
    ("s", (0,)): 7680,
    ("s", (1,)): 3520,
    ("x", (0,)): 4,
    ("x", (1,)): 1,
    ("sx", (0,)): 6724,
    ("sx", (1,)): 3201,
    ("y", (0,)): 8,
    ("y", (1,)): 2,
    ("z", (0,)): 12,
    ("z", (1,)): 3,
    ("sdg", (0,)): 7692,
    ("sdg", (1,)): 3523,
    ("cx", (0, 1)): 16,
    ("cx", (1, 0)): 2336,
    ("cz", (0, 1)): 368,
    ("cz", (1, 0)): 368,
}


########
# Functions for 1-qubit integer Clifford operations
def compose_1q(lhs: Integral, rhs: Integral) -> Integral:
    """Return the composition of 1-qubit clifford integers."""
    return _CLIFFORD_COMPOSE_1Q[lhs][rhs]


def inverse_1q(num: Integral) -> Integral:
    """Return the inverse of 1-qubit clifford integers."""
    return _CLIFFORD_INVERSE_1Q[num]


def num_from_1q_circuit(qc: QuantumCircuit) -> Integral:
    """Convert a given 1-qubit Clifford circuit to the corresponding integer."""
    num = 0
    for inst in qc:
        rhs = _num_from_1q_gate(op=inst.operation)
        num = _CLIFFORD_COMPOSE_1Q[num][rhs]
    return num


def _num_from_1q_gate(op: Instruction) -> int:
    """
    Convert a given 1-qubit clifford operation to the corresponding integer.
    Note that supported operations are limited to ones in `CLIFF_SINGLE_GATE_MAP_1Q` or Rz gate.

    Args:
        op: operation to be converted.

    Returns:
        An integer representing a Clifford consisting of a single operation.

    Raises:
        QiskitError: if the input instruction is not a Clifford instruction.
        QiskitError: if rz is given with a angle that is not Clifford.
    """
    if op.name in {"delay", "barrier"}:
        return 0
    try:
        name = _deparameterized_name(op)
        return _CLIFF_SINGLE_GATE_MAP_1Q[(name, (0,))]
    except QiskitError as err:
        raise QiskitError(
            f"Parameterized instruction {op.name} could not be converted to integer Clifford"
        ) from err
    except KeyError as err:
        raise QiskitError(
            f"Instruction {op.name} could not be converted to integer Clifford"
        ) from err


def _deparameterized_name(inst: Instruction) -> str:
    if inst.name == "rz":
        if np.isclose(inst.params[0], np.pi) or np.isclose(inst.params[0], -np.pi):
            return "z"
        elif np.isclose(inst.params[0], np.pi / 2):
            return "s"
        elif np.isclose(inst.params[0], -np.pi / 2):
            return "sdg"
        else:
            raise QiskitError("Wrong param {} for rz in clifford".format(inst.params[0]))

    return inst.name


def _load_clifford_compose_1q():
    dirname = os.path.dirname(__file__)
    data = np.load(f"{dirname}/data/clifford_compose_1q.npz")
    return data["table"]


def _load_clifford_inverse_1q():
    dirname = os.path.dirname(__file__)
    data = np.load(f"{dirname}/data/clifford_inverse_1q.npz")
    return data["table"]


_CLIFFORD_COMPOSE_1Q = _load_clifford_compose_1q()
_CLIFFORD_INVERSE_1Q = _load_clifford_inverse_1q()


########
# Functions for 2-qubit integer Clifford operations
def compose_2q(lhs: Integral, rhs: Integral) -> Integral:
    """Return the composition of 2-qubit clifford integers."""
    num = lhs
    for layer, idx in enumerate(_layer_indices_from_num(rhs)):
        circ = _CLIFFORD_LAYER[layer][idx]
        num = _compose_num_with_circuit_2q(num, circ)
    return num


def inverse_2q(num: Integral) -> Integral:
    """Return the inverse of 2-qubit clifford integers."""
    return _CLIFFORD_INVERSE_2Q[num]


def num_from_2q_circuit(qc: QuantumCircuit) -> Integral:
    """Convert a given 2-qubit Clifford circuit to the corresponding integer."""
    return _compose_num_with_circuit_2q(0, qc)


# Shortcut to call the function converting circuit to num by number of qubits
# TODO: Too much? (if it is usefule, add more explanation)
num_from_circuit = {
    1: num_from_1q_circuit,
    2: num_from_2q_circuit,
}


def _compose_num_with_circuit_2q(num: Integral, qc: QuantumCircuit) -> Integral:
    """Compose a number that represents a Clifford, with a Clifford circuit, and return the
    number that represents the resulting Clifford."""
    lhs = num
    for inst in qc:
        qubits = tuple(qc.find_bit(q).index for q in inst.qubits)
        rhs = _num_from_2q_gate(op=inst.operation, qubits=qubits)
        try:
            lhs = _CLIFFORD_COMPOSE_2Q_GATE[lhs][rhs]
        except KeyError as err:
            raise Exception(f"_CLIFFORD_COMPOSE_2Q_GATE[{lhs}][{rhs}]") from err
    return lhs


def _num_from_2q_gate(
    op: Instruction, qubits: Optional[Union[Tuple[int, int], Tuple[int]]] = None
) -> int:
    """
    Convert a given 1-qubit clifford operation to the corresponding integer.
    Note that supported operations are limited to ones in `CLIFF_SINGLE_GATE_MAP_2Q` or Rz gate.

    Args:
        op: operation of instruction to be converted.
        qubits: qubits to which the operation applies

    Returns:
        An integer representing a Clifford consisting of a single operation.

    Raises:
        QiskitError: if the input instruction is not a Clifford instruction.
        QiskitError: if rz is given with a angle that is not Clifford.
    """
    if op.name in {"delay", "barrier"}:
        return 0

    qubits = qubits or (0, 1)
    try:
        name = _deparameterized_name(op)
        return _CLIFF_SINGLE_GATE_MAP_2Q[(name, qubits)]
    except QiskitError as err:
        raise QiskitError(
            f"Parameterized instruction {op.name} could not be converted to integer Clifford"
        ) from err
    except KeyError as err:
        raise QiskitError(
            f"Instruction {op.name} on {qubits} could not be converted to integer Clifford"
        ) from err


def _append_v_w(qc, vw0, vw1):
    if vw0 == "v":
        qc.sdg(0)
        qc.h(0)
    elif vw0 == "w":
        qc.h(0)
        qc.s(0)
    if vw1 == "v":
        qc.sdg(1)
        qc.h(1)
    elif vw1 == "w":
        qc.h(1)
        qc.s(1)


def _create_cliff_2q_layer_0():
    """Layer 0 consists of 0 or 1 H gates on each qubit, followed by 0/1/2 V gates on each qubit.
    Number of Cliffords == 36."""
    circuits = []
    num_h = [0, 1]
    v_w_gates = ["i", "v", "w"]
    for h0, h1, v0, v1 in itertools.product(num_h, num_h, v_w_gates, v_w_gates):
        qc = QuantumCircuit(2)
        for _ in range(h0):
            qc.h(0)
        for _ in range(h1):
            qc.h(1)
        _append_v_w(qc, v0, v1)
        circuits.append(qc)
    return circuits


def _create_cliff_2q_layer_1():
    """Layer 1 consists of one of the following:
    - nothing
    - cx(0,1) followed by 0/1/2 V gates on each qubit
    - cx(0,1), cx(1,0) followed by 0/1/2 V gates on each qubit
    - cx(0,1), cx(1,0), cx(0,1)
    Number of Cliffords == 20."""
    circuits = [QuantumCircuit(2)]  # identity at the beginning

    v_w_gates = ["i", "v", "w"]
    for v0, v1 in itertools.product(v_w_gates, v_w_gates):
        qc = QuantumCircuit(2)
        qc.cx(0, 1)
        _append_v_w(qc, v0, v1)
        circuits.append(qc)

    for v0, v1 in itertools.product(v_w_gates, v_w_gates):
        qc = QuantumCircuit(2)
        qc.cx(0, 1)
        qc.cx(1, 0)
        _append_v_w(qc, v0, v1)
        circuits.append(qc)

    qc = QuantumCircuit(2)  # swap at the end
    qc.cx(0, 1)
    qc.cx(1, 0)
    qc.cx(0, 1)
    circuits.append(qc)
    return circuits


def _create_cliff_2q_layer_2():
    """Layer 2 consists of a Pauli gate on each qubit {Id, X, Y, Z}.
    Number of Cliffords == 16."""
    circuits = []
    pauli = ("i", XGate(), YGate(), ZGate())
    for p0, p1 in itertools.product(pauli, pauli):
        qc = QuantumCircuit(2)
        if p0 != "i":
            qc.append(p0, [0])
        if p1 != "i":
            qc.append(p1, [1])
        circuits.append(qc)
    return circuits


_CLIFFORD_LAYER = (
    _create_cliff_2q_layer_0(),
    _create_cliff_2q_layer_1(),
    _create_cliff_2q_layer_2(),
)
_NUM_LAYER_0 = 36
_NUM_LAYER_1 = 20
_NUM_LAYER_2 = 16


def _num_from_layer_indices(triplet: Tuple[Integral, Integral, Integral]) -> Integral:
    """Return the clifford number corresponding to the input triplet."""
    num = triplet[0] * _NUM_LAYER_1 * _NUM_LAYER_2 + triplet[1] * _NUM_LAYER_2 + triplet[2]
    return num


def _layer_indices_from_num(num: Integral) -> Tuple[Integral, Integral, Integral]:
    """Return the triplet of layer indices corresponding to the input number."""
    idx2 = num % _NUM_LAYER_2
    num = num // _NUM_LAYER_2
    idx1 = num % _NUM_LAYER_1
    idx0 = num // _NUM_LAYER_1
    return idx0, idx1, idx2


def _load_clifford_compose_2q_gate():
    dirname = os.path.dirname(__file__)
    data = np.load(f"{dirname}/data/clifford_compose_2q_gate.npz")
    table = []
    for row in data["table"]:
        dic = {rhs: result for result, rhs in zip(row, _CLIFF_SINGLE_GATE_MAP_2Q.values())}
        table.append(dic)
    return table


def _load_clifford_inverse_2q():
    dirname = os.path.dirname(__file__)
    data = np.load(f"{dirname}/data/clifford_inverse_2q.npz")
    return data["table"]


_CLIFFORD_COMPOSE_2Q_GATE = _load_clifford_compose_2q_gate()
_CLIFFORD_INVERSE_2Q = _load_clifford_inverse_2q()
