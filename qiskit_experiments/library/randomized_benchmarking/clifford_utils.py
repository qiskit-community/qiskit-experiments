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

from typing import Optional, Union, List
from functools import lru_cache
import numpy as np
from numpy.random import Generator, default_rng
from math import isclose
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit import Gate
from qiskit.circuit.library import SdgGate, HGate, SGate, SXdgGate
from qiskit.quantum_info import Clifford, random_clifford
from qiskit.compiler import transpile
from qiskit.providers.aer import AerSimulator
from .clifford_data import CLIFF_COMPOSE_DATA, CLIFF_INVERSE_DATA
from qiskit.exceptions import QiskitError


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

    def clifford_1_qubit(self, num):
        """Return the 1-qubit clifford element corresponding to `num`
        where `num` is between 0 and 23.
        """
        return Clifford(self.clifford_1_qubit_circuit(num), validate=False)

    def clifford_2_qubit(self, num):
        """Return the 2-qubit clifford element corresponding to `num`
        where `num` is between 0 and 11519.
        """
        return Clifford(self.clifford_2_qubit_circuit(num), validate=False)

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

    @lru_cache(maxsize=24)
    def clifford_1_qubit_circuit(self, num):
        """Return the 1-qubit clifford circuit corresponding to `num`
        where `num` is between 0 and 23.
        """
        # pylint: disable=unbalanced-tuple-unpacking
        # This is safe since `_unpack_num` returns list the size of the sig
        (i, j, p) = self._unpack_num(num, self.CLIFFORD_1_QUBIT_SIG)
        qr = QuantumRegister(1)
        qc = QuantumCircuit(qr)
        if i == 1:
            qc.h(0)
        if j == 1:
            qc._append(SXdgGate(), [qr[0]], [])
        if j == 2:
            qc._append(SGate(), [qr[0]], [])
        if p == 1:
            qc.x(0)
        if p == 2:
            qc.y(0)
        if p == 3:
            qc.z(0)
        return qc

    @lru_cache(maxsize=11520)
    def clifford_2_qubit_circuit(self, num):
        """Return the 2-qubit clifford circuit corresponding to `num`
        where `num` is between 0 and 11519.
        """
        vals = self._unpack_num_multi_sigs(num, self.CLIFFORD_2_QUBIT_SIGS)
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
                qc._append(VGate(), [qr[0]], [])
            if k0 == 2:
                qc._append(WGate(), [qr[0]], [])
            if k1 == 1:
                qc._append(VGate(), [qr[1]], [])
            if k1 == 2:
                qc._append(VGate(), [qr[1]], [])
                qc._append(VGate(), [qr[1]], [])
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

    def _unpack_num(self, num, sig):
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

    def _unpack_num_multi_sigs(self, num, sigs):
        """Returns the result of `_unpack_num` on one of the
        signatures in `sigs`
        """
        for i, sig in enumerate(sigs):
            sig_size = 1
            for k in sig:
                sig_size *= k
            if num < sig_size:
                return [i] + self._unpack_num(num, sig)
            num -= sig_size
        return None

    def generate_1q_transpiled_clifford_circuits(basis_gates : List[str]):
        utils = CliffordUtils()
        circs = []
        for num in range(0, 24):
            circ = utils.clifford_1_qubit_circuit(num=num)
            circs.append(circ)

        backend = AerSimulator()
        transpiled_circs = []

        for i, circ in enumerate(circs):
            transpiled_circ = transpile(circ, backend, optimization_level=1, basis_gates=basis_gates)
            transpiled_circs.append(transpiled_circ)
        return transpiled_circs

    def num_from_1_qubit_clifford_single_gate(inst, basis_gates):
        """
        This method does the reverse of clifford_1_qubit_circuit -
        given a clifford, it returns the corresponding integer, with the mapping
        defined in the above method.
        The mapping is in the context of the basis_gates. Therefore, we define here
        the possible supersets of basis gates, and verify that the given inst belong to
        one of these sets.
        """
        name = inst.name
        general_cliff_list = ["id", "h", "sdg", "s", "x", "sx", "y", "z"]
        transpiled_cliff_list = ["sx", "rz", "cx"]

        if not name in basis_gates:
            raise QiskitError(
            "Instruction {} is not in the basis gates".format(inst.name)
            )
        if(set(basis_gates).issubset(set(general_cliff_list))):
            num_dict = {"id":0, "h":1, "sxdg":2, "s":4, "x":6, "sx":8, "y":12, "z":18, "sdg":22}
            return num_dict[inst.name]

        if (set(basis_gates).issubset(set(transpiled_cliff_list))):
            if name == "sx":
                return 8
            if name == "rz":
                # The next two are identical up to a phase, which makes no difference
                # for the associated Cliffords
                if isclose(inst.params[0], np.pi) or isclose(inst.params[0], -np.pi):
                    return 18
                if isclose(inst.params[0], np.pi / 2):
                    return 4
                if isclose(inst.params[0], -np.pi / 2):
                    return 22
                else:
                    raise QiskitError("wrong param {} for rz in clifford".format(inst.params[0]))

        raise QiskitError(
            "Instruction {} could not be converted to Clifford gate".format(name)
            )

    def num_from_1_qubit_clifford(qc, basis_gates):
        composed_num = 0
        for inst in qc:
            num = CliffordUtils.num_from_1_qubit_clifford_single_gate(inst[0], basis_gates)
            composed_num = CLIFF_COMPOSE_DATA[(composed_num, num)]
        return composed_num
