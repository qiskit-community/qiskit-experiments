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

"""Transpile mixin class."""

from __future__ import annotations
from typing import Protocol

from qiskit import QuantumCircuit, QuantumRegister
from qiskit.providers import Backend


class TranspileMixInProtocol(Protocol):
    """A protocol to define a class that can be mixed with transpiler mixins."""

    @property
    def physical_qubits(self):
        ...

    @property
    def backend(self) -> Backend | None:
        ...

    def circuits(self) -> list[QuantumCircuit]:
        ...

    def _transpiled_circuits(self) -> list[QuantumCircuit]:
        ...


class SimpleCircuitExtender:
    """A transpiler mixin class that maps virtual qubit index to physical.

    Experiment class returns virtual circuits when the backend is not set.
    """

    def _transpiled_circuits(
        self: TranspileMixInProtocol,
    ) -> list:
        if hasattr(self.backend, "num_qubits"):
            # V2 backend model
            n_qubits = self.backend.num_qubits
        elif hasattr(self.backend, "configuration"):
            # V1 backend model
            n_qubits = self.backend.configuration().n_qubits
        else:
            # Backend is not set. Return virtual circuits as is.
            return self.circuits()
        return [self._index_mapper(c, n_qubits) for c in self.circuits()]

    def _index_mapper(
        self: TranspileMixInProtocol,
        v_circ: QuantumCircuit,
        n_qubits: int,
    ) -> QuantumCircuit:
        p_qregs = QuantumRegister(n_qubits)
        v_p_map = {q: p_qregs[self.physical_qubits[i]] for i, q in enumerate(v_circ.qubits)}
        p_circ = QuantumCircuit(p_qregs, *v_circ.cregs)
        p_circ.metadata = v_circ.metadata
        for inst, v_qubits, clbits in v_circ.data:
            p_qubits = list(map(v_p_map.get, v_qubits))
            p_circ._append(inst, p_qubits, clbits)
        return p_circ
