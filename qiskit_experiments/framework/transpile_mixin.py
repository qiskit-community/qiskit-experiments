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

from qiskit import QuantumCircuit, QuantumRegister, transpile
from qiskit.providers import Backend


class TranspileMixInProtocol(Protocol):
    """A protocol to define a class that can be mixed with transpiler mixins."""

    @property
    def physical_qubits(self):
        """Return the device qubits for the experiment."""

    @property
    def backend(self) -> Backend | None:
        """Return the backend for the experiment"""

    def circuits(self) -> list[QuantumCircuit]:
        """Return a list of experiment circuits.

        Returns:
            A list of :class:`~qiskit.circuit.QuantumCircuit`.

        .. note::
            These circuits should be on qubits ``[0, .., N-1]`` for an
            *N*-qubit experiment. The circuits mapped to physical qubits
            are obtained via the internal :meth:`_transpiled_circuits` method.
        """

    def _transpiled_circuits(self) -> list[QuantumCircuit]:
        ...


class SimpleCircuitExtenderMixin:
    """A transpiler mixin class that maps virtual qubit index to physical.

    When the backend is not set, the experiment class naively assumes
    there are max(physical_qubits) + 1 qubits in the quantum circuits.
    """

    def _transpiled_circuits(
        self: TranspileMixInProtocol,
    ) -> list:
        if hasattr(self.backend, "target"):
            # V2 backend model
            # This model assumes qubit dependent instruction set,
            # but we assume experiment mixed with this class doesn't have such architecture.
            basis_gates = set(self.backend.target.operation_names)
            n_qubits = self.backend.target.num_qubits
        elif hasattr(self.backend, "configuration"):
            # V1 backend model
            basis_gates = set(self.backend.configuration().basis_gates)
            n_qubits = self.backend.configuration().n_qubits
        else:
            # Backend is not set. Naively guess qubit size.
            basis_gates = None
            n_qubits = max(self.physical_qubits) + 1
        return [self._index_mapper(c, basis_gates, n_qubits) for c in self.circuits()]

    def _index_mapper(
        self: TranspileMixInProtocol,
        v_circ: QuantumCircuit,
        basis_gates: set[str] | None,
        n_qubits: int,
    ) -> QuantumCircuit:
        if basis_gates is not None and not basis_gates.issuperset(
            set(v_circ.count_ops().keys()) - {"barrier"}
        ):
            # In Qiskit provider model barrier is not included in target.
            # Use standard circuit transpile when circuit is not ISA.
            return transpile(
                v_circ,
                backend=self.backend,
                initial_layout=list(self.physical_qubits),
            )
        p_qregs = QuantumRegister(n_qubits)
        v_p_map = {q: p_qregs[self.physical_qubits[i]] for i, q in enumerate(v_circ.qubits)}
        p_circ = QuantumCircuit(p_qregs, *v_circ.cregs)
        p_circ.metadata = v_circ.metadata
        for inst, v_qubits, clbits in v_circ.data:
            p_qubits = list(map(v_p_map.get, v_qubits))
            p_circ._append(inst, p_qubits, clbits)
        return p_circ
