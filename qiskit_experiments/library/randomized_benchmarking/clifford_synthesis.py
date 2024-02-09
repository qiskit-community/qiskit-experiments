# This code is part of Qiskit.
#
# (C) Copyright IBM 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
"""
Clifford synthesis plugins for randomized benchmarking
"""
from __future__ import annotations

from typing import Sequence

from qiskit.circuit import QuantumCircuit, Operation
from qiskit.circuit.equivalence_library import SessionEquivalenceLibrary as sel
from qiskit.exceptions import QiskitError
from qiskit.passmanager.flow_controllers import ConditionalController
from qiskit.synthesis.clifford import synth_clifford_full
from qiskit.transpiler import PassManager, CouplingMap, Layout, Target
from qiskit.transpiler.passes import (
    SabreSwap,
    LayoutTransformation,
    BasisTranslator,
    CheckGateDirection,
    GateDirection,
    Optimize1qGatesDecomposition,
)
from qiskit.transpiler.passes.synthesis.plugin import HighLevelSynthesisPlugin


class RBDefaultCliffordSynthesis(HighLevelSynthesisPlugin):
    """Default Clifford synthesis plugin for randomized benchmarking."""

    def run(
        self,
        high_level_object: Operation,
        coupling_map: CouplingMap | None = None,
        target: Target | None = None,
        qubits: Sequence | None = None,
        **options,
    ) -> QuantumCircuit:
        """Run synthesis for the given Clifford.

        Args:
            high_level_object: The operation to synthesize to a
                :class:`~qiskit.circuit.QuantumCircuit` object.
            coupling_map: The reduced coupling map of the backend. For example,
                if physical qubits [5, 6, 7] to be benchmarked is connected
                as 5 - 7 - 6 linearly, the reduced coupling map is 0 - 2 - 1.
            target: A target representing the target backend, which will be ignored in this plugin.
            qubits: List of physical qubits over which the operation is defined,
                which will be ignored in this plugin.
            options: Additional method-specific optional kwargs,
                which must include ``basis_gates``, basis gates to be used for the synthesis.

        Returns:
            The quantum circuit representation of the Operation
            when successful, and ``None`` otherwise.

        Raises:
            QiskitError: If basis_gates is not supplied.
        """
        # synthesize cliffords
        circ = synth_clifford_full(high_level_object)

        # post processing to comply with basis gates and coupling map
        if coupling_map is None:  # Sabre does not work with coupling_map=None
            return circ

        basis_gates = options.get("basis_gates", None)
        if basis_gates is None:
            raise QiskitError("basis_gates are required to run this synthesis plugin")

        basis_gates = list(basis_gates)

        # Run Sabre routing and undo the layout change
        # assuming Sabre routing does not change the initial layout.
        # And then decompose swap gates, fix 2q-gate direction and optimize 1q gates
        initial_layout = Layout.generate_trivial_layout(*circ.qubits)
        undo_layout_change = LayoutTransformation(
            coupling_map=coupling_map, from_layout="final_layout", to_layout=initial_layout
        )

        def _direction_condition(property_set):
            return not property_set["is_direction_mapped"]

        pm = PassManager(
            [
                SabreSwap(coupling_map),
                undo_layout_change,
                BasisTranslator(sel, basis_gates),
                CheckGateDirection(coupling_map),
                ConditionalController(GateDirection(coupling_map), condition=_direction_condition),
                Optimize1qGatesDecomposition(basis=basis_gates),
            ]
        )
        return pm.run(circ)
