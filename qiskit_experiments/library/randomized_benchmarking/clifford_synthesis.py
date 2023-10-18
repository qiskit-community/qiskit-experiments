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
from qiskit.compiler import transpile
from qiskit.synthesis.clifford import synth_clifford_full
from qiskit.transpiler import PassManager, CouplingMap, Layout
from qiskit.transpiler.passes import SabreSwap, LayoutTransformation
from qiskit.transpiler.passes.synthesis.plugin import HighLevelSynthesisPlugin


class RBDefaultCliffordSynthesis(HighLevelSynthesisPlugin):
    """Default Clifford synthesis plugin for randomized benchmarking."""

    def run(
        self,
        high_level_object: Operation,
        basis_gates: Sequence[str] | None = None,
        coupling_map: CouplingMap | None = None,
        **options,
    ) -> QuantumCircuit:
        """Run synthesis for the given Clifford.

        Args:
            high_level_object: The operation to synthesize to a
                :class:`~qiskit.circuit.QuantumCircuit` object.
            basis_gates: The basis gates to be used for the synthesis.
            coupling_map: The reduced coupling map of the backend. For example,
                if physical qubits [5, 6, 7] to be benchmarked is connected
                as 5 - 7 - 6 linearly, the reduced coupling map is 0 - 2 - 1.
            options: Additional method-specific optional kwargs.

        Returns:
            The quantum circuit representation of the Operation
                when successful, and ``None`` otherwise.
        """
        # synthesize cliffords
        circ = synth_clifford_full(high_level_object)

        # post processing to comply with basis gates and coupling map
        if coupling_map is None:  # Sabre does not work with coupling_map=None
            return circ
        # run Sabre routing and undo the layout change
        # assuming Sabre routing does not change the initial layout
        initial_layout = Layout.generate_trivial_layout(*circ.qubits)
        undo_layout_change = LayoutTransformation(
            coupling_map=coupling_map, from_layout="final_layout", to_layout=initial_layout
        )
        pm = PassManager([SabreSwap(coupling_map), undo_layout_change])
        circ = pm.run(circ)
        # for fixing 2q-gate direction and optimizing 1q gates
        return transpile(
            circ,
            basis_gates=basis_gates,
            coupling_map=coupling_map,
            optimization_level=1,
        )
