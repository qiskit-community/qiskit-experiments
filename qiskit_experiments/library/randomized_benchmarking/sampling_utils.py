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
Utilities for sampling layers in randomized benchmarking experiments
"""

import warnings
from typing import Optional, Union, Sequence, List, Tuple

import numpy as np
from numpy.random import Generator, default_rng
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.exceptions import QiskitError
from qiskit.circuit import Gate
from qiskit.circuit.library import (
    IGate,
    XGate,
    YGate,
    ZGate,
    CXGate,
    CYGate,
    SGate,
    SdgGate,
    SXGate,
    SXdgGate,
)
from qiskit.quantum_info import random_unitary, random_clifford, Pauli
from qiskit.extensions import UnitaryGate
from qiskit.converters import circuit_to_dag
from .clifford_utils import CliffordUtils


CZGate = ZGate().control(1)
CSGate = SGate().control(1)
CSdgGate = SdgGate().control(1)
CSXGate = SXGate().control(1)
CSXdgGate = SXdgGate().control(1)

gate_set_dict = {
    "cx": [CXGate()],
    "cy": [CYGate()],
    "cz": [CZGate],
    "cs": [CSGate, CSdgGate],
    "csx": [CSXGate, CSXdgGate],
}


class SamplingUtils:
    """Utilities for sampling one- and two-qubit layers of different gate sets for randomized
    benchmarking experiments"""

    def __init__(self):
        self._clifford_utils = CliffordUtils()

    def random_one_qubit_tensor_products(
        self,
        num_qubits: int,
        size: Optional[int] = 1,
        gate_set: Optional[str] = "clifford",
        rng: Optional[Union[int, Generator]] = None,
    ) -> List[QuantumCircuit]:
        """Generate a list of layers of random Clifford or Haar-random SU(2) gates

        Args:
            num_qubits: Number of qubits
            size: Number of layers of one-qubit gates to sample
            gate_set: Type of one-qubit gates to sample, either "clifford" or
                      "su2" / "su(2)" / "haar"
            rng: Random seed

        Raises:
            QiskitError: If gate_set is not "clifford" or "su2" / "su(2)" / "haar"

        Returns:
            List of QuantumCircuits
        """
        if isinstance(gate_set, list) or not (
            gate_set.casefold() in ["clifford", "su2", "su(2)", "haar"]
        ):
            raise QiskitError("gate_set must be one of 'clifford', 'su2', 'su(2)', 'haar'")

        if rng is None:
            rng = default_rng()

        if isinstance(rng, int):
            rng = default_rng(rng)

        if gate_set.casefold() == "clifford":
            parallel_unitary_list = [QuantumCircuit(num_qubits) for _ in range(size)]
            for i in range(size):
                for j in range(num_qubits):
                    parallel_unitary_list[i].append(
                        UnitaryGate(random_clifford(1, seed=rng)), [j], []
                    )
        elif gate_set.casefold() in ["su2", "su(2)", "haar"]:
            parallel_unitary_list = [QuantumCircuit(num_qubits) for _ in range(size)]
            for i in range(size):
                for j in range(num_qubits):
                    parallel_unitary_list[i].append(
                        UnitaryGate(random_unitary(2, seed=rng)), [j], []
                    )

        return parallel_unitary_list

    def random_edgegrab_circuits(
        self,
        qubits: Sequence[int],
        coupling_map: list,
        one_qubit_gate_set: Optional[str] = "clifford",
        two_qubit_gate_set: Optional[str] = "cx",
        two_qubit_gate_density: Optional[float] = 0.2,
        separate_one_qubit_layer: Optional[bool] = True,
        size: int = 1,
        rng: Optional[Union[int, Generator]] = None,
    ) -> List[QuantumCircuit]:
        """Generate a list of random circuits sampled using the edgegrab algorithm

        Args:
            qubits: Sequence of integers representing the physical qubits
            coupling_map: List of edges, where an edge is a list of 2 integers
            one_qubit_gate_set: a str specifying a set of one-qubit gates of a list of one-qubit
                                gates
            two_qubit_gate_set: a str specifying a set of two-qubit gates or a list of two-qubit
                                gates
            two_qubit_gate_density: :math:`1/2` times the expected fraction of qubits with CX gates
            separate_one_qubit_layer: if False, populate non-grabbed edges with gates from
                                      one_qubit_gate_set. If True, create a separate layer filled
                                      with one qubit gates.
            size: length of RB sequence
            rng: Random seed

        Raises:
            QiskitError: If invalid one qubit gate set is provided
            Warning: If device has no connectivity or two_qubit_gate_density is too high

        Returns:
            List of QuantumCircuits

        Ref: arXiv:2008.11294v2
        """
        if isinstance(one_qubit_gate_set, list) or not (
            one_qubit_gate_set.casefold() in ["clifford", "su2", "su(2)", "haar"]
        ):
            raise QiskitError(
                "one_qubit_gate_set must be one of 'clifford', 'su2', 'su(2)', 'haar'"
            )

        if isinstance(two_qubit_gate_set, str):
            two_qubit_gate_set = gate_set_dict[two_qubit_gate_set.casefold()]
        if len(two_qubit_gate_set) == 1:
            two_qubit_gate = two_qubit_gate_set[0]

        if rng is None:
            rng = default_rng()

        if isinstance(rng, int):
            rng = default_rng(rng)

        num_qubits = len(qubits)
        # if circuit has one qubit, call random_clifford_circuits()
        if num_qubits == 1:
            return self.random_one_qubit_tensor_products(
                num_qubits, size=size, gate_set=one_qubit_gate_set, rng=rng
            )

        qc_list = []
        for _ in list(range(size)):
            all_edges = coupling_map[:]  # make copy of coupling map from which we pop edges
            selected_edges = []
            while all_edges:
                rand_edge = all_edges.pop(rng.integers(len(all_edges)))
                selected_edges.append(
                    rand_edge
                )  # move random edge from all_edges to selected_edges
                old_all_edges = all_edges[:]
                all_edges = []
                # only keep edges in all_edges that do not share a vertex with rand_edge
                for edge in old_all_edges:
                    if rand_edge[0] not in edge and rand_edge[1] not in edge:
                        all_edges.append(edge)

            qr = QuantumRegister(num_qubits)
            qc = QuantumCircuit(qr)
            two_qubit_prob = 0
            try:
                two_qubit_prob = num_qubits * two_qubit_gate_density / len(selected_edges)
            except ZeroDivisionError:
                warnings.warn(
                    "Device has no connectivity. All cliffords will be single-qubit Cliffords"
                )
            if two_qubit_prob > 1:
                warnings.warn(
                    "Mean number of two-qubit gates is higher than number of selected edges for CNOTs. "
                    + "Actual density of two-qubit gates will likely be lower than input density"
                )
            selected_edges_logical = [
                [np.where(q == np.asarray(qubits))[0][0] for q in edge] for edge in selected_edges
            ]
            # selected_edges_logical is selected_edges with logical qubit labels rather than physical
            # ones. Example: qubits = (8,4,5,3,7), selected_edges = [[4,8],[7,5]]
            # ==> selected_edges_logical = [[1,0],[4,2]]
            put_1_qubit_clifford = np.arange(num_qubits)
            # put_1_qubit_clifford is a list of qubits that aren't assigned to a 2-qubit Clifford
            # 1-qubit Clifford will be assigned to these edges if separate_one_qubit_layer is True
            for edge in selected_edges_logical:
                if rng.random() < two_qubit_prob:
                    # with probability two_qubit_prob, place uniformly random two-qubit gate on edge
                    # in selected_edges
                    if len(two_qubit_gate_set) > 1:
                        two_qubit_gate = two_qubit_gate_set[rng.choice(len(two_qubit_gate_set))]
                    qc.append(two_qubit_gate, [edge[0], edge[1]], [])
                    # remove these qubits from put_1_qubit_clifford
                    put_1_qubit_clifford = np.setdiff1d(put_1_qubit_clifford, edge)
            if not separate_one_qubit_layer:
                for q in put_1_qubit_clifford:
                    clifford1q = self._clifford_utils.clifford_1_qubit_circuit(rng.integers(24))
                    insts = [datum[0] for datum in clifford1q.data]
                    for inst in insts:
                        qc.compose(inst, [q], inplace=True)
            qc_list.append(qc)
            if separate_one_qubit_layer:
                qc_1q = self.random_one_qubit_tensor_products(
                    num_qubits, size=1, gate_set=one_qubit_gate_set, rng=rng
                )
                qc_list += qc_1q
        return qc_list

    def should_invert_cp(self, pcontrol: str, ptarget: str, axis2q: str) -> bool:
        """Returns Boolean, whether CP(theta) or CP(-theta) should be added to T
        (see arXiv:2207.07272), p. 4, col. 2, 3(b).
        Note: The cited protocol is currently incorrect as of 08/08/22, but the one here is correct.

        Args:
            pcontrol: Pauli on the control bit
            ptarget: Pauli on the target bit
            axis2q: Pauli axis about which the two-qubit gate, CP, rotates
                    (e.g., if CP = Sdg, axis2q = Z)

        Returns:
            True, if CP(-theta) should be in T, and False, if CP(theta) should be in T

        Raises:
            QiskitError: if pcontrol is not a Pauli"""

        pt_pauli = Pauli(data=ptarget.upper())
        axis2q_pauli = Pauli(data=axis2q.upper())
        if pcontrol.casefold() in ["i", "z"]:
            return not pt_pauli.commutes(axis2q_pauli)
        elif pcontrol.casefold() in ["x", "y"]:
            return pt_pauli.commutes(axis2q_pauli)
        else:
            raise QiskitError("pc is not a Pauli")

    def commute_paulis_through_cp(self, pauli1: str, pauli2: str, cp: str) -> Tuple[np.array, Gate]:
        """Compute the Pauli-axis rotations that should be placed after a controlled Pauli-axis
        rotation cp to correct for pauli1 and pauli2. Specifically, the gates Pc and Pt returned
        by this method satisfy

                  ┌───┐     ┌───┐┌───┐
        ──■──     ┤ P1├──■──┤ Pc├┤ P1├
        ┌─┴─┐  =  ├───┤┌─┴─┐├───┤├───┤
        ┤ P ├     ┤ P2├┤ P ├┤ Pt├┤ P2├
        └───┘     └───┘└───┘└───┘└───┘

        Args:
            pauli1: Pauli before control bit
            pauli2: Pauli before target bit
            cp: controlled Pauli-axis rotation gate

        Raises:
            QiskitError: If controlled pauli axis rotation is invalid

        Returns:
            control_gate: Pauli-axis rotation after control bit
            target_gate: Pauli-axis rotation after target bit
        """
        p1_pauli = Pauli(data=pauli1.upper())
        p2_pauli = Pauli(data=pauli2.upper())

        # Get cp_matrix and rotation axis from cp
        if cp[0].casefold() == "c":
            cp = cp[1:]
        if cp.casefold() in ["x", "y", "z"]:
            cp_matrix = np.asmatrix(Pauli(data=cp.upper()).to_matrix())
            axis2q = cp.upper()
        elif cp.casefold() == "s":
            cp_matrix = np.asmatrix([[1.0, 0], [0, 1.0j]])
            axis2q = "Z"
        elif cp.casefold() == "sdg":
            cp_matrix = np.asmatrix([[1.0, 0], [0, -1.0j]])
            axis2q = "Z"
        elif cp.casefold() == "sx":
            cp_matrix = 0.5 * np.asmatrix([[1.0 + 1.0j, 1.0 - 1.0j], [1.0 - 1.0j, 1.0 + 1.0j]])
            axis2q = "X"
        elif cp.casefold() == "sxdg":
            cp_matrix = 0.5 * np.asmatrix([[1.0 - 1.0j, 1.0 + 1.0j], [1.0 + 1.0j, 1.0 - 1.0j]])
            axis2q = "X"
        else:
            raise QiskitError(
                "Your cp has not yet been implemented or is not a controlled Pauli-axis rotation gate"
            )

        should_invert = self.should_invert_cp(pauli1.upper(), pauli2.upper(), axis2q)
        cp_gate_dict = {
            "x": CXGate(),
            "y": CYGate(),
            "z": CZGate,
            "s": CSGate,
            "sdg": CSdgGate,
            "sx": CSXGate,
            "sxdg": CSXdgGate,
        }
        if cp.casefold() in ["x", "y", "z"]:
            new_cp_gate = cp_gate_dict[cp]
        elif cp.casefold() == "s":
            new_cp_gate = cp_gate_dict["sdg"] if should_invert else cp_gate_dict["s"]
        elif cp.casefold() == "sdg":
            new_cp_gate = cp_gate_dict["s"] if should_invert else cp_gate_dict["sdg"]
        elif cp.casefold() == "sx":
            new_cp_gate = cp_gate_dict["sxdg"] if should_invert else cp_gate_dict["sx"]
        else:  # sxdg
            new_cp_gate = cp_gate_dict["sx"] if should_invert else cp_gate_dict["sxdg"]

        if pauli1.casefold() in ["i", "z"]:
            # Compute the right-hand side of (10) in arXiv:2207.07272,
            #
            # (P1 ⊗ P2) (|0><0| ⊗ I + |1><1| ⊗ P^α) (P1 ⊗ P2) (|0><0| ⊗ I + |1><1| ⊗ P^±α) (*).
            # If P1 = I or Z, (*) reduces to
            #
            #  P1|0><0|P1|0><0| ⊗ I + P1|1><1|P1|1> ⊗ P2 P3^α P2 P3^±α.
            # └────────────────┘     └─────────────┘ └─────────────────┘
            #       ↳coeff0           ↳coeff1/phase       ↳product
            coeff0 = p1_pauli.to_matrix()[0, 0] * np.outer(
                p1_pauli.to_matrix()[:, 0], np.asarray([1, 0])
            )
            if self.should_invert_cp(pauli1, pauli2, axis2q):  # cp(-theta) goes in T
                product = p2_pauli.to_matrix() @ cp_matrix @ p2_pauli.to_matrix() @ cp_matrix
            else:  # cp(theta) goes in T
                product = p2_pauli.to_matrix() @ cp_matrix @ p2_pauli.to_matrix() @ cp_matrix.H
            # product should equal phase*I, so get phase and multiply it in coeff1
            phase = product[0, 0]
            coeff1 = (
                phase
                * p1_pauli.to_matrix()[1, 1]
                * np.outer(p1_pauli.to_matrix()[:, 1], np.asarray([0, 1]))
            )
        else:
            # If P1 = X or Y, (*) reduces to
            #
            #  P1|0><0|P1|1><1| ⊗ P3^±α + P1|1><1|P1|0><0| ⊗ P2 P3^α P2.
            # └────────────────┘         └────────────────┘ └───────────┘
            #       ↳coeff0                ↳coeff1/phase      ↳product
            coeff0 = p1_pauli.to_matrix()[0, 1] * np.outer(
                p1_pauli.to_matrix()[:, 0], np.asarray([0, 1])
            )
            product = p2_pauli.to_matrix() @ cp_matrix @ p2_pauli.to_matrix()
            # product should equal phase*P3^±α, so get phase and multiply it in coeff1
            if cp == "x":
                phase = product[1, 0]
            elif cp == "y":
                phase = product[1, 0] * 1.0j
            elif cp in ["z", "s", "sdg"]:
                phase = product[0, 0]
            elif cp == "sxdg":
                if self.should_invert_cp(pauli1.upper(), pauli2.upper(), axis2q):
                    phase = product[0, 0] * 2.0 / (1.0 - 1.0j)
                else:
                    phase = product[0, 0] * 2.0 / (1.0 + 1.0j)
            elif cp == "sx":
                if self.should_invert_cp(pauli1.upper(), pauli2.upper(), axis2q):
                    phase = product[0, 0] * 2.0 / (1.0 + 1.0j)
                else:
                    phase = product[0, 0] * 2.0 / (1.0 - 1.0j)
            else:
                raise QiskitError(
                    "Your cp has not yet been implemented or is not a controlled "
                    "Pauli-axis rotation gate"
                )
            coeff1 = (
                phase
                * p1_pauli.to_matrix()[1, 0]
                * np.outer(p1_pauli.to_matrix()[:, 1], np.asarray([1, 0]))
            )

        control_gate = coeff0 + coeff1
        target_gate_dict = {
            "x": XGate(),
            "y": YGate(),
            "z": ZGate(),
            "s": SGate(),
            "sdg": SdgGate(),
            "sx": SXGate(),
            "sxdg": SXdgGate(),
        }
        if pauli1.casefold() in ["i", "z"]:
            target_gate = IGate()
        else:
            if cp[0] == "s" and not should_invert:  # not pure controlled Pauli
                if cp[-1] == "g":  # dagger
                    target_gate = target_gate_dict[cp[:-2].casefold()]
                else:
                    target_gate = target_gate_dict[cp.casefold() + "dg"]
            else:
                target_gate = target_gate_dict[cp.casefold()]

        return control_gate, target_gate, new_cp_gate

    def which_pauli_power(self, matrix: np.array) -> Tuple[Gate, float]:
        """Converts a matrix to a power of a Pauli and a global phase

        Args:
            matrix: matrix to be converted to a gate (up to a global phase)

        Raises:
            QiskitError: If matrix does not represent a Pauli or the controlled
                         Pauli rotation has not yet been implemented

        Returns:
            Tuple of gate and a float representing the global phase
        """
        gate, phase = None, None
        if (
            np.abs(matrix[0, 0]) == 1.0
            and matrix[1, 1] != 0
            and matrix[1, 0] == 0
            and matrix[0, 1] == 0
        ):
            ratio = matrix[1, 1] / matrix[0, 0]
            if ratio == 1.0:
                gate, phase = IGate(), matrix[0, 0]
            elif ratio == -1.0:
                gate, phase = ZGate(), matrix[0, 0]
            elif ratio == 1.0j:
                gate, phase = SGate(), matrix[0, 0]
            elif ratio == -1.0j:
                gate, phase = SdgGate(), matrix[0, 0]
            else:
                raise QiskitError("not a Pauli or has not been implemented yet")
        elif (
            np.abs(matrix[1, 0]) == 1.0
            and matrix[0, 1] != 0
            and matrix[0, 0] == 0
            and matrix[1, 1] == 0
        ):
            ratio = matrix[1, 0] / matrix[0, 1]
            if ratio == 1.0:
                gate, phase = XGate(), matrix[1, 0]
            elif ratio == -1.0:
                gate, phase = YGate(), matrix[1, 0] * -1.0j
            else:
                raise QiskitError("not a Pauli or has not been implemented yet")
        elif matrix[0, 0] * matrix[0, 1] * matrix[1, 0] * matrix[1, 1] != 0:
            ratio14 = matrix[1, 1] / matrix[0, 0]
            ratio23 = matrix[1, 0] / matrix[0, 1]
            ratio12 = matrix[0, 1] / matrix[0, 0]
            if ratio14 != 1.0 or ratio23 != 1.0 or ratio12 not in [1.0j, -1.0j]:
                raise QiskitError("not a Pauli or has not been implemented yet")
            if ratio12 == 1.0j:
                gate, phase = SXGate(), matrix[0, 0] * 2.0 / (1.0 + 1.0j)
            else:
                gate, phase = SXdgGate(), matrix[0, 0] * 2.0 / (1.0 - 1.0j)
        else:
            raise QiskitError("not a Pauli or has not been implemented yet")
        return gate, phase

    def generate_correction(
        self,
        pauli_layer: QuantumCircuit,
        two_qubit_layer: QuantumCircuit,
        two_qubit_gate_names: List[str],
    ) -> Tuple[QuantumCircuit, QuantumCircuit]:
        """Generates layers of one-qubit correction gates that undo the effect of pauli_layer

        Args:
            pauli_layer: a QuantumCircuit with a single layer of Paulis
            two_qubit_layer: a QuantumCircuit with a single layer of controlled Pauli-axis rotations
            two_qubit_gate_names: names of gates in the two qubit gate set

        Returns:
            a QuantumCircuit with a single layer of gates that undo the effect of pauli_layer
        """
        pauli_dag = circuit_to_dag(pauli_layer)
        two_qubit_dag = circuit_to_dag(two_qubit_layer)

        control_qubits = []
        target_qubits = []
        two_qubit_gate_location = [0] * pauli_layer.num_qubits
        paulis_by_qubit = ["I"] * pauli_layer.num_qubits
        for node in two_qubit_dag.op_nodes():
            if node.name in two_qubit_gate_names:
                control_qubit = node.qargs[0].index
                target_qubit = node.qargs[1].index
                control_qubits.append(control_qubit)
                target_qubits.append(target_qubit)
                two_qubit_gate_location[control_qubit] = node.name

        for node in pauli_dag.op_nodes():
            paulis_by_qubit[node.qargs[0].index] = node.name

        two_qubit_gate_pairs = {}
        for i, _ in enumerate(control_qubits):
            control_qubit = control_qubits[i]
            target_qubit = target_qubits[i]
            two_qubit_gate_pairs[(control_qubit, target_qubit)] = (
                paulis_by_qubit[control_qubit],
                paulis_by_qubit[target_qubit],
                two_qubit_gate_location[control_qubit],
            )
        # entries are of the form {(control qubit, target_qubit):
        # [Pauli on control, Pauli on target, new_cp_gate, phase]}

        commuted_gates = {}
        for control, target in two_qubit_gate_pairs.items():
            p1, p2, cp = two_qubit_gate_pairs[control, target]
            p_matrix, p_char, new_cp_gate = self.commute_paulis_through_cp(p1, p2, cp)
            matrix_to_char, phase = self.which_pauli_power(p_matrix)
            commuted_gates[control, target] = [matrix_to_char, p_char, new_cp_gate, phase]

        rotation_correction = QuantumCircuit(pauli_layer.num_qubits)
        rc_data = []
        new_two_qubit_layer = QuantumCircuit(pauli_layer.num_qubits)
        ntql_data = []
        for qubits in commuted_gates:
            gate_list = commuted_gates[qubits]
            rc_data.append((gate_list[0], [qubits[0]], []))
            rc_data.append((gate_list[1], [qubits[1]], []))
            ntql_data.append((gate_list[2], [qubits[0], qubits[1]], []))

        rotation_correction.data = rc_data
        new_two_qubit_layer.data = ntql_data

        return new_two_qubit_layer, rotation_correction
