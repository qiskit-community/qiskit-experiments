# This code is part of Qiskit.
#
# (C) Copyright IBM 2019-2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
RB Helper functions
"""
from typing import Tuple, Dict, Optional, List, Union, Sequence

import numpy as np
import uncertainties
from qiskit import QiskitError, QuantumCircuit
from qiskit.providers.backend import Backend

from qiskit_experiments.database_service.device_component import Qubit
from qiskit_experiments.framework import AnalysisResultData
from qiskit_experiments.warnings import deprecated_function


class RBUtils:
    """A collection of utility functions for computing additional data
    from randomized benchmarking experiments"""

    @staticmethod
    @deprecated_function(
        last_version="0.4",
        msg=(
            "This method may return erroneous error ratio. "
            "Please directly provide known gate error ratio to the analysis option."
        ),
    )
    def get_error_dict_from_backend(
        backend: Backend, qubits: Sequence[int]
    ) -> Dict[Tuple[Sequence[int], str], float]:
        """Attempts to extract error estimates for gates from the backend
        properties.
        Those estimates are used to assign weights for different gate types
        when computing error per gate.

        Args:
            backend: The backend from which the properties are taken
            qubits: The qubits participating in the experiment, used
              to filter irrelevant gates from the result.

        Returns:
            A dictionary of the form (qubits, gate) -> value that for each
            gate on the given qubits gives its recorded error estimate.
        """
        error_dict = {}
        try:
            for backend_gate in backend.properties().gates:
                backend_gate = backend_gate.to_dict()
                gate_qubits = tuple(backend_gate["qubits"])
                if all(gate_qubit in qubits for gate_qubit in gate_qubits):
                    for p in backend_gate["parameters"]:
                        if p["name"] == "gate_error":
                            error_dict[(gate_qubits, backend_gate["gate"])] = p["value"]
        except AttributeError:
            # might happen if the backend has no properties (e.g. qasm simulator)
            return None
        return error_dict

    @staticmethod
    @deprecated_function(
        last_version="0.4",
        msg=(
            "Now this method is integrated into 'StandardRB._transpiled_circuits' method. "
            "You don't need to explicitly call this method."
        ),
    )
    def count_ops(
        circuit: QuantumCircuit, qubits: Optional[Sequence[int]] = None
    ) -> Dict[Tuple[Sequence[int], str], int]:
        """Counts occurrences of each gate in the given circuit

        Args:
            circuit: The quantum circuit whose gates are counted
            qubits: A list of qubits to filter irrelevant gates

        Returns:
            A dictionary of the form (qubits, gate) -> value where value
            is the number of occurrences of the gate on the given qubits
        """
        if qubits is None:
            qubits = range(len(circuit.qubits))
        count_ops_result = {}
        for instr, qargs, _ in circuit:
            instr_qubits = []
            skip_instr = False
            for qubit in qargs:
                qubit_index = circuit.qubits.index(qubit)
                if qubit_index not in qubits:
                    skip_instr = True
                instr_qubits.append(qubit_index)
            if not skip_instr:
                instr_qubits = tuple(instr_qubits)
                count_ops_result[(instr_qubits, instr.name)] = (
                    count_ops_result.get((instr_qubits, instr.name), 0) + 1
                )
        return count_ops_result

    @staticmethod
    @deprecated_function(last_version="0.4")
    def gates_per_clifford(
        ops_count: List,
    ) -> Dict[Tuple[Sequence[int], str], float]:
        """
        Computes the average number of gates per clifford for each gate type
        in the input from raw count data coming from multiple circuits.

        Args:
            ops_count: A List of [key, value] pairs where
              key is [qubits, gate_name] and value is the average
              number of gates per clifford of the type for the given key

        Returns:
            A dictionary with the mean value of values corresponding
            to key for each key.

        """
        result = {}
        for ((qubits, gate_name), value) in ops_count:
            qubits = tuple(qubits)  # so we can hash
            # for each (qubit, gate name) pair collect
            # (number of gates, number of cliffords)
            if (qubits, gate_name) not in result:
                result[(qubits, gate_name)] = [0, 0]
            result[(qubits, gate_name)][0] += value[0]
            result[(qubits, gate_name)][1] += value[1]
        return {key: value[0] / value[1] for (key, value) in result.items()}

    @staticmethod
    def coherence_limit(nQ=2, T1_list=None, T2_list=None, gatelen=0.1):
        """
        The error per gate (1-average_gate_fidelity) given by the T1,T2 limit.

        Args:
            nQ (int): Number of qubits (1 and 2 supported).
            T1_list (list): List of T1's (Q1,...,Qn).
            T2_list (list): List of T2's (as measured, not Tphi). If not given assume T2=2*T1 .
            gatelen (float): Length of the gate.

        Returns:
            float: coherence limited error per gate.
        Raises:
            ValueError: if there are invalid inputs
        """
        # pylint: disable = invalid-name

        T1 = np.array(T1_list)

        if T2_list is None:
            T2 = 2 * T1
        else:
            T2 = np.array(T2_list)

        if len(T1) != nQ or len(T2) != nQ:
            raise ValueError("T1 and/or T2 not the right length")

        coherence_limit_err = 0

        if nQ == 1:

            coherence_limit_err = 0.5 * (
                1.0 - 2.0 / 3.0 * np.exp(-gatelen / T2[0]) - 1.0 / 3.0 * np.exp(-gatelen / T1[0])
            )

        elif nQ == 2:

            T1factor = 0
            T2factor = 0

            for i in range(2):
                T1factor += 1.0 / 15.0 * np.exp(-gatelen / T1[i])
                T2factor += (
                    2.0
                    / 15.0
                    * (
                        np.exp(-gatelen / T2[i])
                        + np.exp(-gatelen * (1.0 / T2[i] + 1.0 / T1[1 - i]))
                    )
                )

            T1factor += 1.0 / 15.0 * np.exp(-gatelen * np.sum(1 / T1))
            T2factor += 4.0 / 15.0 * np.exp(-gatelen * np.sum(1 / T2))

            coherence_limit_err = 0.75 * (1.0 - T1factor - T2factor)

        else:
            raise ValueError("Not a valid number of qubits")

        return coherence_limit_err

    @staticmethod
    @deprecated_function(
        last_version="0.4",
        msg="Please use calculate_epg function instead. This works regardless of qubit number.",
    )
    def calculate_1q_epg(
        epc_1_qubit: Union[float, uncertainties.UFloat],
        qubits: Sequence[int],
        gate_error_ratio: Dict[str, float],
        gates_per_clifford: Dict[Tuple[Sequence[int], str], float],
    ) -> Dict[int, Dict[str, uncertainties.UFloat]]:
        r"""
        Convert error per Clifford (EPC) into error per gates (EPGs) of single qubit basis gates.

        Args:
            epc_1_qubit: The error per clifford rate obtained via experiment
            qubits: The qubits for which to compute epg
            gate_error_ratio: Estimate for the ratios between errors on different gates
            gates_per_clifford: The computed gates per clifford data
        Returns:
            A dictionary of the form (qubits, gate) -> value where value
            is the EPG for the given gate on the specified qubits
        """
        out = {(qubit,): {} for qubit in qubits}
        for qubit in qubits:
            error_sum = 0
            found_gates = []
            for (key, value) in gate_error_ratio.items():
                qubits, gate = key
                if len(qubits) == 1 and qubits[0] == qubit and key in gates_per_clifford:
                    found_gates.append(gate)
                    error_sum += gates_per_clifford[key] * value
            for gate in found_gates:
                epg = (gate_error_ratio[((qubit,), gate)] * epc_1_qubit) / error_sum
                out[(qubit,)][gate] = epg
        return out

    @staticmethod
    @deprecated_function(
        last_version="0.4",
        msg="Please use calculate_epg function instead. This works regardless of qubit number.",
    )
    def calculate_2q_epg(
        epc_2_qubit: Union[uncertainties.UFloat, float],
        qubits: Sequence[int],
        gate_error_ratio: Dict[str, float],
        gates_per_clifford: Dict[Tuple[Sequence[int], str], float],
        epg_1_qubit: Optional[List[AnalysisResultData]] = None,
        gate_2_qubit_type: Optional[str] = "cx",
    ) -> Dict[int, Dict[str, uncertainties.UFloat]]:
        r"""
        Convert error per Clifford (EPC) into error per gates (EPGs) of two-qubit basis gates.
        Assumes a single two-qubit gate type is used in transpilation

        Args:
            epc_2_qubit: The error per clifford rate obtained via experiment
            qubits: The qubits for which to compute epg
            gate_error_ratio: Estimate for the ratios between errors on different gates
            gates_per_clifford: The computed gates per clifford data
            epg_1_qubit: analysis results containing EPG for the 1-qubits gate involved,
                assumed to have been obtained from previous experiments
            gate_2_qubit_type: The name of the 2-qubit gate to be analyzed

        Returns:
            The EPG value for the specified gate on the specified qubits
            given in a dictionary form as in calculate_1q_epg

        Raises:
            QiskitError: if a non 2-qubit gate was given
        """
        out = {}
        qubit_pairs = []

        # Extract 1-qubit epgs
        epg_1_qubit_dict = {}
        if epg_1_qubit is not None:
            for result in epg_1_qubit:
                if result.name.startswith("EPG") and len(result.device_components) == 1:
                    qubit = result.device_components[0]
                    if isinstance(qubit, Qubit):
                        qubit = qubit._index
                    if not qubit in epg_1_qubit_dict:
                        epg_1_qubit_dict[qubit] = {}
                    gate = result.name.replace("EPG_", "")

                    # This keeps variance of previous experiment to propagate
                    epg_1_qubit_dict[qubit][gate] = result.value

        for key in gate_error_ratio:
            qubits, gate = key
            if gate == gate_2_qubit_type and key in gates_per_clifford:
                if len(qubits) != 2:
                    raise QiskitError(
                        "The gate {} is a {}-qubit gate (should be 2-qubit)".format(
                            gate, len(qubits)
                        )
                    )
                qubit_pair = tuple(sorted(qubits))
                if qubit_pair not in qubit_pairs:
                    qubit_pairs.append(qubit_pair)
        for qubit_pair in qubit_pairs:
            alpha_1q = [1.0, 1.0]
            if epg_1_qubit_dict:
                list_epgs_1q = [epg_1_qubit_dict[qubit_pair[i]] for i in range(2)]
                for ind, (qubit, epg_1q) in enumerate(zip(qubit_pair, list_epgs_1q)):
                    for gate_name, epg in epg_1q.items():
                        n_gate = gates_per_clifford.get(((qubit,), gate_name), 0)
                        alpha_1q[ind] *= (1 - 2 * epg) ** n_gate
            alpha_c_1q = 1 / 5 * (alpha_1q[0] + alpha_1q[1] + 3 * alpha_1q[0] * alpha_1q[1])
            alpha_c_2q = (1 - 4 / 3 * epc_2_qubit) / alpha_c_1q
            inverse_qubit_pair = (qubit_pair[1], qubit_pair[0])
            n_gate_2q = gates_per_clifford.get(
                (qubit_pair, gate_2_qubit_type), 0
            ) + gates_per_clifford.get((inverse_qubit_pair, gate_2_qubit_type), 0)

            out[qubit_pair] = {gate_2_qubit_type: 3 / 4 * (1 - alpha_c_2q) / n_gate_2q}

        return out
