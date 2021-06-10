# -*- coding: utf-8 -*-

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

from typing import List, Union, Dict, Optional
import numpy as np
from qiskit import QuantumCircuit, QiskitError

class RBUtils():
    @staticmethod
    def get_error_dict_from_backend(backend, qubits):
        error_dict = {}
        for g in backend.properties().gates:
            g = g.to_dict()
            gate_qubits = tuple(g['qubits'])
            if all([gate_qubit in qubits for gate_qubit in gate_qubits]):
                for p in g['parameters']:
                    if p['name'] == 'gate_error':
                        error_dict[(gate_qubits, g['gate'])] = p['value']
        return error_dict

    @staticmethod
    def count_ops(circuit, qubits=None):
        if qubits is None:
            qubits = range(len(circuit.qubits))
        count_ops_result = {}
        for instr, qargs, _ in circuit._data:
            instr_qubits = []
            skip_instr = False
            for qubit in qargs:
                qubit_index = circuit.qubits.index(qubit)
                if qubit_index not in qubits:
                    skip_instr = True
                instr_qubits.append(qubit_index)
            if not skip_instr:
                instr_qubits = tuple(instr_qubits)
                count_ops_result[(instr_qubits, instr.name)] = count_ops_result.get((instr_qubits, instr.name), 0) + 1
        return count_ops_result

    @staticmethod
    def is_qubit_subset(qubit_subset, qubits):
        for q in qubit_subset:
            if not q in qubits:
                return False
        return True

    @staticmethod
    def gates_per_clifford(ops_count):
        # ops_count is of the form [[qubits, gate_name], value]
        result = {}
        for ((qubits, gate_name), value) in ops_count:
            qubits = tuple(qubits) # so we can hash
            if (qubits, gate_name) not in result:
                result[(qubits, gate_name)] = []
            result[(qubits, gate_name)].append(value)
        return {key: np.mean(value) for (key, value) in result.items()}


    @staticmethod
    def coherence_limit(nQ=2, T1_list=None, T2_list=None,
                        gatelen=0.1):
        """
        The error per gate (1-average_gate_fidelity) given by the T1,T2 limit.
        Args:
            nQ (int): number of qubits (1 and 2 supported).
            T1_list (list): list of T1's (Q1,...,Qn).
            T2_list (list): list of T2's (as measured, not Tphi).
                If not given assume T2=2*T1 .
            gatelen (float): length of the gate.
        Returns:
            float: coherence limited error per gate.
        Raises:
            ValueError: if there are invalid inputs
        """

        T1 = np.array(T1_list)

        if T2_list is None:
            T2 = 2*T1
        else:
            T2 = np.array(T2_list)

        if len(T1) != nQ or len(T2) != nQ:
            raise ValueError("T1 and/or T2 not the right length")

        coherence_limit_err = 0

        if nQ == 1:

            coherence_limit_err = 0.5*(1.-2./3.*np.exp(-gatelen/T2[0]) -
                                       1./3.*np.exp(-gatelen/T1[0]))

        elif nQ == 2:

            T1factor = 0
            T2factor = 0

            for i in range(2):
                T1factor += 1./15.*np.exp(-gatelen/T1[i])
                T2factor += 2./15.*(np.exp(-gatelen/T2[i]) +
                                    np.exp(-gatelen*(1./T2[i]+1./T1[1-i])))

            T1factor += 1./15.*np.exp(-gatelen*np.sum(1/T1))
            T2factor += 4./15.*np.exp(-gatelen*np.sum(1/T2))

            coherence_limit_err = 0.75*(1.-T1factor-T2factor)

        else:
            raise ValueError('Not a valid number of qubits')

        return coherence_limit_err

    @staticmethod
    def calculate_1q_epg(epc_1_qubit,
                         qubits,
                         backend,
                         gates_per_clifford
                         ) -> Dict[int, Dict[str, float]]:
        r"""
        Convert error per Clifford (EPC) into error per gates (EPGs) of single qubit basis gates.
        """
        error_dict = RBUtils.get_error_dict_from_backend(backend, qubits)
        epg = {qubit: {} for qubit in qubits}
        for qubit in qubits:
            error_sum = 0
            found_gates = []
            for (key, value) in error_dict.items():
                qubits, gate = key
                if len(qubits) == 1 and qubits[0] == qubit and key in gates_per_clifford:
                    found_gates.append(gate)
                    error_sum += gates_per_clifford[key] * value
            for gate in found_gates:
                epg[qubit][gate] = (error_dict[((qubit,), gate)] * epc_1_qubit) / error_sum
        return epg

    @staticmethod
    def calculate_2q_epg(epc_2_qubit,
                         qubits,
                         backend,
                         gates_per_clifford,
                         epg_1_qubit = None,
                         gate_2_qubit_type='cx',
                         ) -> Dict[int, Dict[str, float]]:
        r"""
        Convert error per Clifford (EPC) into error per gates (EPGs) of two-qubit basis gates.
        Assumed a single two-qubit gate type is used in transpilation
        """
        epg_2_qubit = {}
        error_dict = error_dict = RBUtils.get_error_dict_from_backend(backend, qubits)
        qubit_pairs = []
        for key in error_dict.keys():
            qubits, gate = key
            if gate == gate_2_qubit_type and key in gates_per_clifford:
                if len(qubits) != 2:
                    raise QiskitError("The gate {} is a {}-qubit gate (should be 2-qubit)".format(gate, len(qubits)))
                qubit_pair = tuple(sorted(qubits))
                if not qubit_pair in qubit_pairs:
                    qubit_pairs.append(qubit_pair)
        for qubit_pair in qubit_pairs:
            alpha_1q = [1.0, 1.0]
            if epg_1_qubit is not None:
                list_epgs_1q = [epg_1_qubit[qubit_pair[i]] for i in range(2)]
                for ind, (qubit, epg_1q) in enumerate(zip(qubit_pair, list_epgs_1q)):
                    for gate_name, epg in epg_1q.items():
                        n_gate = gates_per_clifford.get(((qubit,),gate_name), 0)
                        alpha_1q[ind] *= (1 - 2 * epg) ** n_gate
            alpha_c_1q = 1 / 5 * (alpha_1q[0] + alpha_1q[1] + 3 * alpha_1q[0] * alpha_1q[1])
            alpha_c_2q = (1 - 4 / 3 * epc_2_qubit) / alpha_c_1q
            n_gate_2q = gates_per_clifford[(qubit_pair,gate_2_qubit_type)]
            epg = 3 / 4 * (1 - alpha_c_2q) / n_gate_2q
            epg_2_qubit[qubit_pair] = {gate_2_qubit_type: epg}
        return epg_2_qubit