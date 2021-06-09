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
from warnings import warn
from collections import OrderedDict
import numpy as np
from qiskit import QuantumCircuit, QiskitError
from qiskit.qobj import QasmQobj

class RBUtils():
    @staticmethod
    def get_1_qubit_error_dict_from_backend(backend, qubits):
        error_dict = {qubit: {} for qubit in qubits}
        gates_list = []
        for g in backend.properties().gates:
            g = g.to_dict()
            if len(g['qubits']) == 1:
                gate_qubit = g['qubits'][0]
                if gate_qubit in qubits:
                    for p in g['parameters']:
                        if p['name'] == 'gate_error':
                            error_dict[gate_qubit][g['gate']] = p['value']
                            if g['gate'] not in gates_list:
                                gates_list.append(g['gate'])
        return error_dict, gates_list

    @staticmethod
    def count_ops(circuit, qubits=None):
        if qubits is None:
            qubits = range(len(circuit.qubits))
        count_ops_per_qubit = {qubit: {} for qubit in circuit.qubits}
        for instr, qargs, _ in circuit._data:
            for qubit in qargs:
                count_ops_per_qubit[qubit][instr.name] = count_ops_per_qubit[qubit].get(instr.name, 0) + 1
        result = {circuit.qubits.index(qubit): count_ops
                  for qubit, count_ops in count_ops_per_qubit.items()
                  if circuit.qubits.index(qubit) in qubits}
        return result

    @staticmethod
    def gates_per_clifford(ops_count, gates, qubits):
        result = {}
        for qubit in qubits:
            result[qubit] = {}
            for gate in gates:
                result[qubit][gate] = []
        for c in ops_count:
            for qubit, counts in c.items():
                for gate, value in counts.items():
                    if qubit in qubits and gate in gates:
                        result[qubit][gate].append(value)
        found_gates = {}
        for qubit in qubits:
            found_gates[qubit] = []
            for gate in gates:
                if len(result[qubit][gate]) > 0:
                    found_gates[qubit].append(gate)
                    result[qubit][gate] = np.mean(result[qubit][gate])
        return result, found_gates


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
                         count_ops
                         ) -> Dict[int, Dict[str, float]]:
        r"""
        Convert error per Clifford (EPC) into error per gates (EPGs) of single qubit basis gates.
        """

        error_dict, gates_1_qubit = RBUtils.get_1_qubit_error_dict_from_backend(
            backend,
            qubits)
        gates_per_clifford, found_gates = RBUtils.gates_per_clifford(count_ops,
                                                                     gates_1_qubit,
                                                                     qubits)
        epg = {qubit: {} for qubit in qubits}
        for qubit in qubits:
            error_sum = sum([gates_per_clifford[qubit][gate] * error_dict[qubit][gate]
                             for gate in found_gates[qubit]])
            for gate in found_gates[qubit]:
                epg[qubit][gate] = (error_dict[qubit][gate] * epc_1_qubit) / error_sum
        return epg

    @staticmethod
    def calculate_2q_epg(gate_per_cliff: Dict[int, Dict[str, float]],
                         epc_2q: float,
                         qubit_pair: List[int],
                         list_epgs_1q: Optional[List[Dict[str, float]]] = None,
                         two_qubit_name: Optional[str] = 'cx') -> float:
        r"""
        Convert error per Clifford (EPC) into error per gate (EPG) of two qubit ``cx`` gates.
        Given that a standard 2Q RB sequences consist of ``rz``, ``x``, ``sx``, and ``cx`` gates,
        the EPG of ``cx`` gate can be roughly approximated by :math:`EPG_{CX} = EPC/N_{CX}`,
        where :math:`N_{CX}` is number of ``cx`` gates per Clifford which is designed to be 1.5.
        Because an error from two qubit gates are usually dominant and the contribution of
        single qubit gates in 2Q RB experiments is thus able to be ignored.
        If ``list_epgs_1q`` is not provided, the function returns
        the EPG calculated based upon this assumption.
        When we know the EPG of every single qubit gates used in the 2Q RB experiment,
        we can isolate the EPC of the two qubit gate, ie :math:`EPG_{CX} = EPC_{CX}/N_{CX}` [1].
        This will give you more accurate estimation of EPG, especially when the ``cx``
        gate fidelity is close to that of single qubit gate.
        To evaluate EPGs of single qubit gates, you first need to run standard 1Q RB experiments
        separately and feed the fit result and gate counts to :func:`calculate_1q_epg`.
        .. jupyter-execute::
            import qiskit.ignis.verification.randomized_benchmarking as rb
            # assuming we ran 1Q RB experiment for qubit 0 and qubit 1
            gpc = {0: {'cx': 0, 'rz': 0.13, 'x': 0.31, 'sx': 0.51},
                   1: {'cx': 0, 'rz': 0.10, 'x': 0.33, 'sx': 0.51}}
            epc_q0 = 1.5e-3 TODO not accutrate for new gateset
            epc_q1 = 5.8e-4 TODO not accutrate for new gateset
            # calculate 1Q EPGs
            epgs_q0 = rb.rb_utils.calculate_1q_epg(gate_per_cliff=gpc, epc_1q=epc_q0, qubit=0)
            epgs_q1 = rb.rb_utils.calculate_1q_epg(gate_per_cliff=gpc, epc_1q=epc_q1, qubit=1)
            # assuming we ran 2Q RB experiment for qubit 0 and qubit 1
            gpc = {0: {'cx': 1.49, 'rz': 0.25, 'x': 0.95, 'sx': 0.56},
                   1: {'cx': 1.49, 'rz': 0.24, 'x': 0.98, 'sx': 0.49}}
            epc = 2.4e-2 TODO not accutrate for new gateset
            # calculate 2Q EPG
            epg_no_comp = rb.rb_utils.calculate_2q_epg(
                gate_per_cliff=gpc,
                epc_2q=epc,
                qubit_pair=[0, 1])
            epg_comp = rb.rb_utils.calculate_2q_epg(
                gate_per_cliff=gpc,
                epc_2q=epc,
                qubit_pair=[0, 1],
                list_epgs_1q=[epgs_q0, epgs_q1])
            print('EPG without `list_epgs_1q`: %f, with `list_epgs_1q`: %f' % (epg_no_comp, epg_comp))
        Note:
            This function presupposes the basis gate consists
            of ``rz``, ``x``, ``sx`` and ``cx``.
        References:
            [1] D. C. McKay, S. Sheldon, J. A. Smolin, J. M. Chow,
            and J. M. Gambetta, “Three-Qubit Randomized Benchmarking,”
            Phys. Rev. Lett., vol. 122, no. 20, 2019 (arxiv:1712.06550).
        Args:
            gate_per_cliff: dictionary of gate per Clifford. see :func:`gates_per_clifford`.
            epc_2q: EPC fit from 2Q RB experiment data.
            qubit_pair: index of two qubits to calculate EPG.
            list_epgs_1q: list of single qubit EPGs of qubit listed in ``qubit_pair``.
            two_qubit_name: name of two qubit gate in ``basis gates``.
        Returns:
            EPG of 2Q gate.
        Raises:
            QiskitError: when ``cx`` is not found, specified ``qubit_pair`` is not included
                in the gate count dictionary, or length of ``qubit_pair`` is not 2.
        """
        list_epgs_1q = list_epgs_1q or []

        if len(qubit_pair) != 2:
            raise QiskitError('Number of qubit is not 2.')

        # estimate single qubit gate error contribution
        alpha_1q = [1.0, 1.0]
        for ind, (qubit, epg_1q) in enumerate(zip(qubit_pair, list_epgs_1q)):
            if qubit not in gate_per_cliff:
                raise QiskitError('Qubit %d is not included in the `gate_per_cliff`' % qubit)
            gpc_per_qubit = gate_per_cliff[qubit]
            for gate_name, epg in epg_1q.items():
                n_gate = gpc_per_qubit.get(gate_name, 0)
                alpha_1q[ind] *= (1 - 2 * epg) ** n_gate
        alpha_c_1q = 1 / 5 * (alpha_1q[0] + alpha_1q[1] + 3 * alpha_1q[0] * alpha_1q[1])
        alpha_c_2q = (1 - 4 / 3 * epc_2q) / alpha_c_1q

        n_gate_2q = gate_per_cliff[qubit_pair[0]].get(two_qubit_name, 0)

        if n_gate_2q > 0:
            return 3 / 4 * (1 - alpha_c_2q) / n_gate_2q

        raise QiskitError('Two qubit gate %s is not included in the `gate_per_cliff`. '
                          'Set correct `two_qubit_name` or use 2Q RB gate count.' % two_qubit_name)

    @staticmethod
    def calculate_1q_epc(gate_per_cliff: Dict[int, Dict[str, float]],
                         epg_1q: Dict[str, float],
                         qubit: int) -> float:
        r"""
        Convert error per gate (EPG) into error per Clifford (EPC) of single qubit basis gates.
        Given that we know the number of gates per Clifford :math:`N_i` and those EPGs,
        we can predict EPC of that RB sequence:
        .. math::
            EPC = 1 - \prod_i \left( 1 - EPG_i \right)^{N_i}
        To run this function, you need to know EPG of every single qubit basis gates.
        For example, when you prepare 1Q RB experiment with appropriate error model,
        you can define EPG of those basis gate set. Then you can estimate the EPC of
        prepared RB sequence without running experiment.
        .. jupyter-execute::
            import qiskit.ignis.verification.randomized_benchmarking as rb
            # gate counts of your 1Q RB experiment
            gpc = {0: {'cx': 0, 'rZ': 0.13, 'x': 0.31, 'sx': 0.51}}
            # EPGs from error model
            epgs_q0 = {'rz': 0, 'x': 0.001, 'sx': 0.001}
            # calculate 1Q EPC
            epc = rb.rb_utils.calculate_1q_epc(
                gate_per_cliff=gpc,
                epg_1q=epgs_q0,
                qubit=0)
            print(epc)
        Args:
            gate_per_cliff: dictionary of gate per Clifford. see :func:`gates_per_clifford`.
            epg_1q: EPG of single qubit gates estimated by error model.
            qubit: index of qubit to calculate EPC.
        Returns:
            EPG of 2Q gate.
        Raises:
            QiskitError: when specified ``qubit`` is not included in the gate count dictionary
        """
        if qubit not in gate_per_cliff:
            raise QiskitError('Qubit %d is not included in the `gate_per_cliff`' % qubit)

        fid = 1
        gpc_per_qubit = gate_per_cliff[qubit]

        for gate_name, epg in epg_1q.items():
            n_gate = gpc_per_qubit.get(gate_name, 0)
            fid *= (1 - epg) ** n_gate

        return 1 - fid

    @staticmethod
    def calculate_2q_epc(gate_per_cliff: Dict[int, Dict[str, float]],
                         epg_2q: float,
                         qubit_pair: List[int],
                         list_epgs_1q: List[Dict[str, float]],
                         two_qubit_name: Optional[str] = 'cx') -> float:
        r"""
        Convert error per gate (EPG) into error per Clifford (EPC) of two qubit ``cx`` gates.
        Given that we know the number of gates per Clifford :math:`N_i` and those EPGs,
        we can predict EPC of that RB sequence:
        .. math::
            EPC = 1 - \prod_i \left( 1 - EPG_i \right)^{N_i}
        This function isolates the contribution of two qubit gate to the EPC [1].
        This will give you more accurate estimation of EPC, especially when the ``cx``
        gate fidelity is close to that of single qubit gate.
        To run this function, you need to know EPG of both single and two qubit gates.
        For example, when you prepare 2Q RB experiment with appropriate error model,
        you can define EPG of those basis gate set. Then you can estimate the EPC of
        prepared RB sequence without running experiment.
        .. jupyter-execute::
            import qiskit.ignis.verification.randomized_benchmarking as rb
            # gate counts of your 2Q RB experiment
            gpc = {0: {'cx': 1.49, 'rz': 0.25, 'x': 0.95, 'sx': 0.56},
                   1: {'cx': 1.49, 'rz': 0.24, 'x': 0.98, 'sx': 0.49}}
            # EPGs from error model
            epgs_q0 = {'rz': 0, 'x': 0.001, 'sx': 0.001}
            epgs_q1 = {'rz': 0, 'x': 0.002, 'sx': 0.002}
            epg_q01 = 0.03 TODO not accutrate for new gateset
            # calculate 2Q EPC
            epc_2q = rb.rb_utils.calculate_2q_epc(
                gate_per_cliff=gpc,
                epg_2q=epg_q01,
                qubit_pair=[0, 1],
                list_epgs_1q=[epgs_q0, epgs_q1])
            # calculate EPC according to the definition
            fid = 1
            for qubit in (0, 1):
                for epgs in (epgs_q0, epgs_q1):
                    for gate, val in epgs.items():
                        fid *= (1 - val) ** gpc[qubit][gate]
            fid *= (1 - epg_q01) ** 1.49
            epc = 1 - fid
            print('Total sequence EPC: %f, 2Q gate contribution: %f' % (epc, epc_2q))
        As you can see two qubit gate contribution is dominant in this RB sequence.
        References:
            [1] D. C. McKay, S. Sheldon, J. A. Smolin, J. M. Chow,
            and J. M. Gambetta, “Three-Qubit Randomized Benchmarking,”
            Phys. Rev. Lett., vol. 122, no. 20, 2019 (arxiv:1712.06550).
        Args:
            gate_per_cliff: dictionary of gate per Clifford. see :func:`gates_per_clifford`.
            epg_2q: EPG estimated by error model.
            qubit_pair: index of two qubits to calculate EPC.
            list_epgs_1q: list of single qubit EPGs of qubit listed in ``qubit_pair``.
            two_qubit_name: name of two qubit gate in ``basis gates``.
        Returns:
            EPG of 2Q gate.
        Raises:
            QiskitError: when ``cx`` is not found, specified ``qubit_pair`` is not included
                in the gate count dictionary, or length of ``qubit_pair`` is not 2.
        """
        if len(qubit_pair) != 2:
            raise QiskitError('Number of qubit is not 2.')

        n_gate_2q = gate_per_cliff[qubit_pair[0]].get(two_qubit_name, 0)
        if n_gate_2q == 0:
            raise QiskitError('Two qubit gate %s is not included in the `gate_per_cliff`. '
                              'Set correct `two_qubit_name` or use 2Q RB gate count.' % two_qubit_name)

        # estimate single qubit gate error contribution
        alpha_1q = [1.0, 1.0]
        alpha_2q = (1 - 4 / 3 * epg_2q) ** n_gate_2q
        for ind, (qubit, epg_1q) in enumerate(zip(qubit_pair, list_epgs_1q)):
            if qubit not in gate_per_cliff:
                raise QiskitError('Qubit %d is not included in the `gate_per_cliff`' % qubit)
            gpc_per_qubit = gate_per_cliff[qubit]
            for gate_name, epg in epg_1q.items():
                n_gate = gpc_per_qubit.get(gate_name, 0)
                alpha_1q[ind] *= (1 - 2 * epg) ** n_gate
        alpha_c_2q = 1 / 5 * (alpha_1q[0] + alpha_1q[1] + 3 * alpha_1q[0] * alpha_1q[1]) * alpha_2q

        return 3 / 4 * (1 - alpha_c_2q)