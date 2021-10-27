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
HEAT experiment elements for ZX Hamiltonian.
"""

import numpy as np
from qiskit import circuit, QuantumCircuit

from qiskit_experiments.library.heat.base_experiment import BaseHeatElement


class HeatElementPrepIIMeasIY(BaseHeatElement):
    r"""A single error amplification sequence of Y error with the control qubit in 0 state.

    # section: overview

        This experiment generates a following circuit.

        .. parsed-literal::

                             ░ ┌─────┐      ░
            q_0: ────────────░─┤0    ├──────░────
                 ┌─────────┐ ░ │  cr │┌───┐ ░ ┌─┐
            q_1: ┤ Ry(π/2) ├─░─┤1    ├┤ Y ├─░─┤M├
                 └─────────┘ ░ └─────┘└───┘ ░ └╥┘
            c: 1/══════════════════════════════╩═
                                               0

        Circuit block in the middle is repeated N times to amplify the target error.
        The ``cr`` gate represents a unitary of :math:`ZX(\pi/2)`, and its pulse schedule
        should be provided by users.

    """
    def _prep_circuit(self) -> QuantumCircuit:
        circ = circuit.QuantumCircuit(2)
        circ.ry(np.pi/2, 1)

        return circ

    def _echo_circuit(self) -> QuantumCircuit:
        circ = circuit.QuantumCircuit(2)
        circ.y(1)

        return circ


class HeatElementPrepXIMeasIY(BaseHeatElement):
    r"""A single error amplification sequence of Y error with the control qubit in 1 state.

    # section: overview

        This experiment generates a following circuit.

        .. parsed-literal::

                    ┌───┐    ░ ┌─────┐      ░
            q_0: ───┤ X ├────░─┤0    ├──────░────
                 ┌──┴───┴──┐ ░ │  cr │┌───┐ ░ ┌─┐
            q_1: ┤ Ry(π/2) ├─░─┤1    ├┤ Y ├─░─┤M├
                 └─────────┘ ░ └─────┘└───┘ ░ └╥┘
            c: 1/══════════════════════════════╩═
                                               0

        Circuit block in the middle is repeated N times to amplify the target error.
        The ``cr`` gate represents a unitary of :math:`ZX(\pi/2)`, and its pulse schedule
        should be provided by users.

    """

    def _prep_circuit(self) -> QuantumCircuit:
        circ = circuit.QuantumCircuit(2)
        circ.x(0)
        circ.ry(np.pi/2, 1)

        return circ

    def _echo_circuit(self) -> QuantumCircuit:
        circ = circuit.QuantumCircuit(2)
        circ.y(1)

        return circ


class HeatElementPrepIIMeasIZ(BaseHeatElement):
    r"""A single error amplification sequence of Z error with the control qubit in 0 state.

    # section: overview

        This experiment generates a following circuit.

        .. parsed-literal::

                             ░ ┌─────┐      ░
            q_0: ────────────░─┤0    ├──────░───────────────
                 ┌─────────┐ ░ │  cr │┌───┐ ░ ┌─────────┐┌─┐
            q_1: ┤ Ry(π/2) ├─░─┤1    ├┤ Z ├─░─┤ Rx(π/2) ├┤M├
                 └─────────┘ ░ └─────┘└───┘ ░ └─────────┘└╥┘
            c: 1/═════════════════════════════════════════╩═
                                                          0

        Circuit block in the middle is repeated N times to amplify the target error.
        The ``cr`` gate represents a unitary of :math:`ZX(\pi/2)`, and its pulse schedule
        should be provided by users.

    """

    def _prep_circuit(self) -> QuantumCircuit:
        circ = circuit.QuantumCircuit(2)
        circ.ry(np.pi/2, 1)

        return circ

    def _meas_circuit(self) -> QuantumCircuit:
        circ = circuit.QuantumCircuit(2)
        circ.rx(np.pi/2, 1)

        return circ

    def _echo_circuit(self) -> QuantumCircuit:
        circ = circuit.QuantumCircuit(2)
        circ.z(1)

        return circ


class HeatElementPrepXIMeasIZ(BaseHeatElement):
    r"""A single error amplification sequence of Z error with the control qubit in 1 state.

    # section: overview

        This experiment generates a following circuit.

        .. parsed-literal::

                    ┌───┐    ░ ┌─────┐      ░
            q_0: ───┤ X ├────░─┤0    ├──────░───────────────
                 ┌──┴───┴──┐ ░ │  cr │┌───┐ ░ ┌─────────┐┌─┐
            q_1: ┤ Ry(π/2) ├─░─┤1    ├┤ Z ├─░─┤ Rx(π/2) ├┤M├
                 └─────────┘ ░ └─────┘└───┘ ░ └─────────┘└╥┘
            c: 1/═════════════════════════════════════════╩═
                                                          0

        Circuit block in the middle is repeated N times to amplify the target error.
        The ``cr`` gate represents a unitary of :math:`ZX(\pi/2)`, and its pulse schedule
        should be provided by users.

    """

    def _prep_circuit(self) -> QuantumCircuit:
        circ = circuit.QuantumCircuit(2)
        circ.x(0)
        circ.ry(np.pi / 2, 1)

        return circ

    def _meas_circuit(self) -> QuantumCircuit:
        circ = circuit.QuantumCircuit(2)
        circ.rx(np.pi / 2, 1)

        return circ

    def _echo_circuit(self) -> QuantumCircuit:
        circ = circuit.QuantumCircuit(2)
        circ.z(1)

        return circ
