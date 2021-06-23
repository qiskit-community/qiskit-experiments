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
Pauli preparation and measurement tomography bases.
"""

from qiskit.circuit import QuantumCircuit
from qiskit.circuit.library import HGate, XGate, ZGate, SGate, SdgGate
from .tomography_basis import TomographyMeasurementBasis, TomographyPreparationBasis


class PauliMeasurementBasis(TomographyMeasurementBasis):
    r"""Standard Pauli measurement basis.

    This basis has 3 indices with corresponding measurement configurations

    .. list-table:: Single-qubit measurement basis circuits

        * - Index
          - Measurement Basis
          - Circuit
        * - 0
          - Pauli-Z
          - ``-[I]-``
        * - 1
          - Pauli-X
          - ``-[H]-``
        * - 2
          - Pauli-Y
          - ``-[SDG]-[H]-``

    The POVM matrices for each index and outcome are

    .. list-table:: Single-qubit basis outcome elements
        :header-rows: 1

        * - Index
          - Outcome
          - Element
          - Matrix
        * - 0
          - 0
          - :math:`| 0 \rangle\!\langle 0 |`
          - :math:`\begin{pmatrix} 1 & 0 \\ 0 & 0 \end{pmatrix}`
        * -
          - 1
          - :math:`| 1 \rangle\!\langle 1 |`
          - :math:`\begin{pmatrix} 0 & 0 \\ 0 & 1 \end{pmatrix}`
        * - 1
          - 0
          - :math:`| + \rangle\!\langle + |`
          - :math:`\begin{pmatrix} 0.5 & 0.5 \\ 0.5 & 0.5 \end{pmatrix}`
        * -
          - 1
          - :math:`| - \rangle\!\langle - |`
          - :math:`\begin{pmatrix} 0.5 & -0.5 \\ -0.5 & 0.5 \end{pmatrix}`
        * - 2
          - 0
          - :math:`| +i \rangle\!\langle +i |`
          - :math:`\begin{pmatrix} 0.5 & -0.5i \\ 0.5i & 0.5 \end{pmatrix}`
        * -
          - 1
          - :math:`| -i \rangle\!\langle -i |`
          - :math:`\begin{pmatrix} 0.5 & 0.5i \\ -0.5i & 0.5 \end{pmatrix}`

    """

    def __init__(self):
        """Initialize Pauli measurement basis"""
        # Z-meas rotation
        meas_z = QuantumCircuit(1, name="PauliMeasZ")
        # X-meas rotation
        meas_x = QuantumCircuit(1, name="PauliMeasX")
        meas_x.append(HGate(), [0])
        # Y-meas rotation
        meas_y = QuantumCircuit(1, name="PauliMeasY")
        meas_y.append(SdgGate(), [0])
        meas_y.append(HGate(), [0])
        super().__init__([meas_z, meas_x, meas_y])


class PauliPreparationBasis(TomographyPreparationBasis):
    r"""Minimal 4-element Pauli measurement basis.

    This is a minimal size 4 preparation basis where each qubit
    index corresponds to the following initial state density matrices

    .. list-table:: Single-qubit basis elements
        :header-rows: 1

        * - Index
          - State
          - Matrix
          - Circuit
        * - 0
          - :math:`| 0 \rangle\!\langle 0 |`
          - :math:`\begin{pmatrix} 1 & 0 \\ 0 & 0 \end{pmatrix}`
          - ``-[I]-``
        * - 1
          - :math:`| 1 \rangle\!\langle 1 |`
          - :math:`\begin{pmatrix} 0 & 0 \\ 0 & 1 \end{pmatrix}`
          - ``-[X]-``
        * - 2
          - :math:`| + \rangle\!\langle + |`
          - :math:`\begin{pmatrix} 0.5 & 0.5 \\ 0.5 & 0.5 \end{pmatrix}`
          - ``-[H]-``
        * - 3
          - :math:`| +i \rangle\!\langle +i |`
          - :math:`\begin{pmatrix} 0.5 & -0.5i \\ 0.5i & 0.5 \end{pmatrix}`
          - ``-[H]-[S]-``
    """

    def __init__(self):
        """Initialize Pauli preparation basis"""
        # |0> Zp rotation
        prep_zp = QuantumCircuit(1, name="PauliPrepZp")
        # |1> Zm rotation
        prep_zm = QuantumCircuit(1, name="PauliPrepZm")
        prep_zm.append(XGate(), [0])
        # |+> Xp rotation
        prep_xp = QuantumCircuit(1, name="PauliPrepXp")
        prep_xp.append(HGate(), [0])
        # |+i> Yp rotation
        prep_yp = QuantumCircuit(1, name="PauliPrepYp")
        prep_yp.append(HGate(), [0])
        prep_yp.append(SGate(), [0])
        super().__init__([prep_zp, prep_zm, prep_xp, prep_yp])


class Pauli6PreparationBasis(TomographyPreparationBasis):
    r"""Over-complete 6-element Pauli preparation basis.

    This is an over-complete size 6 preparation basis where each qubit
    index corresponds to the following initial state density matrices

    .. list-table:: Single-qubit basis elements
        :header-rows: 1

        * - Index
          - State
          - Matrix
          - Circuit
        * - 0
          - :math:`| 0 \rangle\!\langle 0 |`
          - :math:`\begin{pmatrix} 1 & 0 \\ 0 & 0 \end{pmatrix}`
          - ``-[I]-``
        * - 1
          - :math:`| 1 \rangle\!\langle 1 |`
          - :math:`\begin{pmatrix} 0 & 0 \\ 0 & 1 \end{pmatrix}`
          - ``-[X]-``
        * - 2
          - :math:`| + \rangle\!\langle + |`
          - :math:`\begin{pmatrix} 0.5 & 0.5 \\ 0.5 & 0.5 \end{pmatrix}`
          - ``-[H]-``
        * - 3
          - :math:`| - \rangle\!\langle - |`
          - :math:`\begin{pmatrix} 0.5 & -0.5 \\ -0.5 & 0.5 \end{pmatrix}`
          - ``-[H]-[Z]-``
        * - 4
          - :math:`| +i \rangle\!\langle +i |`
          - :math:`\begin{pmatrix} 0.5 & -0.5i \\ 0.5i & 0.5 \end{pmatrix}`
          - ``-[H]-[S]-``
        * - 5
          - :math:`| -i \rangle\!\langle -i |`
          - :math:`\begin{pmatrix} 0.5 & 0.5i \\ -0.5i & 0.5 \end{pmatrix}`
          - ``-[H]-[SDG]-``
    """

    def __init__(self):
        # |0> Zp rotation
        prep_zp = QuantumCircuit(1, name="PauliPrepZp")
        # |1> Zm rotation
        prep_zm = QuantumCircuit(1, name="PauliPrepZm")
        prep_zm.append(XGate(), [0])
        # |+> Xp rotation
        prep_xp = QuantumCircuit(1, name="PauliPrepXp")
        prep_xp.append(HGate(), [0])
        # |+i> Yp rotation
        prep_yp = QuantumCircuit(1, name="PauliPrepYp")
        prep_yp.append(HGate(), [0])
        prep_yp.append(SGate(), [0])
        # |-> Xm rotation
        prep_xm = QuantumCircuit(1, name="PauliPrepXm")
        prep_xm.append(HGate(), [0])
        prep_xm.append(ZGate(), [0])
        # |-i> Ym rotation
        prep_ym = QuantumCircuit(1, name="PauliPrepYm")
        prep_ym.append(HGate(), [0])
        prep_ym.append(SdgGate(), [0])
        super().__init__(
            [
                prep_zp,
                prep_zm,
                prep_xp,
                prep_xm,
                prep_yp,
                prep_ym,
            ]
        )
