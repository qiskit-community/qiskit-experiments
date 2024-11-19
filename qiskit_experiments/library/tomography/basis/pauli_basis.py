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
from typing import Optional
import numpy as np
from qiskit.circuit import QuantumCircuit
from qiskit.circuit.library import HGate, XGate, ZGate, SGate, SdgGate
from qiskit.quantum_info import DensityMatrix
from qiskit.exceptions import QiskitError
from qiskit_experiments.data_processing import LocalReadoutMitigator
from .local_basis import LocalMeasurementBasis, LocalPreparationBasis


class PauliMeasurementBasis(LocalMeasurementBasis):
    r"""Standard Pauli measurement basis.

    This basis has 3 indices each with with 2 measurement outcomes. The
    corresponding single-qubit measurement circuits and outcome POVM
    matrices are:

    .. table:: Single-qubit measurement circuits and POVM matrices

        +-------+-------+---------------+---------+-------------------------------+
        | Index | Basis | Circuit       | Outcome | POVM Matrix                   |
        +=======+=======+===============+=========+===============================+
        | 0     | Z     |``-[I]-``      | 0       |``[[1, 0], [0, 0]]``           |
        +-------+-------+---------------+---------+-------------------------------+
        |       |       |               | 1       |``[[0, 0], [0, 1]]``           |
        +-------+-------+---------------+---------+-------------------------------+
        | 1     | X     | ``-[H]-``     | 0       |``[[0.5, 0.5], [0.5, 0.5]]``   |
        +-------+-------+---------------+---------+-------------------------------+
        |       |       |               | 1       |``[[0.5, -0.5], [-0.5, 0.5]]`` |
        +-------+-------+---------------+---------+-------------------------------+
        | 2     | Y     |``-[SDG]-[H]-``| 0       |``[[0.5, -0.5j], [0.5j, 0.5]]``|
        +-------+-------+---------------+---------+-------------------------------+
        |       |       |               | 1       |``[[0.5, 0.5j], [-0.5j, 0.5]]``|
        +-------+-------+---------------+---------+-------------------------------+

    """

    def __init__(self, mitigator: Optional[LocalReadoutMitigator] = None):
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
        instructions = [meas_z, meas_x, meas_y]
        self._mitigator = mitigator

        super().__init__(
            "PauliMeasurementBasis", instructions, qubit_povms=self._mitigator_povm(self._mitigator)
        )

    @staticmethod
    def _mitigator_povm(mitigator: Optional[LocalReadoutMitigator] = None):
        """Construct LocalMeasurementBasis qubit_povm from mitigator."""
        if mitigator is None:
            return None
        povm = {}
        if isinstance(mitigator, LocalReadoutMitigator):
            povm = {}
            for qubit in mitigator.qubits:
                amat = mitigator.assignment_matrix(qubit)
                povm[(qubit,)] = {(0,): [DensityMatrix(np.diag(row)) for row in amat]}
        else:
            raise QiskitError("Invalid mitigator: must be LocalReadoutMitigator")
        return povm

    def __json_encode__(self):
        # Override LocalMeasurementBasis's encoder
        if self._mitigator is not None:
            return {"mitigator": self._mitigator}
        return {}


class PauliPreparationBasis(LocalPreparationBasis):
    """Minimal 4-element Pauli measurement basis.

    This is a minimal size 4 preparation basis where each qubit
    index corresponds to the following initial state preparation
    circuits and density matrices:

    .. table:: Single-qubit preparation circuits and states

        +-------+-------+---------------------+---------------------------------+
        | Index | State | Preparation Circuit | Density Matrix                  |
        +=======+=======+=====================+=================================+
        | 0     | Zp    | ``-[I]-``           | ``[[1, 0], [0, 0]]``            |
        +-------+-------+---------------------+---------------------------------+
        | 1     | Zm    | ``-[X]-``           | ``[[0, 0], [0, 1]]``            |
        +-------+-------+---------------------+---------------------------------+
        | 2     | Xp    | ``-[H]-``           | ``[[0.5, 0.5], [0.5, 0.5]]``    |
        +-------+-------+---------------------+---------------------------------+
        | 3     | Yp    | ``-[H]-[S]-``       | ``[[0.5, -0.5j], [0.5j, 0.5]]`` |
        +-------+-------+---------------------+---------------------------------+

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
        super().__init__("PauliPreparationBasis", [prep_zp, prep_zm, prep_xp, prep_yp])

    def __json_encode__(self):
        # Override LocalPreparationBasis's encoder
        return {}


class Pauli6PreparationBasis(LocalPreparationBasis):
    """Over-complete 6-element Pauli preparation basis.

    This is an over-complete size 6 preparation basis where each qubit
    index corresponds to the following initial state density matrices:

    .. table:: Single-qubit preparation circuits and states

        +-------+-------+---------------------+---------------------------------+
        | Index | State | Preparation Circuit | Density Matrix                  |
        +=======+=======+=====================+=================================+
        | 0     | Zp    | ``-[I]-``           | ``[[1, 0], [0, 0]]``            |
        +-------+-------+---------------------+---------------------------------+
        | 1     | Zm    | ``-[X]-``           | ``[[0, 0], [0, 1]]``            |
        +-------+-------+---------------------+---------------------------------+
        | 2     | Xp    | ``-[H]-``           | ``[[0.5, 0.5], [0.5, 0.5]]``    |
        +-------+-------+---------------------+---------------------------------+
        | 3     | Xm    | ``-[H]-[Z]-``       | ``[[0.5, -0.5], [-0.5, 0.5]]``  |
        +-------+-------+---------------------+---------------------------------+
        | 2     | Yp    | ``-[H]-[S]-``       | ``[[0.5, -0.5j], [0.5j, 0.5]]`` |
        +-------+-------+---------------------+---------------------------------+
        | 3     | Ym    | ``-[H]-[Sdg]-``     | ``[[0.5, 0.5j], [-0.5j, 0.5]]`` |
        +-------+-------+---------------------+---------------------------------+

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
        # |-> Xm rotation
        prep_xm = QuantumCircuit(1, name="PauliPrepXm")
        prep_xm.append(HGate(), [0])
        prep_xm.append(ZGate(), [0])
        # |+i> Yp rotation
        prep_yp = QuantumCircuit(1, name="PauliPrepYp")
        prep_yp.append(HGate(), [0])
        prep_yp.append(SGate(), [0])
        # |-i> Ym rotation
        prep_ym = QuantumCircuit(1, name="PauliPrepYm")
        prep_ym.append(HGate(), [0])
        prep_ym.append(SdgGate(), [0])
        super().__init__(
            "Pauli6PreparationBasis",
            [
                prep_zp,
                prep_zm,
                prep_xp,
                prep_xm,
                prep_yp,
                prep_ym,
            ],
        )

    def __json_encode__(self):
        # Override LocalPreparationBasis's encoder
        return {}
