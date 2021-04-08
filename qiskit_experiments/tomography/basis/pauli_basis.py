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
Pauli basis tomography preparation and measurement circuits
"""
import numpy as np
from qiskit.circuit import QuantumCircuit
from qiskit.circuit.library import HGate, XGate, ZGate, SGate, SdgGate
from .basis import TomographyBasis, CircuitBasis, FitterBasis


class PauliMeasurementCircuitBasis(CircuitBasis):
    """A Pauli measurement basis"""

    # Z-meas rotation
    _meas_z = QuantumCircuit(1, name="PauliMeasZ")
    # X-meas rotation
    _meas_x = QuantumCircuit(1, name="PauliMeasX")
    _meas_x.append(HGate(), [0])
    # Y-meas rotation
    _meas_y = QuantumCircuit(1, name="PauliMeasY")
    _meas_y.append(SdgGate(), [0])
    _meas_y.append(HGate(), [0])

    def __init__(self):
        super().__init__(
            [self._meas_z, self._meas_x, self._meas_y], num_outcomes=2, name="PauliMeas"
        )


class PauliMeasurementFitterBasis(FitterBasis):
    """A Pauli PVM measurement basis"""

    def __init__(self):
        povms = np.array(
            [
                [[1, 0], [0, 0]],
                [[0, 0], [0, 1]],
                [[0.5, 0.5], [0.5, 0.5]],
                [[0.5, -0.5], [-0.5, 0.5]],
                [[0.5, -0.5j], [0.5j, 0.5]],
                [[0.5, 0.5j], [-0.5j, 0.5]],
            ]
        )
        super().__init__(povms, num_outcomes=2, name=type(self).__name__)


class PauliMeasurementBasis(TomographyBasis):
    """Pauli measurement tomography basis"""

    def __init__(self):
        """Pauli measurement tomography basis"""
        super().__init__(
            PauliMeasurementCircuitBasis(), PauliMeasurementFitterBasis(), name=type(self).__name__
        )


class PauliPreparationCircuitBasis(CircuitBasis):
    """A Pauli measurement basis"""

    def __init__(self):
        # |0> Zp rotation
        prep_zp = QuantumCircuit(1, name="PauliPrepZp")
        # |1> Zm rotation
        prep_zm = QuantumCircuit(1, name="PauliPrepZp")
        prep_zm.append(XGate(), [0])
        # |+> Xp rotation
        prep_xp = QuantumCircuit(1, name="PauliPrepXp")
        prep_xp.append(HGate(), [0])
        # |+i> Yp rotation
        prep_yp = QuantumCircuit(1, name="PauliPrepYp")
        prep_yp.append(HGate(), [0])
        prep_yp.append(SGate(), [0])
        super().__init__([prep_zp, prep_zm, prep_xp, prep_yp], name="PauliPrep")


class PauliPreparationFitterBasis(FitterBasis):
    """Minimum 4-element Pauli preparation basis"""

    def __init__(self):
        povms = np.array(
            [
                [[1, 0], [0, 0]],
                [[0, 0], [0, 1]],
                [[0.5, 0.5], [0.5, 0.5]],
                [[0.5, -0.5j], [0.5j, 0.5]],
            ]
        )
        super().__init__(povms, name="PauliPrep")


class PauliPreparationBasis(TomographyBasis):
    """Minimum 4-element Pauli preparation basis"""

    def __init__(self):
        """Pauli measurement tomography basis"""
        super().__init__(
            PauliPreparationCircuitBasis(), PauliPreparationFitterBasis(), type(self).__name__
        )


class Pauli6PreparationCircuitBasis(CircuitBasis):
    """A 6-element Pauli preparation basis"""

    def __init__(self):
        # |0> Zp rotation
        prep_zp = QuantumCircuit(1, name="PauliPrepZp")
        # |1> Zm rotation
        prep_zm = QuantumCircuit(1, name="PauliPrepZp")
        prep_zm.append(XGate(), [0])
        # |+> Xp rotation
        prep_xp = QuantumCircuit(1, name="PauliPrepXp")
        prep_xp.append(HGate(), [0])
        # |+i> Yp rotation
        prep_yp = QuantumCircuit(1, name="PauliPrepYp")
        prep_yp.append(HGate(), [0])
        prep_yp.append(SGate(), [0])
        # |-> Xm rotation
        prep_xm = QuantumCircuit(1, name="PauliPrepXp")
        prep_xm.append(HGate(), [0])
        prep_xm.append(ZGate(), [0])
        # |-i> Ym rotation
        prep_ym = QuantumCircuit(1, name="PauliPrepYp")
        prep_ym.append(HGate(), [0])
        prep_ym.append(SdgGate(), [0])
        super().__init__([prep_zp, prep_zm, prep_xp, prep_yp])


class Pauli6PreparationFitterBasis(FitterBasis):
    """A 6-element Pauli preparation basis"""

    def __init__(self):
        povms = np.array(
            [
                [[1, 0], [0, 0]],
                [[0, 0], [0, 1]],
                [[0.5, 0.5], [0.5, 0.5]],
                [[0.5, -0.5], [-0.5, 0.5]],
                [[0.5, -0.5j], [0.5j, 0.5]],
                [[0.5, 0.5j], [-0.5j, 0.5]],
            ]
        )
        super().__init__(povms, name=type(self).__name__)


class Pauli6PreparationBasis(TomographyBasis):
    """Pauli-6 preparation tomography basis"""

    def __init__(self):
        """Pauli-6 preparation tomography basis"""
        super().__init__(
            Pauli6PreparationCircuitBasis(), Pauli6PreparationFitterBasis(), "Pauli6Prep"
        )
