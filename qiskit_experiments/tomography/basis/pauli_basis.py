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
        """Initialize Pauli measurement basis"""
        super().__init__([self._meas_z, self._meas_x, self._meas_y])


class PauliPreparationBasis(TomographyPreparationBasis):
    """A Pauli measurement basis"""

    # |0> Zp rotation
    _prep_zp = QuantumCircuit(1, name="PauliPrepZp")
    # |1> Zm rotation
    _prep_zm = QuantumCircuit(1, name="PauliPrepZm")
    _prep_zm.append(XGate(), [0])
    # |+> Xp rotation
    _prep_xp = QuantumCircuit(1, name="PauliPrepXp")
    _prep_xp.append(HGate(), [0])
    # |+i> Yp rotation
    _prep_yp = QuantumCircuit(1, name="PauliPrepYp")
    _prep_yp.append(HGate(), [0])
    _prep_yp.append(SGate(), [0])

    def __init__(self):
        super().__init__([self._prep_zp, self._prep_zm, self._prep_xp, self._prep_yp])


class Pauli6PreparationBasis(TomographyPreparationBasis):
    """A 6-element Pauli preparation basis"""

    # |0> Zp rotation
    _prep_zp = QuantumCircuit(1, name="PauliPrepZp")
    # |1> Zm rotation
    _prep_zm = QuantumCircuit(1, name="PauliPrepZm")
    _prep_zm.append(XGate(), [0])
    # |+> Xp rotation
    _prep_xp = QuantumCircuit(1, name="PauliPrepXp")
    _prep_xp.append(HGate(), [0])
    # |+i> Yp rotation
    _prep_yp = QuantumCircuit(1, name="PauliPrepYp")
    _prep_yp.append(HGate(), [0])
    _prep_yp.append(SGate(), [0])
    # |-> Xm rotation
    _prep_xm = QuantumCircuit(1, name="PauliPrepXm")
    _prep_xm.append(HGate(), [0])
    _prep_xm.append(ZGate(), [0])
    # |-i> Ym rotation
    _prep_ym = QuantumCircuit(1, name="PauliPrepYm")
    _prep_ym.append(HGate(), [0])
    _prep_ym.append(SdgGate(), [0])

    def __init__(self):
        super().__init__(
            [
                self._prep_zp,
                self._prep_zm,
                self._prep_xp,
                self._prep_xm,
                self._prep_yp,
                self._prep_ym,
            ]
        )
