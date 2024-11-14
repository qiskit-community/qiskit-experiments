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
Common methods for tomography tests
"""
import numpy as np
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister
from qiskit_aer.noise import NoiseModel


FITTERS = [
    None,
    "linear_inversion",
    "cvxpy_linear_lstsq",
    "cvxpy_gaussian_lstsq",
]


def filter_results(analysis_results, name):
    """Filter list of analysis results by result name"""
    for result in analysis_results:
        if result.name == name:
            return result
    return None


def teleport_circuit(flatten_creg=True):
    """Teleport qubit 0 to qubit 2"""
    if flatten_creg:
        teleport = QuantumCircuit(3, 2)
        creg = teleport.cregs[0]
    else:
        qr = QuantumRegister(3)
        c0 = ClassicalRegister(1, "c0")
        c1 = ClassicalRegister(1, "c1")
        teleport = QuantumCircuit(qr, c0, c1)
        creg = [c0, c1]
    teleport.h(1)
    teleport.cx(1, 2)
    teleport.cx(0, 1)
    teleport.h(0)
    teleport.measure(0, creg[0])
    teleport.measure(1, creg[1])
    # Conditionals
    with teleport.if_test((creg[0], True)):
        teleport.z(2)
    with teleport.if_test((creg[1], True)):
        teleport.x(2)
    return teleport


def teleport_bell_circuit(flatten_creg=True):
    """Teleport entangled qubit 0 -> 2"""
    if flatten_creg:
        teleport = QuantumCircuit(4, 2)
        creg = teleport.cregs[0]
    else:
        qr = QuantumRegister(4)
        c0 = ClassicalRegister(1)
        c1 = ClassicalRegister(1)
        teleport = QuantumCircuit(qr, c0, c1)
        creg = [c0, c1]
    teleport.h(0)
    teleport.cx(0, 3)
    teleport.h(1)
    teleport.cx(1, 2)
    teleport.cx(0, 1)
    teleport.h(0)
    teleport.measure(0, creg[0])
    teleport.measure(1, creg[1])
    with teleport.if_test((creg[0], True)):
        teleport.z(2)
    with teleport.if_test((creg[1], True)):
        teleport.x(2)
    return teleport


def readout_noise_model(num_qubits, seed=None):
    """Generate noise model of random local readout errors"""
    rng = np.random.default_rng(seed=seed)
    p1g0s = 0.15 * rng.random(num_qubits)
    p0g1s = 0.3 * rng.random(num_qubits)
    amats = np.stack([[1 - p1g0s, p1g0s], [p0g1s, 1 - p0g1s]]).T
    noise_model = NoiseModel()
    for i, amat in enumerate(amats):
        noise_model.add_readout_error(amat.T, [i])
    return noise_model
