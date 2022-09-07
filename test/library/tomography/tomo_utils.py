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
from qiskit import QuantumCircuit


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


def teleport_circuit():
    """Teleport qubit 0 to qubit 2"""
    teleport = QuantumCircuit(3, 2)
    teleport.h(1)
    teleport.cx(1, 2)
    teleport.cx(0, 1)
    teleport.h(0)
    teleport.measure(0, 0)
    teleport.measure(1, 1)
    # Conditionals
    creg = teleport.cregs[0]
    teleport.z(2).c_if(creg[0], 1)
    teleport.x(2).c_if(creg[1], 1)
    return teleport
