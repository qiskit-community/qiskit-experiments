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
Utils to generate quantum volume experiment data reference.
"""

from qiskit.providers.aer.noise import NoiseModel
from qiskit.providers.aer.noise.errors import readout_error
from qiskit.providers.aer.noise.errors.standard_errors import (
    depolarizing_error,
    thermal_relaxation_error,
)


def create_noise_model():
    """
    create noise model with depolarizing error, thermal error and readout error
    Returns:
        NoiseModel: noise model
    """
    noise_model = NoiseModel()
    p1q = 0.0004
    p2q = 0.01
    depol_sx = depolarizing_error(p1q, 1)
    depol_x = depolarizing_error(p1q, 1)
    depol_cx = depolarizing_error(p2q, 2)

    # Add T1/T2 noise to the simulation
    t_1 = 110e3
    t_2 = 120e3
    gate1q = 50
    gate2q = 100
    termal_sx = thermal_relaxation_error(t_1, t_2, gate1q)
    termal_x = thermal_relaxation_error(t_1, t_2, gate1q)
    termal_cx = thermal_relaxation_error(t_1, t_2, gate2q).tensor(
        thermal_relaxation_error(t_1, t_2, gate2q)
    )

    noise_model.add_all_qubit_quantum_error(depol_sx.compose(termal_sx), "sx")
    noise_model.add_all_qubit_quantum_error(depol_x.compose(termal_x), "x")
    noise_model.add_all_qubit_quantum_error(depol_cx.compose(termal_cx), "cx")

    read_err = readout_error.ReadoutError([[0.98, 0.02], [0.04, 0.96]])
    noise_model.add_all_qubit_readout_error(read_err)
    return noise_model


def create_high_noise_model():
    """
    create high noise model with depolarizing error, thermal error and readout error
    Returns:
        NoiseModel: noise model
    """
    noise_model = NoiseModel()
    p1q = 0.004
    p2q = 0.05
    depol_sx = depolarizing_error(p1q, 1)
    depol_x = depolarizing_error(p1q, 1)
    depol_cx = depolarizing_error(p2q, 2)

    # Add T1/T2 noise to the simulation
    t_1 = 110e2
    t_2 = 120e2
    gate1q = 50
    gate2q = 100
    termal_sx = thermal_relaxation_error(t_1, t_2, gate1q)
    termal_x = thermal_relaxation_error(t_1, t_2, gate1q)
    termal_cx = thermal_relaxation_error(t_1, t_2, gate2q).tensor(
        thermal_relaxation_error(t_1, t_2, gate2q)
    )

    noise_model.add_all_qubit_quantum_error(depol_sx.compose(termal_sx), "sx")
    noise_model.add_all_qubit_quantum_error(depol_x.compose(termal_x), "x")
    noise_model.add_all_qubit_quantum_error(depol_cx.compose(termal_cx), "cx")

    read_err = readout_error.ReadoutError([[0.98, 0.02], [0.04, 0.96]])
    noise_model.add_all_qubit_readout_error(read_err)
    return noise_model
