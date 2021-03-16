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

"""Rough amplitude calibration."""

from qiskit_experiments.base_experiment import BaseExperiment


class RoughAmplitude(BaseExperiment):

    # Analysis class for experiment
    __analysis_class__ = None

    def __init__(self):

        qubits = None
        super.__init__(qubits, 'rough_amplitude_calibration')

    def circuits(self, backend=None, **circuit_options):
        """"""
