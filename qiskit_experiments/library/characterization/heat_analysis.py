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
Analysis for HEAT experiments.
"""

import numpy as np

from qiskit_experiments.curve_analysis import ErrorAmplificationAnalysis
from qiskit_experiments.framework import Options


class HeatAnalysis(ErrorAmplificationAnalysis):
    """An analysis class for HEAT experiment to define the fixed parameters."""

    __fixed_parameters__ = ["angle_per_gate", "phase_offset", "amp"]

    @classmethod
    def _default_options(cls) -> Options:
        """Default analysis options."""
        options = super()._default_options()
        options.angle_per_gate = 0.0
        options.phase_offset = -np.pi / 2
        options.amp = 1.0

        return options
