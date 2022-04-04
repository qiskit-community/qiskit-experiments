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

"""Fine DRAG calibration analysis."""

import warnings

import numpy as np
from qiskit_experiments.curve_analysis import ErrorAmplificationAnalysis
from qiskit_experiments.framework import Options


class FineDragAnalysis(ErrorAmplificationAnalysis):
    r"""An analysis class for fine DRAG calibrations to define the fixed parameters.

    # section: note

        The following parameters are fixed.

        * :math:`{\rm apg}` The angle per gate is set by the user, for example pi for a pi-pulse.
        * :math:`{\rm phase\_offset}` The phase offset in the cosine oscillation, for example,
          :math:`\pi/2` if a square-root of X gate is added before the repeated gates.
        * :math:`{\rm amp}` The amplitude of the oscillation.
    """

    __fixed_parameters__ = ["angle_per_gate", "phase_offset", "amp"]

    def __init__(self):
        super().__init__()

        warnings.warn(
            f"{self.__class__.__name__} has been deprecated. Use ErrorAmplificationAnalysis "
            "instance with the analysis options involving the fixed_parameters.",
            DeprecationWarning,
            stacklevel=2,
        )

    @classmethod
    def _default_options(cls) -> Options:
        """Default analysis options."""
        options = super()._default_options()
        options.normalization = True
        options.angle_per_gate = 0.0
        options.phase_offset = np.pi / 2
        options.amp = 1.0
        return options
