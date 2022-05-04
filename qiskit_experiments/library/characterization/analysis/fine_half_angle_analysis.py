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

"""Fine half angle calibration analysis."""

import warnings

import numpy as np
from qiskit_experiments.framework import Options
from qiskit_experiments.curve_analysis import ErrorAmplificationAnalysis, ParameterRepr


class FineHalfAngleAnalysis(ErrorAmplificationAnalysis):
    r"""Analysis class for the HalfAngle experiment to define the fixed parameters.

    # section: note

        The following parameters are held fixed during fitting.

        * :math:`{\rm apg}` The angle per gate is set by the user, for example pi for a pi-pulse.
        * :math:`{\rm phase\_offset}` The phase offset in the cosine oscillation.
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
        r"""Default analysis options.

        If the rotation error is very small the fit may chose a d_theta close to
        :math:`\pm\pi`. To prevent this we impose bounds on d_theta. Note that the
        options angle per gate, phase offset and amp are not intended to be changed.
        """
        options = super()._default_options()
        options.result_parameters = [ParameterRepr("d_theta", "d_hac", "rad")]
        options.normalization = True
        options.angle_per_gate = np.pi
        options.phase_offset = -np.pi / 2
        options.amp = 1.0
        options.bounds.update({"d_theta": (-np.pi / 2, np.pi / 2)})

        return options
