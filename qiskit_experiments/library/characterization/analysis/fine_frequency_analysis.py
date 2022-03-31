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

"""Fine frequency experiment analysis."""

import warnings

import numpy as np

from qiskit_experiments.curve_analysis import ErrorAmplificationAnalysis
from qiskit_experiments.framework import Options


class FineFrequencyAnalysis(ErrorAmplificationAnalysis):
    r"""An analysis class for fine frequency experiments.

    # section: note

        The following parameters are fixed.

        * :math:`{\rm apg}` The angle per gate is pi / 2 for fine frequency analysis.
        * :math:`{\rm phase\_offset}` The phase offset in the cosine oscillation which is 0.
    # section: see_also
        :py:class:`~qiskit_experiments.curve_analysis.ErrorAmplificationAnalysis`
    """

    __fixed_parameters__ = ["angle_per_gate", "phase_offset"]

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
        options.angle_per_gate = np.pi / 2
        options.phase_offset = 0.0
        return options
