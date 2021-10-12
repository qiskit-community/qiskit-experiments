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

from qiskit_experiments.curve_analysis import ErrorAmplificationAnalysis


class FineHalfAngleAnalysis(ErrorAmplificationAnalysis):
    r"""Analysis class for the HalfAngle experiment to define the fixed parameters.

    # section: note

        The following parameters are fixed.

        * :math:`{\rm apg}` The angle per gate is set by the user, for example pi for a pi-pulse.
        * :math:`{\rm phase\_offset}` The phase offset in the cosine oscillation.
        * :math:`{\rm amp}` The amplitude of the oscillation.
    """

    __fixed_parameters__ = ["angle_per_gate", "phase_offset", "amp"]
