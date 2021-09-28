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

from qiskit_experiments.curve_analysis.standard_analysis.error_amplification_analysis import ErrorAmplificationAnalysis


class FineDragAnalysis(ErrorAmplificationAnalysis):
    """An analysis class for fine DRAG calibrations in which the amplitude of the fit is fixed."""

    __fixed_parameters__ = ["angle_per_gate", "phase_offset", "amp"]
