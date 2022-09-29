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

"""Fine Amplitude calibration analysis."""

import lmfit

import qiskit_experiments.curve_analysis as curve


class FineAmplitudeAnalysis(curve.ErrorAmplificationAnalysis):
    r"""An analysis class for fine amplitude calibrations to define the fixed parameters.

    # section: note

        The following parameters are fixed.

        * :math:`{\rm apg}` The angle per gate is set by the user, for example pi for a pi-pulse.
        * :math:`{\rm phase\_offset}` The phase offset in the cosine oscillation, for example,
          :math:`\pi/2` if a square-root of X gate is added before the repeated gates.
    """

    # pylint: disable=super-init-not-called
    def __init__(self):

        # pylint: disable=non-parent-init-called
        curve.CurveAnalysis.__init__(
            self,
            models=[
                lmfit.models.ExpressionModel(
                    expr="amp / 2 * (2 * x - 1) + base",
                    name="spam cal.",
                    data_sort_key={"series": "spam-cal"},
                ),
                lmfit.models.ExpressionModel(
                    expr="amp / 2 * cos((d_theta + angle_per_gate) * x - phase_offset) + base",
                    name="fine amp.",
                    data_sort_key={"series": 1},
                ),
            ],
        )

    @classmethod
    def _default_options(cls):
        """Return the default analysis options."""
        default_options = super()._default_options()
        return default_options
