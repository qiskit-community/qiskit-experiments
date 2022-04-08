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

import numpy as np

import qiskit_experiments.curve_analysis as curve
from qiskit_experiments.curve_analysis import ErrorAmplificationAnalysis


class FineAmplitudeAnalysis(ErrorAmplificationAnalysis):
    r"""An analysis class for fine amplitude calibrations to define the fixed parameters.

    # section: note

        The following parameters are fixed.

        * :math:`{\rm apg}` The angle per gate is set by the user, for example pi for a pi-pulse.
        * :math:`{\rm phase\_offset}` The phase offset in the cosine oscillation, for example,
          :math:`\pi/2` if a square-root of X gate is added before the repeated gates.
    """

    # The intended angle per gat of the gate being calibrated, e.g. pi for a pi-pulse.

    __series__ = [
        curve.SeriesDef(
            # pylint: disable=line-too-long
            fit_func=lambda x, amp, d_theta, phase_offset, base, angle_per_gate: base
            + 0.5 * amp * (2 * x - 1),
            plot_color="green",
            model_description=r"{\rm base} + \frac{{\rm amp}}{2} * (2 * x - 1)",
            name="spam cal.",
            filter_kwargs={"series": "spam-cal"},
        ),
        curve.SeriesDef(
            # pylint: disable=line-too-long
            fit_func=lambda x, amp, d_theta, phase_offset, base, angle_per_gate: curve.fit_function.cos(
                x,
                amp=0.5 * amp,
                freq=(d_theta + angle_per_gate) / (2 * np.pi),
                phase=-phase_offset,
                baseline=base,
            ),
            plot_color="blue",
            model_description=r"\frac{{\rm amp}}{2}\cos\left(x[{\rm d}\theta + {\rm apg} ] "
            r"+ {\rm phase\_offset}\right)+{\rm base}",
            name="fine amp.",
            filter_kwargs={"series": 1},
        ),
    ]
