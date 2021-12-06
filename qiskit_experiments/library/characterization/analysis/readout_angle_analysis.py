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
Readout Angle Analysis class.
"""

import numpy as np

from qiskit_experiments.framework import BaseAnalysis, AnalysisResultData


class ReadoutAngleAnalysis(BaseAnalysis):
    """
    A class to analyze readout angle experiments
    """

    def _run_analysis(self, experiment_data):
        angles = []
        for i in range(2):
            center = complex(*experiment_data.data(i)["memory"][0])
            angles.append(np.angle(center))

        angle = (angles[0] + angles[1]) / 2
        if (np.abs(angles[0] - angles[1])) % (2 * np.pi) > np.pi:
            angle += np.pi

        analysis_results = [
            AnalysisResultData(
                name="ReadoutAngle",
                value=angle,
                extra={"angle_ground": angles[0], "angle_excited": angles[1]},
            )
        ]

        return analysis_results, []
