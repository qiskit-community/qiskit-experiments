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

import numpy as np


def hs_distance_result(result, target):
    """Hilbert-Schmidtt norm for the target and gst results"""

    hs_distance_results = {}
    for key in result.keys():
        if key not in ["E", "rho"]:
            hs_distance_results[key] = hs_distance(result[key].data, target[key].data)
    return hs_distance_results


def froebenius_distance(result, target):
    """froebenius distance between target and gst results"""
    froeb_distance = {}
    for key in result.keys():
        if key not in ["E", "rho"]:
            froeb_distance[key] = froeb_dist(result[key].data, target[key].data)
    return froeb_distance


def hs_distance(A, B):
    # Computes the Hilbert-Schmidt distance between two matrices A and B

    return sum([np.abs(x) ** 2 for x in np.nditer(A - B)])


def froeb_dist(A, B):

    # Computes the Frobenius distance between two matrices A and B.

    return np.sqrt(np.trace(np.dot(np.subtract(A, B), np.subtract(A, B).T)))
