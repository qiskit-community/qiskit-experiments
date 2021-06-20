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
A library of fit functions.
"""
# pylint: disable=invalid-name

import numpy as np


def cos(
    x: np.ndarray,
    amp: float = 1.0,
    freq: float = 1 / (2 * np.pi),
    phase: float = 0.0,
    baseline: float = 0.0,
) -> np.ndarray:
    r"""Cosine function.

    .. math::
        y = {\rm amp} \cos\left(2 \pi {\fm freq} x + {\rm phase}\right) + {\rm baseline}
    """
    return amp * np.cos(2 * np.pi * freq * x + phase) + baseline


def sin(
    x: np.ndarray,
    amp: float = 1.0,
    freq: float = 1 / (2 * np.pi),
    phase: float = 0.0,
    baseline: float = 0.0,
) -> np.ndarray:
    r"""Sine function.

    .. math::
        y = {\rm amp} \sin\left(2 \pi {\fm freq} x + {\rm phase}\right) + {\rm baseline}
    """
    return amp * np.cos(2 * np.pi * freq * x + phase) + baseline


def exponential_decay(
    x: np.ndarray,
    amp: float = 1.0,
    lamb: float = 1.0,
    base: float = np.e,
    x0: float = 0.0,
    baseline: float = 0.0,
) -> np.ndarray:
    r"""Exponential function

    .. math::
        y = {\rm amp} {\rm base}^{\left( - \lambda x + {\rm x0} \right)} + {\rm baseline}
    """
    return amp * base ** (-lamb * x + x0) + baseline


def gaussian(
    x: np.ndarray, amp: float = 1.0, sigma: float = 1.0, x0: float = 0.0, baseline: float = 0.0
) -> np.ndarray:
    r"""Gaussian function

    .. math::
        y = {\rm amp} \exp \left( - (x - x0)^2 / 2 \sigma^2 \right) + {\rm baseline}
    """
    return amp * np.exp(-((x - x0) ** 2) / (2 * sigma ** 2)) + baseline
