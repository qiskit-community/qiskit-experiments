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
# pylint: disable=invalid-name, no-member

import functools
from typing import Callable, Union

import numpy as np
from uncertainties import unumpy as unp, UFloat


def typecast_float(fit_func: Callable) -> Callable:
    """A decorator to typecast y values to a float array if the input parameters have no error.

    Args:
        fit_func: Fit function that returns a ufloat array or an array of float.

    Returns:
        Fit function with typecast.
    """

    @functools.wraps(fit_func)
    def _wrapper(x, *args, **kwargs) -> Union[float, UFloat, np.ndarray]:
        yvals = fit_func(x, *args, **kwargs)
        try:
            if isinstance(x, float):
                return float(yvals)
            return yvals.astype(float)
        except TypeError:
            return yvals

    return _wrapper


@typecast_float
def cos(
    x: np.ndarray,
    amp: float = 1.0,
    freq: float = 1 / (2 * np.pi),
    phase: float = 0.0,
    baseline: float = 0.0,
) -> np.ndarray:
    r"""Cosine function.

    .. math::
        y = {\rm amp} \cdot \cos\left(2 \pi {\rm freq} \cdot x
            + {\rm phase}\right) + {\rm baseline}
    """
    return amp * unp.cos(2 * np.pi * freq * x + phase) + baseline


@typecast_float
def sin(
    x: np.ndarray,
    amp: float = 1.0,
    freq: float = 1 / (2 * np.pi),
    phase: float = 0.0,
    baseline: float = 0.0,
) -> np.ndarray:
    r"""Sine function.

    .. math::
        y = {\rm amp} \cdot \sin\left(2 \pi {\rm freq} \cdot x
            + {\rm phase}\right) + {\rm baseline}
    """
    return amp * unp.sin(2 * np.pi * freq * x + phase) + baseline


@typecast_float
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
        y = {\rm amp} \cdot {\rm base}^{\left( - \lambda x + {\rm x0} \right)} + {\rm baseline}
    """
    return amp * base ** (-lamb * x + x0) + baseline


@typecast_float
def gaussian(
    x: np.ndarray, amp: float = 1.0, sigma: float = 1.0, x0: float = 0.0, baseline: float = 0.0
) -> np.ndarray:
    r"""Gaussian function

    .. math::
        y = {\rm amp} \cdot \exp \left( - (x - x0)^2 / 2 \sigma^2 \right) + {\rm baseline}
    """
    return amp * unp.exp(-((x - x0) ** 2) / (2 * sigma**2)) + baseline


@typecast_float
def sqrt_lorentzian(
    x: np.ndarray, amp: float = 1.0, kappa: float = 1.0, x0: float = 0.0, baseline: float = 0.0
) -> np.ndarray:
    r"""Square-root Lorentzian function for spectroscopy.

    .. math::
        y = \frac{{\rm amp} |\kappa|}{\sqrt{\kappa^2 + 4(x -x_0)^2}} + {\rm baseline}
    """
    return amp * np.abs(kappa) / unp.sqrt(kappa**2 + 4 * (x - x0) ** 2) + baseline


@typecast_float
def cos_decay(
    x: np.ndarray,
    amp: float = 1.0,
    tau: float = 1.0,
    freq: float = 1 / (2 * np.pi),
    phase: float = 0.0,
    baseline: float = 0.0,
) -> np.ndarray:
    r"""Cosine function with exponential decay.

    .. math::
        y = {\rm amp} \cdot e^{-x/\tau} \cos\left(2 \pi \cdot {\rm freq} \cdot x
        + {\rm phase}\right) + {\rm baseline}
    """
    return exponential_decay(x, lamb=1 / tau) * cos(x, amp=amp, freq=freq, phase=phase) + baseline


@typecast_float
def sin_decay(
    x: np.ndarray,
    amp: float = 1.0,
    tau: float = 1.0,
    freq: float = 1 / (2 * np.pi),
    phase: float = 0.0,
    baseline: float = 0.0,
) -> np.ndarray:
    r"""Sine function with exponential decay.

    .. math::
        y = {\rm amp} \cdot e^{-x/\tau} \sin\left(2 \pi \cdot {\rm freq} \cdot x
        + {\rm phase}\right) + {\rm baseline}
    """
    return exponential_decay(x, lamb=1 / tau) * sin(x, amp=amp, freq=freq, phase=phase) + baseline


@typecast_float
def bloch_oscillation_x(
    x: np.ndarray, px: float = 0.0, py: float = 0.0, pz: float = 0.0, baseline: float = 0.0
):
    r"""Bloch oscillation in x basis.

    .. math::
        y = \frac{\left( - p_z p_x + p_z p_x \cos (\omega x)
        + \omega p_y \sin (\omega x) \right)}{\omega^2} + {\rm baseline},

    where :math:`\omega = \sqrt{p_x^2 + p_y^2 + p_z^2}`. The `p_i` stands for the
    measured probability in :math:`i \in \left\{ X, Y, Z \right\}` basis.
    """
    w = unp.sqrt(px**2 + py**2 + pz**2)

    return (-pz * px + pz * px * unp.cos(w * x) + w * py * unp.sin(w * x)) / (w**2) + baseline


@typecast_float
def bloch_oscillation_y(
    x: np.ndarray, px: float = 0.0, py: float = 0.0, pz: float = 0.0, baseline: float = 0.0
):
    r"""Bloch oscillation in y basis.

    .. math::
        y = \frac{\left( p_z p_y - p_z p_y \cos (\omega x)
        - \omega p_x \sin (\omega x) \right)}{\omega^2} + {\rm baseline},

    where :math:`\omega = \sqrt{p_x^2 + p_y^2 + p_z^2}`. The `p_i` stands for the
    measured probability in :math:`i \in \left\{ X, Y, Z \right\}` basis.
    """
    w = unp.sqrt(px**2 + py**2 + pz**2)

    return (pz * py - pz * py * unp.cos(w * x) - w * px * unp.sin(w * x)) / (w**2) + baseline


@typecast_float
def bloch_oscillation_z(
    x: np.ndarray, px: float = 0.0, py: float = 0.0, pz: float = 0.0, baseline: float = 0.0
):
    r"""Bloch oscillation in z basis.

    .. math::
        y = \frac{\left( p_z^2 + (p_x^2 + p_y^2) \cos (\omega x) \right)}{\omega^2}
        + {\rm baseline},

    where :math:`\omega = \sqrt{p_x^2 + p_y^2 + p_z^2}`. The `p_i` stands for the
    measured probability in :math:`i \in \left\{ X, Y, Z \right\}` basis.
    """
    w = unp.sqrt(px**2 + py**2 + pz**2)

    return (pz**2 + (px**2 + py**2) * unp.cos(w * x)) / (w**2) + baseline
