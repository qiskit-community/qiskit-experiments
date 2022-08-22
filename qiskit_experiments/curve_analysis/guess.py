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
A library of parameter guess functions.
"""
# pylint: disable=invalid-name

import functools
from typing import Optional, Tuple, Callable

import numpy as np
from scipy import signal

from qiskit_experiments.exceptions import AnalysisError
from qiskit_experiments.warnings import deprecated_function


def frequency(
    x: np.ndarray,
    y: np.ndarray,
    filter_window: int = 5,
    filter_dim: int = 2,
) -> float:
    r"""Get frequency of oscillating signal.

    First this tries FFT. If the true value is likely below or near the frequency resolution,
    the function tries low frequency fit with

    .. math::

        f_{\rm est} = \frac{1}{2\pi {\rm max}\left| y \right|}
            {\rm max} \left| \frac{dy}{dx} \right|

    given :math:`y = A \cos (2\pi f x + phi)`. In this mode, y data points are
    smoothed by a Savitzky-Golay filter to protect against outlier points.

    .. note::

        This function returns always positive frequency.
        This function is sensitive to the DC offset.
        This function assumes sorted, no-overlapping x values.

    Args:
        x: Array of x values.
        y: Array of y values.
        filter_window: Window size of Savitzky-Golay filter. This should be odd number.
        filter_dim: Dimension of Savitzky-Golay filter.

    Returns:
        Frequency estimation of oscillation signal.
    """
    # to run FFT x interval should be identical
    intervals = np.unique(np.diff(x).round(decimals=4))
    try:
        dt = float(intervals)
        x_ = x
        y_ = y
    except TypeError:
        # Interpolate when data is nonuniform series.
        dt = np.min(intervals)
        x_ = np.arange(min(x), max(x), dt)
        y_ = np.interp(x_, xp=x, fp=y)

    fft_data = np.fft.fft(y_ - np.average(y_))
    freqs = np.fft.fftfreq(len(x_), dt)

    positive_freqs = freqs[freqs >= 0]
    positive_fft_data = fft_data[freqs >= 0]

    freq_guess = positive_freqs[np.argmax(np.abs(positive_fft_data))]

    if freq_guess < 1.5 / (dt * len(x_)):
        # low frequency fit, use this mode when the estimate is near the resolution
        y_smooth = signal.savgol_filter(y_, window_length=filter_window, polyorder=filter_dim)

        # no offset is assumed
        y_amp = max(np.abs(y_smooth))

        if np.isclose(y_amp, 0.0):
            # no oscillation signal
            return 0.0

        freq_guess = max(np.abs(np.diff(y_smooth) / dt)) / (y_amp * 2 * np.pi)

    return freq_guess


def max_height(
    y: np.ndarray,
    percentile: Optional[float] = None,
    absolute: bool = False,
) -> Tuple[float, int]:
    """Get maximum value of y curve and its index.

    Args:
        y: Array of y values.
        percentile: Return that percentile value if provided, otherwise just return max value.
        absolute: Use absolute y value.

    Returns:
        The maximum y value and index.
    """
    if percentile is not None:
        return get_height(y, functools.partial(np.percentile, q=percentile), absolute)
    return get_height(y, np.nanmax, absolute)


def min_height(
    y: np.ndarray,
    percentile: Optional[float] = None,
    absolute: bool = False,
) -> Tuple[float, int]:
    """Get minimum value of y curve and its index.

    Args:
        y: Array of y values.
        percentile: Return that percentile value if provided, otherwise just return min value.
        absolute: Use absolute y value.

    Returns:
        The minimum y value and index.
    """
    if percentile is not None:
        return get_height(y, functools.partial(np.percentile, q=percentile), absolute)
    return get_height(y, np.nanmin, absolute)


def get_height(
    y: np.ndarray,
    find_height: Callable,
    absolute: bool = False,
) -> Tuple[float, int]:
    """Get specific value of y curve defined by a callback and its index.

    Args:
        y: Array of y values.
        find_height: A callback to find preferred y value.
        absolute: Use absolute y value.

    Returns:
        The target y value and index.
    """
    if absolute:
        y_ = np.abs(y)
    else:
        y_ = y

    y_target = find_height(y_)
    index = int(np.argmin(np.abs(y_ - y_target)))

    return y_target, index


def exp_decay(x: np.ndarray, y: np.ndarray) -> float:
    r"""Get exponential decay parameter.

    This assumes following function form.

    .. math::

        y(x) = e^{\alpha x} F(x)

    Where :math:`F(x)` might be a sinusoidal function or constant value.

    .. note::

        This function assumes the data is roughly evenly spaced and that
        the y data goes through a few periods so that the peak to peak
        value early in the data can be compared to the peak to peak later
        in the data to estimate the decay constant.

    This function splits the data at the middle of x data, and compare the
    10-90 percentile peak to peak value on the
    left-hand :math:`y_{p-p}^R` and right-hand :math:`y_{p-p}^L` side. Namely,

    .. math::

        y_{p-p}^R = \exp(\alpha x_R)
        y_{p-p}^L = \exp(\alpha x_L)

    and the exponent :math:`\alpha` can be solved by

    .. math::

        \alpha = \frac{\log y_{p-p}^R / y_{p-p}^L}{x_R - x_L}

    Args:
        x: Array of x values.
        y: Array of y values.

    Returns:
         Exponent of curve.
    """
    x_median = np.median(x)
    i_l = x < x_median
    i_r = x > x_median

    y_l_min, y_l_max = np.percentile(y[i_l], [10, 90])
    y_r_min, y_r_max = np.percentile(y[i_r], [10, 90])
    dy_l = y_l_max - y_l_min
    dy_r = y_r_max - y_r_min

    x_l = np.average(x[i_l])
    x_r = np.average(x[i_r])

    return np.log(dy_r / dy_l) / (x_r - x_l)


@deprecated_function(last_version="0.6", msg="Use exp_decay instead.")
def oscillation_exp_decay(
    x: np.ndarray,
    y: np.ndarray,
    filter_window: int = 5,
    filter_dim: int = 2,
    freq_guess: Optional[float] = None,
) -> float:
    r"""Get exponential decay parameter from oscillating signal.

    This assumes following function form.

    .. math::

        y(x) = e^{\alpha x} F(x),

    where :math:`F(x)` is arbitrary oscillation function oscillating at ``freq_guess``.
    This function first applies a Savitzky-Golay filter to y value,
    then run scipy peak search to extract peak positions.
    If ``freq_guess`` is provided, the search function will be robust to fake peaks due to noise.
    This function calls :py:func:`exp_decay` function for extracted x and y values at peaks.

    .. note::

        y values should contain more than one cycle of oscillation to use this guess approach.

    Args:
        x: Array of x values.
        y: Array of y values.
        filter_window: Window size of Savitzky-Golay filter. This should be odd number.
        filter_dim: Dimension of Savitzky-Golay filter.
        freq_guess: Optional. Initial frequency guess of :math:`F(x)`.

    Returns:
         Decay rate of signal.
    """
    y_smoothed = signal.savgol_filter(y, window_length=filter_window, polyorder=filter_dim)

    if freq_guess is not None and np.abs(freq_guess) > 0:
        period = 1 / np.abs(freq_guess)
        dt = np.mean(np.diff(x))
        width_samples = int(np.round(0.8 * period / dt))
    else:
        width_samples = 1

    peak_pos, _ = signal.find_peaks(y_smoothed, distance=width_samples)

    if len(peak_pos) < 2:
        return 0.0

    x_peaks = x[peak_pos]
    y_peaks = y_smoothed[peak_pos]

    return exp_decay(x_peaks, y_peaks)


def full_width_half_max(
    x: np.ndarray,
    y: np.ndarray,
    peak_index: int,
) -> float:
    """Get full width half maximum value of the peak. Offset of y should be removed.

    Args:
        x: Array of x values.
        y: Array of y values.
        peak_index: Index of peak.

    Returns:
        FWHM of the peak.

    Raises:
        AnalysisError: When peak is too broad and line width is not found.
    """
    y_ = np.abs(y)
    peak_height = y_[peak_index]
    halfmax_removed = np.sign(y_ - 0.5 * peak_height)

    try:
        r_bound = np.min(x[(halfmax_removed == -1) & (x > x[peak_index])])
    except ValueError:
        r_bound = None
    try:
        l_bound = np.max(x[(halfmax_removed == -1) & (x < x[peak_index])])
    except ValueError:
        l_bound = None

    if r_bound and l_bound:
        return r_bound - l_bound
    elif r_bound:
        return 2 * (r_bound - x[peak_index])
    elif l_bound:
        return 2 * (x[peak_index] - l_bound)

    raise AnalysisError("FWHM of input curve was not found. Perhaps scanning range is too narrow.")


def constant_spectral_offset(
    y: np.ndarray, filter_window: int = 5, filter_dim: int = 2, ratio: float = 0.1
) -> float:
    """Get constant offset of spectral baseline.

    This function searches constant offset by finding a region where 1st and 2nd order
    differentiation are close to zero. A return value is an average y value of that region.
    To suppress the noise contribution to derivatives, this function also applies a
    Savitzky-Golay filter to y value.

    This method is more robust to offset error than just taking median or average of y values
    especially when a peak width is wider compared to the scan range.

    Args:
        y: Array of y values.
        filter_window: Window size of Savitzky-Golay filter. This should be odd number.
        filter_dim: Dimension of Savitzky-Golay filter.
        ratio: Threshold value to decide flat region. This value represent a ratio
            to the maximum derivative value.

    Returns:
        Offset value.
    """
    y_smoothed = signal.savgol_filter(y, window_length=filter_window, polyorder=filter_dim)

    ydiff1 = np.abs(np.diff(y_smoothed, 1, append=np.nan))
    ydiff2 = np.abs(np.diff(y_smoothed, 2, append=np.nan, prepend=np.nan))
    non_peaks = y_smoothed[
        (ydiff1 < ratio * np.nanmax(ydiff1)) & (ydiff2 < ratio * np.nanmax(ydiff2))
    ]

    if len(non_peaks) == 0:
        return float(np.median(y))

    return np.average(non_peaks)


def constant_sinusoidal_offset(y: np.ndarray) -> float:
    """Get constant offset of sinusoidal signal.

    This function finds 95 and 5 percentile y values and take an average of them.
    This method is robust to the dependency on sampling window, i.e.
    if we sample sinusoidal signal for 2/3 of its period, simple averaging may induce
    a drift towards positive or negative direction depending on the phase offset.

    Args:
        y: Array of y values.

    Returns:
        Offset value.
    """
    maxv, _ = max_height(y, percentile=95)
    minv, _ = min_height(y, percentile=5)

    return 0.5 * (maxv + minv)


def rb_decay(
    x: np.ndarray,
    y: np.ndarray,
    b: float = 0.5,
) -> float:
    r"""Get base of exponential decay function which is assumed to be close to 1.

    This assumes following model:

    .. math::

        y(x) = a \alpha^x + b.

    To estimate the base of decay function :math:`\alpha`, we consider

    .. math::

        y'(x) = y(x) - b = a \alpha^x,

    and thus,

    .. math::

        y'(x+dx) = a \alpha^x \alpha^dx.

    By considering the ratio of y values at :math:`x+dx` to :math:`x`,

    .. math::

        ry = \frac{a \alpha^x \alpha^dx}{a \alpha^x} = \alpha^dx.

    From this relationship, we can estimate :math:`\alpha` as

    .. math::

        \alpha = ry^\frac{1}{dx}.

    Args:
        x: Array of x values.
        y: Array of y values.
        b: Asymptote of decay function.

    Returns:
         Base of decay function.
    """
    valid_inds = y > b

    # Remove y values below b
    y = y[valid_inds]
    x = x[valid_inds]

    if len(x) < 2:
        # If number of element is 1, assume y(0) = 1.0 and directly compute alpha.
        a = 1.0 - b
        return ((y[0] - b) / a) ** (1 / x[0])

    ry = (y[1:] - b) / (y[:-1] - b)
    dx = np.diff(x)

    return np.average(ry ** (1 / dx))


def sinusoidal_freq_offset_amp(
    x: np.ndarray,
    y: np.ndarray,
    delay: int,
) -> Tuple[float, float]:
    r"""Get frequency, offset and amplitude of sinusoidal signal.

    This function simultaneously estimates the frequency and offset
    by using two delayed data sets :math:`y(t-\lambda)` and :math:`y(t-2\lambda)`
    generated from the input :math:`y(t)` values, where :math:`\lambda` is a
    delay parameter specified by the function argument.
    See the following paper for the details of the protocol.

    Khac, T.; Vlasov, S. and Iureva, R. (2021).
    Estimating the Frequency of the Sinusoidal Signal using
    the Parameterization based on the Delay Operators.
    DOI: 10.5220/0010536506560660

    Once offset :math:`y_0` is estimated, the amplitude of the sinusoidal
    is obtained by :math:`A = \max |y - y_0|`.

    .. note::

        This algorithm poorly works for y values containing only less than a half oscillation cycle.
        When the parameter cannot be estimated, this switches to the standard FFT analysis
        for the frequency and estimates the offset by simply averaging over y values.

    Args:
        x: X values.
        y: Y values.
        delay: Integer parameter to specify the delay samples. Using smaller value
            increases (decreases) sensitivity for high (low) frequency signal.

    Returns:
        A tuple of frequency, offset and amplitude.
    """
    intervals = np.unique(np.diff(x).round(decimals=4))
    try:
        dt = float(intervals)
        y_ = y
    except TypeError:
        # Interpolate when data is nonuniform series.
        dt = np.min(intervals)
        x_ = np.arange(min(x), max(x), dt)
        y_ = np.interp(x_, xp=x, fp=y)

    y0 = y_[2 * delay:]
    y1 = y_[delay:-delay]
    y2 = y_[:-2 * delay]

    # Liner regression form
    # psi = xi theta
    xi = np.vstack((y1, np.ones_like(y1))).T
    psi = (y0 + y2).reshape((y0.size, 1))

    # Solve linear regression
    # theta = Inv(xi^T xi) xi^T psi
    xtx = np.dot(xi.T, xi)
    theta = np.dot(np.dot(np.linalg.inv(xtx), xi.T), psi)

    theta1 = float(theta[0])
    theta2 = float(theta[1])

    # Offset
    if np.isclose(theta1, 2.0):
        offset = np.average(y)
    else:
        offset = theta2 / (2 - theta1)

    # Frequency
    a1 = theta1 / 2.0
    if np.abs(a1) >= 1.0:
        # out of arc cosine domain, use FFT
        freq = frequency(x, y - offset)
    else:
        freq = np.arccos(a1) / (delay * 2 * np.pi) / dt

    # Amplitude
    amp = max_height(y - offset, absolute=True)[0]

    return freq, offset, amp
