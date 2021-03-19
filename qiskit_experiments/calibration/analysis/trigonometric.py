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

"""Trigonometric analysis."""

from typing import Iterator, List, Tuple
import numpy as np
from scipy import signal

from .calibration_analysis import BaseCalibrationAnalysis


def freq_guess(xvals: np.ndarray, yvals: np.ndarray) -> float:
    """Initial frequency guess for oscillating data.

    Args:
        xvals: The independent values.
        yvals: The dependent values.

    Returns:
        frequency: An estimation of the frequency based on a FFT.
    """

    # Subtract DC component
    fft_data = np.fft.fft(yvals - np.mean(yvals))
    fft_freq = np.fft.fftfreq(len(xvals), xvals[1] - xvals[0])

    # Fit positive part of the spectrum
    f0_guess = np.abs(fft_freq[np.argmax(np.abs(fft_data[0:len(fft_freq) // 2]))])

    if f0_guess == 0:
        # sampling duration is shorter than oscillation period
        yvals = np.convolve(yvals, [0.5, 0.5], mode='same')
        peaks, = signal.argrelmin(yvals, order=int(len(xvals) / 4))
        if len(peaks) == 0 or len(peaks) > 4:
            return 0
        else:
            return 1 / (2 * xvals[peaks[0]])

    return f0_guess


class CosineFit(BaseCalibrationAnalysis):
    r"""Fit with $F(x) = a \cos(2\pi f x + \phi) + b$."""

    def initial_guess(self, xvals: np.ndarray, yvals: np.ndarray) -> Iterator[np.ndarray]:
        """Initial guess based on data.

        Args:
            xvals: The independent values.
            yvals: The dependent values.

        Yields:
            guess: An array containing the values of the initial guess.
        """
        y_mean = np.mean(yvals)
        a0 = np.max(np.abs(yvals)) - np.abs(y_mean)
        f0 = max(0, freq_guess(xvals, yvals))

        for phi in np.linspace(-np.pi, np.pi, 10):
            yield np.array([a0, f0, phi, y_mean])

    def fit_function(self, xvals: np.ndarray, *args) -> np.ndarray:
        """The cosine fit function.

        Args:
            xvals: The values along the x-axis.
            args: The parameters of the fit: as [a, f, phi, b], see class docstring.

        Returns:
             The value of the function with the given x value and parameters.
        """
        return args[0] * np.cos(2 * np.pi * args[1] * xvals + args[2]) + args[3]

    def fit_boundary(self, xvals: np.ndarray, yvals: np.ndarray) -> List[Tuple[float, float]]:
        """Allowed ranges for the parameters."""
        return [(-np.inf, np.inf), (0, np.inf), (-np.pi, np.pi), (-np.inf, np.inf)]
