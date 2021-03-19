# This code is part of Qiskit.
#
# (C) Copyright IBM 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Helper methods to extract data from the fits."""

from typing import Type
import numpy as np

from qiskit_experiments.calibration.exceptions import CalibrationError
from .trigonometric import CosineFit
from .fit_result import FitResult


def get_period_fraction(analysis: Type, angle: float, fit_result: FitResult) -> float:
    """
    Returns the x location corresponding to a given rotation angle. E.g.
    if angle = pi and the function function is cos(2 pi a x) then return pi/2*pi*a.
    Not all analysis routines will implement this.

    Args:
        analysis: The analysis routing from which to retrieve a periodicity.
        angle: The desired rotation angle.
        fit_result: the result of the fit with the fit values.

    Returns:
        period fraction: The x location corresponding to a given rotation angle.

    Raises:
        CalibrationError: if the fit function is not recognized.
    """
    if issubclass(analysis, CosineFit):
        return angle / (2 * np.pi * fit_result.fitvals[1])

    raise CalibrationError(f'Analysis class {analysis} is not supported.')

def get_min_location(analysis: Type, fit_result: FitResult) -> float:
    """
    Args:
        analysis: The analysis routing from which to retrieve the minimum location.
        fit_result: the result of the fit with the fit values.

    Returns:
        The location where the fit is minimum.

    Raises:
        CalibrationError: if the fit function is not recognized.
    """
    if isinstance(analysis, CosineFit):
        fit_params = fit_result.fitvals
        return (-np.pi - fit_params[2]) / (2*np.pi*fit_params[1])

    raise CalibrationError(f'Analysis class {analysis} is not supported.')
