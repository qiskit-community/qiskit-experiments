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

"""Analysis utility functions."""

from typing import Optional

from qiskit_experiments.curve_analysis.curve_data import FitData
from qiskit_experiments.framework import FitVal


def get_fitval(fit_data: FitData, param_name: str, unit: Optional[str] = None) -> FitVal:
    """A helper function to format fit value object from fit data.

    Args:
        fit_data: Fitting data set.
        param_name: Name of parameters to extract.
        unit: Optional. Unit of this value.

    Returns:
        FitVal object.

    Raises:
        KeyError: When the result does not contain parameter information.
        ValueError: When specified parameter is not defined.
    """
    if not fit_data.popt_keys:
        raise KeyError(
            "Fit result has not fit parameter name information. "
            "Please confirm if the fit is successfully completed."
        )

    try:
        index = fit_data.popt_keys.index(param_name)
        return FitVal(
            value=fit_data.popt[index],
            stderr=fit_data.popt_err[index],
            unit=unit,
        )
    except ValueError as ex:
        raise ValueError(f"Parameter {param_name} is not defined.") from ex
