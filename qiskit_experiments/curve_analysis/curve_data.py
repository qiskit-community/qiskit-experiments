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
Curve data classes.
"""

import dataclasses
from typing import Any, Dict, Callable, Union
import numpy as np


@dataclasses.dataclass(frozen=True)
class SeriesDef:
    """Description of curve."""

    # Arbitrary callback to define the fit function. First argument should be x.
    fit_func: Callable

    # Keyword dictionary to define the series with circuit metadata
    filter_kwargs: Dict[str, Any] = dataclasses.field(default_factory=dict)

    # Name of this series. This name will appear in the figure and raw x-y value report.
    name: str = "Series-0"

    # Color of this line.
    plot_color: str = "black"

    # Symbol to represent data points of this line.
    plot_symbol: str = "o"

    # Whether to plot fit uncertainty for this line.
    plot_fit_uncertainty: bool = False

    # Latex description of this fit model
    model_description: str = "no description"


@dataclasses.dataclass(frozen=True)
class CurveData:
    """Set of extracted experiment data."""

    label: str
    x: np.ndarray
    y: np.ndarray
    y_err: np.ndarray
    data_index: Union[np.ndarray, int]
    metadata: np.ndarray = None
