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
HEAT experiments for ZX Hamiltonian.
"""

from .elements import zx_elements
from .base_experiment import BaseCompositeHeat
from .base_analysis import CompositeHeatAnalysis


class HeatYAnalysis(CompositeHeatAnalysis):
    """"""
    __fit_params__ = ["d_heat_y0", "d_heat_y1"]
    __out_params__ = ["A_iy", "A_zy"]


class ZXHeatYError(BaseCompositeHeat):
    """HEAT experiments for Y error amplification.

    # section: overview
        TODO
    """

    __heat_elements__ = {
        "d_heat_y0": zx_elements.HeatElementPrepIIMeasIY,
        "d_heat_y1": zx_elements.HeatElementPrepXIMeasIY,
    }

    __analysis_class__ = HeatYAnalysis


class HeatZAnalysis(CompositeHeatAnalysis):
    """"""
    __fit_params__ = ["d_heat_z0", "d_heat_z1"]
    __out_params__ = ["A_iz", "A_zz"]


class ZXHeatZError(BaseCompositeHeat):
    """HEAT experiments for Z error amplification.

    # section: overview
        TODO
    """

    __heat_elements__ = {
        "d_heat_z0": zx_elements.HeatElementPrepIIMeasIZ,
        "d_heat_z1": zx_elements.HeatElementPrepXIMeasIZ,
    }

    __analysis_class__ = HeatZAnalysis
