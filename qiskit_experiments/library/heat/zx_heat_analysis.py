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
HEAT analysis for ZX Hamiltonian.
"""

from .base_analysis import CompositeHeatAnalysis


class HeatYAnalysis(CompositeHeatAnalysis):
    """"""
    __fit_params__ = ["d_heat_y0", "d_heat_y1"]
    __out_params__ = ["A_iy", "A_zy"]


class HeatZAnalysis(CompositeHeatAnalysis):
    """"""
    __fit_params__ = ["d_heat_z0", "d_heat_z1"]
    __out_params__ = ["A_iz", "A_zz"]
