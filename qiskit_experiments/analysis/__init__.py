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
=====================================================
Analysis Library (:mod:`qiskit_experiments.analysis`)
=====================================================

.. currentmodule:: qiskit_experiments.analysis

Helper functions for experiment data analysis


Curve Fitting
=============
.. autosummary::
    :toctree: ../stubs/

    curve_fit
    multi_curve_fit
    process_curve_data
    process_multi_curve_data

Plotting
========
.. autosummary::
    :toctree: ../stubs/

    plot_curve_fit
    plot_errorbar
    plot_scatter
"""

from .curve_fitting import curve_fit, multi_curve_fit, process_curve_data, process_multi_curve_data
from .plotting import plot_curve_fit, plot_errorbar, plot_scatter
