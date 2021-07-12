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
=========================================================================
Calibration management (:mod:`qiskit_experiments.calibration.management`)
=========================================================================

.. currentmodule:: qiskit_experiments.calibration.management

Calibration management
======================
.. autosummary::
    :toctree: ../stubs/

    Calibrations
    BackendCalibrations
    ParameterValue

Parameter value updating
========================
.. autosummary::
    :toctree: ../stubs/

    Frequency
    Amplitude
    Drag

"""

from .calibrations import Calibrations
from .backend_calibrations import BackendCalibrations
from .parameter_value import ParameterValue

from .update_library import Frequency, Drag, Amplitude
