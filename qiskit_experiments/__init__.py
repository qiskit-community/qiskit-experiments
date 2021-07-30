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
==============================================
Qiskit Experiments (:mod:`qiskit_experiments`)
==============================================

.. currentmodule:: qiskit_experiments

.. warning::

    This package is still under active development and it is very likely
    that there will be breaking API changes in future releases.
    If you encounter any bugs please open an issue on
    `Github <https://github.com/Qiskit/qiskit-experiments/issues>`_

Qiskit Experiments provides both a general
:mod:`~qiskit_experiments.framework` for creating and
running experiments through Qiskit and optionally storing results in an
online :mod:`~qiskit_experiments.database_service`, as well as a
:mod:`~qiskit_experiments.library` of standard
quantum characterization, calibration, and verification experiments.


Modules
=======

.. list-table::

    * - :mod:`~qiskit_experiments.library`
      - Library of available experiments.
    * - :mod:`~qiskit_experiments.framework`
      - Core classes for experiments and analysis.
    * - :mod:`~qiskit_experiments.data_processing`
      - Tools for building data processor workflows of experiment
        measurement data.
    * - :mod:`~qiskit_experiments.curve_analysis`
      - Utility functions for curve fitting and analysis.
    * - :mod:`~qiskit_experiments.calibration_management`
      - Classes for managing calibration experiment result data.
    * - :mod:`~qiskit_experiments.database_service`
      - Classes for saving and retrieving experiment and analysis results
        from a database.

Certain experiments also have additional utilities contained which can be
accessed by importing the following modules.

- :mod:`qiskit_experiments.library.calibration`
- :mod:`qiskit_experiments.library.characterization`
- :mod:`qiskit_experiments.library.randomized_benchmarking`
- :mod:`qiskit_experiments.library.tomography`
"""

from .version import __version__

# Modules
from . import framework
from . import library
from . import curve_analysis
from . import calibration_management
from . import data_processing
from . import database_service
