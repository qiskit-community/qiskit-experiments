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

    This package is still under active development, there will be breaking
    API changes, and re-organization of the package layout. If you
    encounter any bugs please open an issue on
    `Github <https://github.com/Qiskit/qiskit-experiments/issues>`_

Modules
=======

Experiment Library
******************

The :mod:`qiskit_experiments.library` module contains a list of available
experiments.

Experiment Utility Modules
--------------------------

Certain experiments also have additional utilities contained which can be
accessed by importing the following modules.

- :mod:`qiskit_experiments.library.calibration`
- :mod:`qiskit_experiments.library.characterization`
- :mod:`qiskit_experiments.library.composite`
- :mod:`qiskit_experiments.library.randomized_benchmarking`
- :mod:`qiskit_experiments.library.tomography`

Analysis
********

This :mod:`qiskit_experiments.analysis` module contains utility functions for
analysis experiment data.

Data Processing
***************

This :mod:`qiskit_experiments.data_processing` module contains tools for processing
experiment measurement data.

Calibration Management
**********************

This :mod:`qiskit_experiments.calibration_management` module contains classes
for managing calibration experiment result data.

Database Service
****************

This :mod:`qiskit_experiments.database_service` module contains classes for saving
and retrieving experiment and analysis results from a database.

Experiment Data Classes
=======================

These container classes store the data and results from running experiments

.. autosummary::
    :toctree: ../stubs/

    ExperimentData

Composite Experment Classes
===========================

.. autosummary::
    :toctree: ../stubs/

    BatchExperiment
    ParallelExperiment

Experiment Base Classes
=======================

Construction of custom experiments should be done by making subclasses of the following
base classes

.. autosummary::
    :toctree: ../stubs/

    BaseExperiment
    BaseAnalysis
"""

from .version import __version__

# Base Classes
from .experiment_data import ExperimentData
from .base_analysis import BaseAnalysis
from .base_experiment import BaseExperiment
from .composite import BatchExperiment, ParallelExperiment

# Modules
from . import library
from . import analysis
from . import calibration_management
from . import composite
from . import data_processing
from . import database_service
