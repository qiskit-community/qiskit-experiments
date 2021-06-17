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
===================================================
Stored Data (:mod:`qiskit_experiments.stored_data`)
===================================================

.. currentmodule:: qiskit_experiments.stored_data

This module contains the classes used to define the data structure of
an experiment, including its data, metadata, analysis results, and figures.
The classes also provide an interface with a database service for storing
and retrieving experiment-related data.

Classes
=======

.. autosummary::
   :toctree: ../stubs/

   StoredData
   StoredDataV1
   AnalysisResult
   AnalysisResultV1
   ExperimentService
   ExperimentServiceV1


Exceptions
==========

.. autosummary::
   :toctree: ../stubs/

   ExperimentError
   ExperimentEntryNotFound
   ExperimentEntryExists
"""

from .stored_data import StoredData, StoredDataV1
from .analysis_result import AnalysisResult, AnalysisResultV1
from .experiment_service import ExperimentService, ExperimentServiceV1
from .exceptions import ExperimentError, ExperimentEntryExists, ExperimentEntryNotFound
