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
=============================================================
Database Service (:mod:`qiskit_experiments.database_service`)
=============================================================

.. currentmodule:: qiskit_experiments.database_service

This subpackage contains classes used to define the data structure of
an experiment, including its data, metadata, analysis results, and figures, as
well as the interface to an experiment database service. An experiment database
service allows one to store, retrieve, and query experiment related data.

Classes
=======

.. autosummary::
   :toctree: ../stubs/

   DbExperimentData
   DbExperimentDataV1
   DbAnalysisResult
   DbAnalysisResultV1
   DatabaseService
   DatabaseServiceV1


Exceptions
==========

.. autosummary::
   :toctree: ../stubs/

   DbExperimentDataError
   DbExperimentEntryExists
   DbExperimentEntryNotFound
"""

from .db_experiment_data import DbExperimentData, DbExperimentDataV1
from .db_analysis_result import DbAnalysisResult, DbAnalysisResultV1
from .database_service import DatabaseService, DatabaseServiceV1
from .exceptions import DbExperimentDataError, DbExperimentEntryExists, DbExperimentEntryNotFound
