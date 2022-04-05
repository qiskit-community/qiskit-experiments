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

:class:`DbExperimentDataV1` is the main class that defines the structure of
experiment data, which consists of the following:

    * Results from circuit execution, which is called ``data`` in this class.
      The :meth:`DbExperimentDataV1.add_data`
      method allows you to add circuit jobs and job results. If jobs are added,
      the method asynchronously waits for them to finish and extracts job results.
      :meth:`DbExperimentDataV1.data` can then be used to retrieve this data.
      Note that this data is not saved in the database. It is included in this
      class only for convenience.

    * Experiment metadata. This is a freeform keyword-value dictionary. You can
      use this to save extra information, such as the physical qubits the experiment
      operated on, in the database. :meth:`DbExperimentDataV1.set_metadata` and
      :meth:`DbExperimentDataV1.metadata` are methods to set and retrieve metadata,
      respectively.

    * Analysis results. It is likely that some analysis is to be done on the
      experiment data once the circuit jobs finish, and the result of this
      analysis can be stored in the database. Similar to ``DbExperimentDataV1``,
      :class:`DbAnalysisResultV1` defines the data structure of an analysis
      result and provides methods to interface with the database. Being a separate
      class, :class:`DbAnalysisResultV1` allows you to modify an analysis result
      without modifying the experiment data.

    * Figures. Some analysis functions also generate figures, which can also be
      saved in the database.

:class:`DatabaseServiceV1` provides low-level abstract interface for accessing the
database, such as :meth:`DatabaseServiceV1.create_experiment` for creating a
new experiment entry and :meth:`DatabaseServiceV1.update_experiment` for
updating an existing entry. :class:`DbExperimentDataV1` has methods that wrap
around some of these low-level database methods. For example,
:meth:`DbExperimentDataV1.save` calls :meth:`DatabaseServiceV1.create_experiment`
under the cover to save experiment related data. The low-level methods are only
expected to be used when you want to interact with the database directly - for
example, to retrieve a saved analysis result.

Currently only IBM Quantum provides this database service. See
the :mod:`Experiment <qiskit.providers.ibmq.experiment>` module in the IBM
Quantum provider for more details.

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
