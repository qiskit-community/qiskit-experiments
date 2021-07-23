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
==========================================================
Experiment Framework (:mod:`qiskit_experiments.framework`)
==========================================================

.. currentmodule:: qiskit_experiments.framework

.. note::

    This page provides useful information for developers to implement new
    experiments.

Classes
=======

Experiment Data Classes
***********************
.. autosummary::
    :toctree: ../stubs/

    AnalysisResult
    ExperimentData

Composite Experiment Classes
****************************
.. autosummary::
    :toctree: ../stubs/

    ParallelExperiment
    BatchExperiment
    CompositeAnalysis
    CompositeExperimentData

Base Classes
************

.. autosummary::
    :toctree: ../stubs/

    BaseExperiment
    BaseAnalysis


Creating Experiments
====================

Experiments and analysis of experiment data is done by subclassing the
:class:`BaseExperiment` and :class:`BaseAnalysis` classes.

Experiment Subclasses
*********************

To create an experiment subclass

- Implement the abstract :meth:`BaseExperiment.circuits` method.
  This should return a list of ``QuantumCircuit`` objects defining
  the experiment payload.
- Call the :meth:`BaseExperiment.__init__` method during the subclass
  constructor with a list of physical qubits. The length of this list must
  be equal to the number of qubits in each circuit and is used to map these
  circuits to this layout during execution.
  Arguments in the constructor can be overridden so that a subclass can
  be initialized with some experiment configuration.
- Set :attr:`BaseExperiment.__analysis_class__` class attribute to
  specify the :class:`BaseAnalysis` subclass for analyzing result data.

Optionally the following methods can also be overridden in the subclass to
allow configuring various experiment and execution options

- :meth:`BaseExperiment._default_experiment_options`
  to set default values for configurable option parameters for the experiment.
- :meth:`BaseExperiment._default_transpile_options`
  to set custom default values for the ``qiskit.transpile`` used to
  transpile the generated circuits before execution.
- :meth:`BaseExperiment._default_run_options`
  to set default backend options for running the transpiled circuits on a backend.
- :meth:`BaseExperiment._default_analysis_options`
  to set default values for configurable options for the experiments analysis class.
  Note that these should generally be set by overriding the :class:`BaseAnalysis`
  method :meth:`BaseAnalysis._default_options` instead of this method except in the
  case where the experiment requires different defaults to the used analysis class.
- :meth:`BaseExperiment._post_process_transpiled_circuits`
  to implement any post-processing of the transpiled circuits before execution.
- :meth:`BaseExperiment._additional_metadata`
  to add any experiment metadata to the result data.

Analysis Subclasses
*******************

To create an analysis subclass one only needs to implement the abstract
:meth:`BaseAnalysis._run_analysis` method. This method takes a
:class:`ExperimentData` container and kwarg analysis options. If any
kwargs are used the :meth:`BaseAnalysis._default_options` method should be
overriden to define default values for these options.

The :meth:`BaseAnalysis._run_analysis` method should return a pair
``(result_data, figures)`` where ``result_data`` is a list of
:class:`AnalysisResult` and ``figures`` is a list of
:class:`matplotlib.figure.Figure`.

The :mod:`qiskit_experiments.data_processing` module contains classes for
building data processor workflows to help with advanced analysis of
experiment data.
"""
from qiskit.providers.options import Options
from .base_analysis import BaseAnalysis
from .base_experiment import BaseExperiment
from .analysis_result import AnalysisResult
from .experiment_data import ExperimentData
from .composite import (
    ParallelExperiment,
    BatchExperiment,
    CompositeAnalysis,
    CompositeExperimentData,
)
