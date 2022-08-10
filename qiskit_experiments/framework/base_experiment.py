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
Base Experiment class.
"""

from abc import ABC, abstractmethod
import copy
from collections import OrderedDict
from typing import Sequence, Optional, Tuple, List, Dict, Union
import warnings

from qiskit import transpile, QuantumCircuit
from qiskit.providers import Job, Backend
from qiskit.exceptions import QiskitError
from qiskit.qobj.utils import MeasLevel
from qiskit.providers.options import Options
from qiskit_experiments.framework import BackendData
from qiskit_experiments.framework.store_init_args import StoreInitArgs
from qiskit_experiments.framework.base_analysis import BaseAnalysis
from qiskit_experiments.framework.experiment_data import ExperimentData
from qiskit_experiments.framework.configs import ExperimentConfig


class BaseExperiment(ABC, StoreInitArgs):
    """Abstract base class for experiments."""

    def __init__(
        self,
        qubits: Sequence[int],
        analysis: Optional[BaseAnalysis] = None,
        backend: Optional[Backend] = None,
        experiment_type: Optional[str] = None,
    ):
        """Initialize the experiment object.

        Args:
            qubits: list of physical qubits for the experiment.
            analysis: Optional, the analysis to use for the experiment.
            backend: Optional, the backend to run the experiment on.
            experiment_type: Optional, the experiment type string.

        Raises:
            QiskitError: if qubits contains duplicates.
        """
        # Experiment identification metadata
        self._type = experiment_type if experiment_type else type(self).__name__

        # Circuit parameters
        self._num_qubits = len(qubits)
        self._physical_qubits = tuple(qubits)
        if self._num_qubits != len(set(self._physical_qubits)):
            raise QiskitError("Duplicate qubits in physical qubits list.")

        # Experiment options
        self._experiment_options = self._default_experiment_options()
        self._transpile_options = self._default_transpile_options()
        self._run_options = self._default_run_options()

        # Store keys of non-default options
        self._set_experiment_options = set()
        self._set_transpile_options = set()
        self._set_run_options = set()
        self._set_analysis_options = set()

        # Set analysis
        self._analysis = None
        if analysis:
            self.analysis = analysis
        # TODO: Hack for backwards compatibility with old base class.
        # Remove after updating subclasses
        elif hasattr(self, "__analysis_class__"):
            warnings.warn(
                "Defining a default BaseAnalysis class for an experiment using the "
                "__analysis_class__ attribute is deprecated as of 0.2.0. "
                "Use the `analysis` kwarg of BaseExperiment.__init__ "
                "to specify a default analysis class."
            )
            analysis_cls = getattr(self, "__analysis_class__")
            self.analysis = analysis_cls()  # pylint: disable = not-callable

        # Set backend
        # This should be called last in case `_set_backend` access any of the
        # attributes created during initialization
        self._backend = None
        self._backend_data = None
        if isinstance(backend, Backend):
            self._set_backend(backend)

    @property
    def experiment_type(self) -> str:
        """Return experiment type."""
        return self._type

    @property
    def physical_qubits(self) -> Tuple[int, ...]:
        """Return the device qubits for the experiment."""
        return self._physical_qubits

    @property
    def num_qubits(self) -> int:
        """Return the number of qubits for the experiment."""
        return self._num_qubits

    @property
    def analysis(self) -> Union[BaseAnalysis, None]:
        """Return the analysis instance for the experiment"""
        return self._analysis

    @analysis.setter
    def analysis(self, analysis: Union[BaseAnalysis, None]) -> None:
        """Set the analysis instance for the experiment"""
        if analysis is not None and not isinstance(analysis, BaseAnalysis):
            raise TypeError("Input is not a None or a BaseAnalysis subclass.")
        self._analysis = analysis

    @property
    def backend(self) -> Union[Backend, None]:
        """Return the backend for the experiment"""
        return self._backend

    @backend.setter
    def backend(self, backend: Union[Backend, None]) -> None:
        """Set the backend for the experiment"""
        if not isinstance(backend, Backend):
            raise TypeError("Input is not a backend.")
        self._set_backend(backend)

    def _set_backend(self, backend: Backend):
        """Set the backend for the experiment.

        Subclasses can override this method to extract additional
        properties from the supplied backend if required.
        """
        self._backend = backend
        self._backend_data = BackendData(backend)

    def copy(self) -> "BaseExperiment":
        """Return a copy of the experiment"""
        # We want to avoid a deep copy be default for performance so we
        # need to also copy the Options structures so that if they are
        # updated on the copy they don't effect the original.
        ret = copy.copy(self)
        if self.analysis:
            ret.analysis = self.analysis.copy()

        ret._experiment_options = copy.copy(self._experiment_options)
        ret._run_options = copy.copy(self._run_options)
        ret._transpile_options = copy.copy(self._transpile_options)

        ret._set_experiment_options = copy.copy(self._set_experiment_options)
        ret._set_transpile_options = copy.copy(self._set_transpile_options)
        ret._set_run_options = copy.copy(self._set_run_options)
        return ret

    def config(self) -> ExperimentConfig:
        """Return the config dataclass for this experiment"""
        args = tuple(getattr(self, "__init_args__", OrderedDict()).values())
        kwargs = dict(getattr(self, "__init_kwargs__", OrderedDict()))
        # Only store non-default valued options
        experiment_options = dict(
            (key, getattr(self._experiment_options, key)) for key in self._set_experiment_options
        )
        transpile_options = dict(
            (key, getattr(self._transpile_options, key)) for key in self._set_transpile_options
        )
        run_options = dict((key, getattr(self._run_options, key)) for key in self._set_run_options)
        return ExperimentConfig(
            cls=type(self),
            args=args,
            kwargs=kwargs,
            experiment_options=experiment_options,
            transpile_options=transpile_options,
            run_options=run_options,
        )

    @classmethod
    def from_config(cls, config: Union[ExperimentConfig, Dict]) -> "BaseExperiment":
        """Initialize an experiment from experiment config"""
        if isinstance(config, dict):
            config = ExperimentConfig(**dict)
        ret = cls(*config.args, **config.kwargs)
        if config.experiment_options:
            ret.set_experiment_options(**config.experiment_options)
        if config.transpile_options:
            ret.set_transpile_options(**config.transpile_options)
        if config.run_options:
            ret.set_run_options(**config.run_options)
        return ret

    def run(
        self,
        backend: Optional[Backend] = None,
        analysis: Optional[Union[BaseAnalysis, None]] = "default",
        timeout: Optional[float] = None,
        **run_options,
    ) -> ExperimentData:
        """Run an experiment and perform analysis.

        Args:
            backend: Optional, the backend to run the experiment on. This
                     will override any currently set backends for the single
                     execution.
            analysis: Optional, a custom analysis instance to use for performing
                      analysis. If None analysis will not be run. If ``"default"``
                      the experiments :meth:`analysis` instance will be used if
                      it contains one.
            timeout: Time to wait for experiment jobs to finish running before
                     cancelling.
            run_options: backend runtime options used for circuit execution.

        Returns:
            The experiment data object.

        Raises:
            QiskitError: if experiment is run with an incompatible existing
                         ExperimentData container.
        """

        if backend is not None or analysis != "default" or run_options:
            # Make a copy to update analysis or backend if one is provided at runtime
            experiment = self.copy()
            if backend:
                experiment._set_backend(backend)
            if isinstance(analysis, BaseAnalysis):
                experiment.analysis = analysis
            if run_options:
                experiment.set_run_options(**run_options)
        else:
            experiment = self

        if experiment.backend is None:
            raise QiskitError("Cannot run experiment, no backend has been set.")

        # Finalize experiment before executions
        experiment._finalize()

        # Generate and transpile circuits
        transpiled_circuits = experiment._transpiled_circuits()

        # Initialize result container
        experiment_data = experiment._initialize_experiment_data()

        # Run options
        run_opts = experiment.run_options.__dict__

        # Run jobs
        jobs = experiment._run_jobs(transpiled_circuits, **run_opts)
        experiment_data.add_jobs(jobs, timeout=timeout)

        # Optionally run analysis
        if analysis and experiment.analysis:
            return experiment.analysis.run(experiment_data)
        else:
            return experiment_data

    def _initialize_experiment_data(self) -> ExperimentData:
        """Initialize the return data container for the experiment run"""
        return ExperimentData(experiment=self)

    def _finalize(self):
        """Finalize experiment object before running jobs.

        Subclasses can override this method to set any final option
        values derived from other options or attributes of the
        experiment before `_run` is called.
        """
        pass

    def _run_jobs(self, circuits: List[QuantumCircuit], **run_options) -> List[Job]:
        """Run circuits on backend as 1 or more jobs."""
        # Run experiment jobs
        max_circuits = self._backend_data.max_circuits
        if max_circuits and len(circuits) > max_circuits:
            # Split jobs for backends that have a maximum job size
            job_circuits = [
                circuits[i : i + max_circuits] for i in range(0, len(circuits), max_circuits)
            ]
        else:
            # Run as single job
            job_circuits = [circuits]

        # Run jobs
        jobs = [self.backend.run(circs, **run_options) for circs in job_circuits]

        return jobs

    @abstractmethod
    def circuits(self) -> List[QuantumCircuit]:
        """Return a list of experiment circuits.

        Returns:
            A list of :class:`QuantumCircuit`.

        .. note::
            These circuits should be on qubits ``[0, .., N-1]`` for an
            *N*-qubit experiment. The circuits mapped to physical qubits
            are obtained via the :meth:`transpiled_circuits` method.
        """
        # NOTE: Subclasses should override this method using the `options`
        # values for any explicit experiment options that affect circuit
        # generation

    def _transpiled_circuits(self) -> List[QuantumCircuit]:
        """Return a list of experiment circuits, transpiled.

        This function can be overridden to define custom transpilation.
        """
        transpile_opts = copy.copy(self.transpile_options.__dict__)
        transpile_opts["initial_layout"] = list(self.physical_qubits)
        transpiled = transpile(self.circuits(), self.backend, **transpile_opts)

        return transpiled

    @classmethod
    def _default_experiment_options(cls) -> Options:
        """Default kwarg options for experiment"""
        # Experiment subclasses should override this method to return
        # an `Options` object containing all the supported options for
        # that experiment and their default values. Only options listed
        # here can be modified later by the different methods for
        # setting options.
        return Options()

    @property
    def experiment_options(self) -> Options:
        """Return the options for the experiment."""
        return self._experiment_options

    def set_experiment_options(self, **fields):
        """Set the experiment options.

        Args:
            fields: The fields to update the options

        Raises:
            AttributeError: If the field passed in is not a supported options
        """
        for field in fields:
            if not hasattr(self._experiment_options, field):
                raise AttributeError(
                    f"Options field {field} is not valid for {type(self).__name__}"
                )
        self._experiment_options.update_options(**fields)
        self._set_experiment_options = self._set_experiment_options.union(fields)

    @classmethod
    def _default_transpile_options(cls) -> Options:
        """Default transpiler options for transpilation of circuits"""
        # Experiment subclasses can override this method if they need
        # to set specific default transpiler options to transpile the
        # experiment circuits.
        return Options(optimization_level=0)

    @property
    def transpile_options(self) -> Options:
        """Return the transpiler options for the :meth:`run` method."""
        return self._transpile_options

    def set_transpile_options(self, **fields):
        """Set the transpiler options for :meth:`run` method.

        Args:
            fields: The fields to update the options

        Raises:
            QiskitError: if `initial_layout` is one of the fields.
        """
        if "initial_layout" in fields:
            raise QiskitError(
                "Initial layout cannot be specified as a transpile option"
                " as it is determined by the experiment physical qubits."
            )
        self._transpile_options.update_options(**fields)
        self._set_transpile_options = self._set_transpile_options.union(fields)

    @classmethod
    def _default_run_options(cls) -> Options:
        """Default options values for the experiment :meth:`run` method."""
        return Options(meas_level=MeasLevel.CLASSIFIED)

    @property
    def run_options(self) -> Options:
        """Return options values for the experiment :meth:`run` method."""
        return self._run_options

    def set_run_options(self, **fields):
        """Set options values for the experiment  :meth:`run` method.

        Args:
            fields: The fields to update the options
        """
        self._run_options.update_options(**fields)
        self._set_run_options = self._set_run_options.union(fields)

    @property
    def analysis_options(self) -> Options:
        """Return the analysis options for :meth:`run` analysis.

        .. deprecated:: 0.2.0
            This is replaced by calling ``experiment.analysis.options`` using
            the :meth:`analysis`and :meth:`~qiskit_experiments.framework.BaseAnalysis.options`
            properties.
        """
        warnings.warn(
            "`BaseExperiment.analysis_options` is deprecated as of qiskit-experiments"
            " 0.2.0 and will be removed in the 0.3.0 release."
            " Use `experiment.analysis.options instead",
            DeprecationWarning,
        )
        return self.analysis.options

    def _metadata(self) -> Dict[str, any]:
        """Return experiment metadata for ExperimentData.

        Subclasses can override this method to add custom experiment
        metadata to the returned experiment result data.
        """
        metadata = {"physical_qubits": list(self.physical_qubits)}
        return metadata

    def __json_encode__(self):
        """Convert to format that can be JSON serialized"""
        return self.config()

    @classmethod
    def __json_decode__(cls, value):
        """Load from JSON compatible format"""
        return cls.from_config(value)
