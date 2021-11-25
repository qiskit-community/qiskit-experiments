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
import dataclasses
from collections import OrderedDict
from typing import Sequence, Optional, Tuple, List, Dict, Union, Any

from qiskit import transpile, assemble, QuantumCircuit
from qiskit.test.mock import FakeBackend

from qiskit.providers import BaseJob
from qiskit.providers import Backend, BaseBackend
from qiskit.providers.basebackend import BaseBackend as LegacyBackend
from qiskit.exceptions import QiskitError
from qiskit.qobj.utils import MeasLevel
from qiskit.providers.options import Options
from qiskit_experiments.framework.settings import Settings
from qiskit_experiments.framework.experiment_data import ExperimentData
from qiskit_experiments.version import __version__


@dataclasses.dataclass(frozen=True)
class ExperimentConfig:
    """Store configuration settings for an Experiment

    This stores the current configuration of a
    :class:~qiskit_experiments.framework.BaseExperiment` and
    can be used to reconstruct the experiment using either the
    :meth:`experiment` property if the experiment class type is
    currently stored, or the
    :meth:~qiskit_experiments.framework.BaseExperiment.from_config`
    class method of the appropriate experiment.
    """

    cls: type = None
    args: Tuple[Any] = dataclasses.field(default_factory=tuple)
    kwargs: Dict[str, Any] = dataclasses.field(default_factory=dict)
    experiment_options: Dict[str, Any] = dataclasses.field(default_factory=dict)
    transpile_options: Dict[str, Any] = dataclasses.field(default_factory=dict)
    run_options: Dict[str, Any] = dataclasses.field(default_factory=dict)
    version: str = __version__

    @property
    def experiment(self) -> "BaseExperiment":
        """Return the experiment constructed from this config.

        Returns:
            The experiment reconstructed from the config.

        Raises:
            QiskitError: if the experiment class is not stored,
                         was not successful deserialized, or reconstruction
                         of the experiment fails.
        """
        cls = self.cls
        if cls is None:
            raise QiskitError("No experiment class in experiment config")
        if isinstance(cls, dict):
            raise QiskitError(
                "Unable to load experiment class. Try manually loading "
                "experiment using `Experiment.from_config(config)` instead."
            )
        try:
            return cls.from_config(self)
        except Exception as ex:
            msg = "Unable to construct experiments from config."
            if cls.version != __version__:
                msg += (
                    f" Note that config version ({cls.version}) differs from the current"
                    f" qiskit-experiments version ({__version__}). You could try"
                    " installing a compatible qiskit-experiments version."
                )
            raise QiskitError("{}\nError Message:\n{}".format(msg, str(ex))) from ex


class BaseExperiment(ABC, Settings):
    """Abstract base class for experiments.

    Class Attributes:

        __analysis_class__: Optional, the default Analysis class to use for
                            data analysis. If None no data analysis will be
                            done on experiment data (Default: None).
    """

    # Analysis class for experiment
    __analysis_class__ = None

    def __init__(
        self,
        qubits: Sequence[int],
        backend: Optional[Backend] = None,
        experiment_type: Optional[str] = None,
    ):
        """Initialize the experiment object.

        Args:
            qubits: list of physical qubits for the experiment.
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
        self._analysis_options = self._default_analysis_options()

        # Store keys of non-default options
        self._set_experiment_options = set()
        self._set_transpile_options = set()
        self._set_run_options = set()
        self._set_analysis_options = set()

        # Set backend
        # This should be called last incase `_set_backend` access any of the
        # attributes created during initialization
        self._backend = None
        if isinstance(backend, (Backend, BaseBackend)):
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
    def backend(self) -> Union[Backend, None]:
        """Return the backend for the experiment"""
        return self._backend

    @backend.setter
    def backend(self, backend: Union[Backend, None]) -> None:
        """Set the backend for the experiment"""
        self._set_backend(backend)

    def _set_backend(self, backend: Backend):
        """Set the backend for the experiment.

        Subclasses can override this method to extract additional
        properties from the supplied backend if required.
        """
        self._backend = backend

    def copy(self) -> "BaseExperiment":
        """Return a copy of the experiment"""
        # We want to avoid a deep copy be default for performance so we
        # need to also copy the Options structures so that if they are
        # updated on the copy they don't effect the original.
        ret = copy.copy(self)
        ret._experiment_options = copy.copy(self._experiment_options)
        ret._run_options = copy.copy(self._run_options)
        ret._transpile_options = copy.copy(self._transpile_options)
        ret._analysis_options = copy.copy(self._analysis_options)
        return ret

    @property
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
        analysis: bool = True,
        **run_options,
    ) -> ExperimentData:
        """Run an experiment and perform analysis.

        Args:
            backend: Optional, the backend to run the experiment on. This
                     will override any currently set backends for the single
                     execution.
            analysis: If True run analysis on the experiment data.
            run_options: backend runtime options used for circuit execution.

        Returns:
            The experiment data object.

        Raises:
            QiskitError: if experiment is run with an incompatible existing
                         ExperimentData container.
        """
        if backend is None:
            experiment = self
        else:
            experiment = self.copy()
            experiment._set_backend(backend)
        if experiment.backend is None:
            raise QiskitError("Cannot run experiment, no backend has been set.")

        # Initialize result container
        experiment_data = experiment._initialize_experiment_data()

        # Run options
        run_opts = copy.copy(experiment.run_options)
        run_opts.update_options(**run_options)
        run_opts = run_opts.__dict__

        # Generate and transpile circuits
        transpile_opts = copy.copy(experiment.transpile_options.__dict__)
        transpile_opts["initial_layout"] = list(experiment.physical_qubits)

        circuits = experiment.circuits()

        # Scheduling parameters
        if (
            experiment.backend
            and not experiment.backend.configuration().simulator
            and not isinstance(experiment.backend, FakeBackend)
        ):
            updated_schedule = False
            for circ in circuits:
                for op, _, _ in circ.data:
                    if op.name == "delay":
                        timing_constraints = transpile_opts.get("timing_constraints", {})
                        if "acquire_alignment" not in timing_constraints:
                            timing_constraints["acquire_alignment"] = 16
                        scheduling_method = transpile_opts.get("scheduling_method", "alap")
                        transpile_opts["timing_constraints"] = timing_constraints
                        transpile_opts["scheduling_method"] = scheduling_method
                        updated_schedule = True
                        break
                if updated_schedule:
                    break

        circuits = transpile(circuits, experiment.backend, **transpile_opts)
        experiment._postprocess_transpiled_circuits(circuits, **run_options)

        # Run jobs
        jobs = experiment._run_jobs(circuits, **run_opts)
        experiment_data.add_data(jobs)
        experiment._add_job_metadata(experiment_data.metadata, jobs, **run_opts)

        # Optionally run analysis
        if analysis and self.__analysis_class__ is not None:
            return self.run_analysis(experiment_data)
        else:
            return experiment_data

    def _initialize_experiment_data(self) -> ExperimentData:
        """Initialize the return data container for the experiment run"""
        return ExperimentData(experiment=self)

    def run_analysis(
        self, experiment_data: ExperimentData, replace_results: bool = False, **options
    ) -> ExperimentData:
        """Run analysis and update ExperimentData with analysis result.

        See :meth:`BaseAnalysis.run` for additional information.

        Args:
            experiment_data: the experiment data to analyze.
            replace_results: if True clear any existing analysis results and
                             figures in the experiment data and replace with
                             new results.
            options: additional analysis options. Any values set here will
                     override the value from :meth:`analysis_options`
                     for the current run.

        Returns:
            An experiment data object containing the analysis results and figures.

        Raises:
            QiskitError: if experiment_data container is not valid for analysis.
        """
        # Get analysis options
        analysis_options = copy.copy(self.analysis_options)
        analysis_options.update_options(**options)
        analysis_options = analysis_options.__dict__

        # Run analysis
        analysis = self.analysis()
        return analysis.run(experiment_data, replace_results=replace_results, **analysis_options)

    def _run_jobs(self, circuits: List[QuantumCircuit], **run_options) -> List[BaseJob]:
        """Run circuits on backend as 1 or more jobs."""
        # Run experiment jobs
        max_experiments = getattr(self.backend.configuration(), "max_experiments", None)
        if max_experiments and len(circuits) > max_experiments:
            # Split jobs for backends that have a maximum job size
            job_circuits = [
                circuits[i : i + max_experiments] for i in range(0, len(circuits), max_experiments)
            ]
        else:
            # Run as single job
            job_circuits = [circuits]

        # Run jobs
        jobs = []
        for circs in job_circuits:
            if isinstance(self.backend, LegacyBackend):
                qobj = assemble(circs, backend=self.backend, **run_options)
                job = self.backend.run(qobj)
            else:
                job = self.backend.run(circs, **run_options)
            jobs.append(job)
        return jobs

    @classmethod
    def analysis(cls):
        """Return the default Analysis class for the experiment."""
        if cls.__analysis_class__ is None:
            raise QiskitError(f"Experiment {cls.__name__} does not have a default Analysis class")
        # pylint: disable = not-callable
        return cls.__analysis_class__()

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
        # values for any explicit experiment options that effect circuit
        # generation

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

    @classmethod
    def _default_analysis_options(cls) -> Options:
        """Default options for analysis of experiment results."""
        # Experiment subclasses can override this method if they need
        # to set specific analysis options defaults that are different
        # from the Analysis subclass `_default_options` values.
        if cls.__analysis_class__:
            return cls.__analysis_class__._default_options()
        return Options()

    @property
    def analysis_options(self) -> Options:
        """Return the analysis options for :meth:`run` analysis."""
        return self._analysis_options

    def set_analysis_options(self, **fields):
        """Set the analysis options for :meth:`run` method.

        Args:
            fields: The fields to update the options
        """
        self._analysis_options.update_options(**fields)
        self._set_analysis_options = self._set_analysis_options.union(fields)

    def _postprocess_transpiled_circuits(self, circuits: List[QuantumCircuit], **run_options):
        """Additional post-processing of transpiled circuits before running on backend"""
        pass

    def _metadata(self) -> Dict[str, any]:
        """Return experiment metadata for ExperimentData.

        The :meth:`_add_job_metadata` method will be called for each
        experiment execution to append job metadata, including current
        option values, to the ``job_metadata`` list.
        """
        metadata = {
            "experiment_type": self._type,
            "num_qubits": self.num_qubits,
            "physical_qubits": list(self.physical_qubits),
        }
        # Add additional metadata if subclasses specify it
        for key, val in self._additional_metadata().items():
            metadata[key] = val
        return metadata

    def _additional_metadata(self) -> Dict[str, any]:
        """Add additional subclass experiment metadata.

        Subclasses can override this method if it is necessary to store
        additional experiment metadata in ExperimentData.
        """
        return {}

    def _add_job_metadata(self, metadata: Dict[str, Any], jobs: BaseJob, **run_options):
        """Add runtime job metadata to ExperimentData.

        Args:
            metadata: the metadata dict to update with job data.
            jobs: the job objects.
            run_options: backend run options for the job.
        """
        metadata["job_metadata"] = [
            {
                "job_ids": [job.job_id() for job in jobs],
                "experiment_options": copy.copy(self.experiment_options.__dict__),
                "transpile_options": copy.copy(self.transpile_options.__dict__),
                "analysis_options": copy.copy(self.analysis_options.__dict__),
                "run_options": copy.copy(run_options),
            }
        ]

    def __json_encode__(self):
        """Convert to format that can be JSON serialized"""
        return self.config

    @classmethod
    def __json_decode__(cls, value):
        """Load from JSON compatible format"""
        return cls.from_config(value)
