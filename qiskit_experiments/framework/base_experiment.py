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
from numbers import Integral
from typing import Sequence, Optional, Tuple, List, Dict, Union

from qiskit import transpile, assemble, QuantumCircuit
from qiskit.providers import BaseJob
from qiskit.providers.backend import Backend
from qiskit.providers.basebackend import BaseBackend as LegacyBackend
from qiskit.exceptions import QiskitError
from qiskit.qobj.utils import MeasLevel
from qiskit_experiments.framework import Options
from qiskit_experiments.framework.experiment_data import ExperimentData


class BaseExperiment(ABC):
    """Abstract base class for experiments.

    Class Attributes:

        __analysis_class__: Optional, the default Analysis class to use for
                            data analysis. If None no data analysis will be
                            done on experiment data (Default: None).
        __experiment_data__: ExperimentData class that is produced by the
                             experiment (Default: ExperimentData).
    """

    # Analysis class for experiment
    __analysis_class__ = None

    # ExperimentData class for experiment
    __experiment_data__ = ExperimentData

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

        # Backend
        self._backend = None
        if backend is not None:
            self._set_backend(backend)

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
        circuits = transpile(experiment.circuits(), experiment.backend, **transpile_opts)
        experiment._postprocess_transpiled_circuits(circuits, **run_options)

        # Run jobs
        jobs = experiment._run_jobs(circuits, **run_opts)
        experiment_data.add_data(jobs)
        experiment._add_job_metadata(experiment_data, jobs, **run_opts)

        # Optionally run analysis
        if analysis and self.__analysis_class__ is not None:
            experiment_data.add_analysis_callback(self.run_analysis)

        # Return the ExperimentData future
        return experiment_data

    def _initialize_experiment_data(self) -> ExperimentData:
        """Initialize the return data container for the experiment run"""
        return self.__experiment_data__(experiment=self)

    def run_analysis(self, experiment_data: ExperimentData, **options) -> ExperimentData:
        """Run analysis and update ExperimentData with analysis result.

        Args:
            experiment_data: the experiment data to analyze.
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
        analysis.run(experiment_data, **analysis_options)
        return experiment_data

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
            "job_metadata": [],
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

    def _add_job_metadata(self, experiment_data: ExperimentData, jobs: BaseJob, **run_options):
        """Add runtime job metadata to ExperimentData.

        Args:
            experiment_data: the experiment data container.
            jobs: the job objects.
            run_options: backend run options for the job.
        """
        metadata = {
            "job_ids": [job.job_id() for job in jobs],
            "experiment_options": copy.copy(self.experiment_options.__dict__),
            "transpile_options": copy.copy(self.transpile_options.__dict__),
            "analysis_options": copy.copy(self.analysis_options.__dict__),
            "run_options": copy.copy(run_options),
        }
        experiment_data._metadata["job_metadata"].append(metadata)
