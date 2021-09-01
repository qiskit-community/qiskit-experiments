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
from typing import Iterable, Optional, Tuple, List, Dict
import copy
from numbers import Integral

from qiskit import transpile, assemble, QuantumCircuit
from qiskit.providers import BaseJob
from qiskit.providers.backend import Backend
from qiskit.providers.basebackend import BaseBackend as LegacyBackend
from qiskit.test.mock import FakeBackend
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

    def __init__(self, qubits: Iterable[int], experiment_type: Optional[str] = None):
        """Initialize the experiment object.

        Args:
            qubits: the number of qubits or list of physical qubits for
                    the experiment.
            experiment_type: Optional, the experiment type string.

        Raises:
            QiskitError: if qubits is a list and contains duplicates.
        """
        # Experiment identification metadata
        self._type = experiment_type if experiment_type else type(self).__name__

        # Circuit parameters
        if isinstance(qubits, Integral):
            self._num_qubits = qubits
            self._physical_qubits = tuple(range(qubits))
        else:
            self._num_qubits = len(qubits)
            self._physical_qubits = tuple(qubits)
            if self._num_qubits != len(set(self._physical_qubits)):
                raise QiskitError("Duplicate qubits in physical qubits list.")

        # Experiment options
        self._experiment_options = self._default_experiment_options()
        self._transpile_options = self._default_transpile_options()
        self._run_options = self._default_run_options()
        self._analysis_options = self._default_analysis_options()

    def run(
        self,
        backend: Backend,
        analysis: bool = True,
        experiment_data: Optional[ExperimentData] = None,
        **run_options,
    ) -> ExperimentData:
        """Run an experiment and perform analysis.

        Args:
            backend: The backend to run the experiment on.
            analysis: If True run analysis on the experiment data.
            experiment_data: Optional, add results to existing
                experiment data. If None a new ExperimentData object will be
                returned.
            run_options: backend runtime options used for circuit execution.

        Returns:
            The experiment data object.

        Raises:
            QiskitError: if experiment is run with an incompatible existing
                         ExperimentData container.
        """
        # Create experiment data container
        experiment_data = self._initialize_experiment_data(backend, experiment_data)

        # Run options
        run_opts = copy.copy(self.run_options)
        run_opts.update_options(**run_options)
        run_opts = run_opts.__dict__

        # Generate and transpile circuits
        circuits = self.run_transpile(backend)

        # Execute experiment
        if isinstance(backend, LegacyBackend):
            qobj = assemble(circuits, backend=backend, **run_opts)
            job = backend.run(qobj)
        else:
            job = backend.run(circuits, **run_opts)

        # Add experiment option metadata
        self._add_job_metadata(experiment_data, job, **run_opts)

        # Run analysis
        if analysis:
            experiment_data = self.run_analysis(experiment_data, job)
        else:
            experiment_data.add_data(job)

        return experiment_data

    def _initialize_experiment_data(
        self, backend: Backend, experiment_data: Optional[ExperimentData] = None
    ) -> ExperimentData:
        """Initialize the return data container for the experiment run"""
        if experiment_data is None:
            return self.__experiment_data__(experiment=self, backend=backend)

        # Validate experiment is compatible with existing data
        if not isinstance(experiment_data, ExperimentData):
            raise QiskitError("Input `experiment_data` is not a valid ExperimentData.")
        if experiment_data.experiment_type != self._type:
            raise QiskitError("Existing ExperimentData contains data from a different experiment.")
        if experiment_data.metadata.get("physical_qubits") != list(self.physical_qubits):
            raise QiskitError(
                "Existing ExperimentData contains data for a different set of physical qubits."
            )

        return experiment_data._copy_metadata()

    def _pre_transpile_hook(self, backend: Backend):
        """An extra transpile subroutine executed before transpilation.

        Args:
            backend: Target backend.
        """
        pass

    def _post_transpile_hook(
            self, circuits: List[QuantumCircuit], backend: Backend
    ) -> List[QuantumCircuit]:
        """An extra transpile subroutine executed after transpilation.

        Args:
            circuits: List of transpiled circuits.
            backend: Target backend.

        Returns:
            List of circuits to execute.
        """
        return circuits

    def run_transpile(self, backend: Backend, **options):
        """Run transpile and returns transpiled circuits.

        Args:
            backend: Target backend.

        Returns:
            Transpiled circuit to execute.
        """
        # Run pre transpile. This is implemented by each experiment subclass.
        self._pre_transpile_hook(backend)

        # Get transpile options
        transpile_options = copy.copy(self.transpile_options)
        transpile_options.update_options(
            initial_layout=list(self._physical_qubits),
            **options,
        )
        transpile_options = transpile_options.__dict__

        circuits = transpile(circuits=self.circuits(backend), backend=backend, **transpile_options)

        # Run post transpile. This is implemented by each experiment subclass.
        circuits = self._post_transpile_hook(circuits, backend)

        return circuits

    def _post_analysis_hook(self, experiment_data: ExperimentData):
        """An extra analysis subroutine executed after analysis.

        Args:
            experiment_data: A future object of the experiment result.

        Note:
            The experiment_data may contain a future object as an experiment result
            and the previous analysis routine has not completed yet.
            If the hook should be executed immediately, call :meth:`block_for_results` method
            before starting the data processing code.
        """
        pass

    def run_analysis(
            self, experiment_data: ExperimentData, job: BaseJob = None, **options
    ) -> ExperimentData:
        """Run analysis and update ExperimentData with analysis result.

        Args:
            experiment_data: The experiment data to analyze.
            job: The future object of experiment result which is currently running on the backend.
            options: Additional analysis options. Any values set here will
                override the value from :meth:`analysis_options` for the current run.

        Returns:
            An experiment data object containing the analysis results and figures.

        Raises:
            QiskitError: Method is called with an empty experiment result.
        """
        run_analysis = self.analysis().run if self.__analysis_class__ else None

        # Get analysis options
        analysis_options = copy.copy(self.analysis_options)
        analysis_options.update_options(**options)
        analysis_options = analysis_options.__dict__

        if not job and run_analysis is not None:
            # Run analysis immediately
            if not experiment_data.data():
                raise QiskitError(
                    "Experiment data seems to be empty and no running job is provided. "
                    "At least one data entry is required to run analysis."
                )
            experiment_data = run_analysis(experiment_data, **analysis_options)
        else:
            # Run analysis when job is completed
            experiment_data.add_data(
                job, post_processing_callback=run_analysis, **analysis_options
            )

        # Run post analysis. This is implemented by each experiment subclass.
        self._post_analysis_hook(experiment_data)

        return experiment_data

    @property
    def num_qubits(self) -> int:
        """Return the number of qubits for this experiment."""
        return self._num_qubits

    @property
    def physical_qubits(self) -> Tuple[int]:
        """Return the physical qubits for this experiment."""
        return self._physical_qubits

    @property
    def experiment_type(self) -> str:
        """Return experiment type."""
        return self._type

    @classmethod
    def analysis(cls):
        """Return the default Analysis class for the experiment."""
        if cls.__analysis_class__ is None:
            raise QiskitError(f"Experiment {cls.__name__} does not have a default Analysis class")
        # pylint: disable = not-callable
        return cls.__analysis_class__()

    @abstractmethod
    def circuits(self, backend: Optional[Backend] = None) -> List[QuantumCircuit]:
        """Return a list of experiment circuits.

        Args:
            backend: Optional, a backend object.

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
        for key, val in self._additional_metadata():
            metadata[key] = val
        return metadata

    def _additional_metadata(self) -> Dict[str, any]:
        """Add additional subclass experiment metadata.

        Subclasses can override this method if it is necessary to store
        additional experiment metadata in ExperimentData.
        """
        return {}

    def _add_job_metadata(self, experiment_data: ExperimentData, job: BaseJob, **run_options):
        """Add runtime job metadata to ExperimentData.

        Args:
            experiment_data: the experiment data container.
            job: the job object.
            run_options: backend run options for the job.
        """
        metadata = {
            "job_id": job.job_id(),
            "experiment_options": copy.copy(self.experiment_options.__dict__),
            "transpile_options": copy.copy(self.transpile_options.__dict__),
            "analysis_options": copy.copy(self.analysis_options.__dict__),
            "run_options": copy.copy(run_options),
        }
        experiment_data._metadata["job_metadata"].append(metadata)
