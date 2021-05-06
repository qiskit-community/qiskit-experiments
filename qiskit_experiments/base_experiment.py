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
from typing import Union, Iterable, Optional, Tuple, List
import copy
from numbers import Integral
from typing import List, Optional, Iterable, Tuple, Union

from qiskit import transpile, assemble, QuantumCircuit
from qiskit.providers.options import Options
from qiskit.providers.backend import Backend
from qiskit.providers.basebackend import BaseBackend as LegacyBackend
from qiskit.exceptions import QiskitError
from qiskit.providers.backend import Backend
from qiskit.providers.basebackend import BaseBackend as LegacyBackend

from .experiment_data import ExperimentData


class BaseExperiment(ABC):
    """Base Experiment class

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

    def __init__(self, qubits: Iterable[int], experiment_type: Optional[str] = None, **options):
        """Initialize the experiment object.

        Args:
            qubits: the number of qubits or list of physical qubits for
                    the experiment.
            experiment_type: Optional, the experiment type string.
            options: kwarg options for experiment circuits.

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
                print(self._num_qubits, self._physical_qubits)
                raise QiskitError("Duplicate qubits in physical qubits list.")

        # Experiment options
        self._options = self._default_options()
        self.set_options(**options)

        # Execution and analysis options
        self._transpile_options = self._default_transpile_options()
        self._backend_options = self._default_backend_options()
        self._analysis_options = self._default_analysis_options()

        # Set initial layout from qubits
        self._transpile_options.initial_layout = self._physical_qubits

    def run(
        self,
        backend: Backend,
        analysis: bool = True,
        experiment_data: Optional[ExperimentData] = None,
        **kwargs,
    ) -> ExperimentData:
        """Run an experiment and perform analysis.

        Args:
            backend: The backend to run the experiment on.
            analysis: If True run analysis on the experiment data.
            experiment_data: Optional, add results to existing
                experiment data. If None a new ExperimentData object will be
                returned.
            kwargs: runtime keyword arguments for backend.run.

        Returns:
            The experiment data object.
        """
        # Create new experiment data
        if experiment_data is None:
            experiment_data = self.__experiment_data__(self, backend=backend)

        # Generate and transpile circuits
        circuits = self._transpile(
            self.circuits(backend), backend, **self.transpile_options.__dict__
        )

        # Run circuits on backend
        run_options = copy.copy(self.backend_options)
        run_options.update_options(**kwargs)
        run_options = run_options.__dict__

        if isinstance(backend, LegacyBackend):
            qobj = assemble(circuits, backend=backend, **run_options)
            job = backend.run(qobj)
        else:
            job = backend.run(circuits, **run_options)

        # Add Job to ExperimentData
        experiment_data.add_data(job)

        # Queue analysis of data for when job is finished
        if analysis and self.__analysis_class__ is not None:
            # pylint: disable = not-callable
            self.analysis(**self.analysis_options.__dict__).run(experiment_data)

        # Return the ExperimentData future
        return experiment_data

    @property
    def num_qubits(self) -> int:
        """Return the number of qubits for this experiment."""
        return self._num_qubits

    @property
    def physical_qubits(self) -> Tuple[int]:
        """Return the physical qubits for this experiment."""
        return self._physical_qubits

    @classmethod
    def analysis(cls, **analysis_options) -> "BaseAnalysis":
        """Return the default Analysis class for the experiment."""
        if cls.__analysis_class__ is None:
            raise QiskitError(f"Experiment {cls.__name__} does not have a default Analysis class")
        # pylint: disable = not-callable
        return cls.__analysis_class__(**analysis_options)

    @abstractmethod
    def circuits(self, backend: Optional[Backend] = None) -> List[QuantumCircuit]:
        """Return a list of experiment circuits.

        Args:
            backend: Optional, a backend object.

        Returns:
            A list of :class:`QuantumCircuit`s.

        .. note::
            These circuits should be on qubits ``[0, .., N-1]`` for an
            *N*-qubit experiment. The circuits mapped to physical qubits
            are obtained via the :meth:`transpiled_circuits` method.
        """
        # NOTE: Subclasses should override this method using the `options`
        # values for any explicit experiment options that effect circuit
        # generation

    def _transpile(
        self,
        circuits: Union[QuantumCircuit, List[QuantumCircuit]],
        backend: Optional[Backend] = None,
        **transpile_options,
    ) -> List[QuantumCircuit]:
        """Custom transpilation of circuits for running on backend.

        Subclasses may modify this method if they need to customize how
        transpilation is done, for example to update metadata in the
        transpiled circuits.
        """
        return transpile(circuits, backend=backend, **transpile_options)

    @classmethod
    def _default_options(cls) -> Options:
        """Default kwarg options for experiment"""
        # Experiment subclasses should override this method to return
        # an `Options` object containing all the supported options for
        # that experiment and their default values. Only options listed
        # here can be modified later by the `set_options` method.
        return Options()

    @property
    def options(self) -> Options:
        """Return the options for the experiment."""
        return self._options

    def set_options(self, **fields):
        """Set the experiment options.

        Args:
            fields: The fields to update the options

        Raises:
            AttributeError: If the field passed in is not a supported options
        """
        for field in fields:
            if not hasattr(self._options, field):
                raise AttributeError(
                    f"Options field {field} is not valid for {type(self).__name__}"
                )
        self._options.update_options(**fields)

    @classmethod
    def _default_transpile_options(cls) -> Options:
        """Default transpiler options for transpilation of circuits"""
        # Experiment subclasses can override this method if they need
        # to set specific transpiler options defaults for running the
        # experiment.
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
    def _default_backend_options(cls) -> Options:
        """Default backend options for running experiment"""
        return Options()

    @property
    def backend_options(self) -> Options:
        """Return the backend options for the :meth:`run` method."""
        return self._backend_options

    def set_backend_options(self, **fields):
        """Set the backend options for the :meth:`run` method.

        Args:
            fields: The fields to update the options
        """
        self._backend_options.update_options(**fields)

    @classmethod
    def _default_analysis_options(cls) -> Options:
        """Default options for analysis of experiment results."""
        # Experiment subclasses can override this method if they need
        # to set specific analysis options defaults that are different
        # from the Analysis subclass `_default_options` values.
        if cls.__analysis_class__:
            return cls.__analysis_class__._default_options()
        return None

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
