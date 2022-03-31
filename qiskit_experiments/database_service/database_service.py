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

"""Experiment database service abstract interface."""

from abc import ABC, abstractmethod
from typing import Optional, Dict, List, Any, Union, Tuple, Type
import json

from .device_component import DeviceComponent


class DatabaseService:
    """Base common type for all versioned DatabaseService abstract classes.

    Note this class should not be inherited from directly, it is intended
    to be used for type checking. When implementing a subclass you should use
    the versioned abstract classes as the parent class and not this class
    directly.
    """

    version = 0


class DatabaseServiceV1(DatabaseService, ABC):
    """Interface for providing experiment database service.

    This class defines the interface ``qiskit_experiments`` expects from an
    experiment database service.

    An experiment database service allows you to store experiment data and metadata
    in a database. An experiment can have one or more jobs, analysis results,
    and figures.

    Each implementation of this service may use different data structure and
    should issue a warning on unsupported keywords.
    """

    version = 1

    @abstractmethod
    def create_experiment(
        self,
        experiment_type: str,
        backend_name: str,
        metadata: Optional[Dict] = None,
        experiment_id: Optional[str] = None,
        parent_id: Optional[str] = None,
        job_ids: Optional[List[str]] = None,
        tags: Optional[List[str]] = None,
        notes: Optional[str] = None,
        json_encoder: Type[json.JSONEncoder] = json.JSONEncoder,
        **kwargs: Any,
    ) -> str:
        """Create a new experiment in the database.

        Args:
            experiment_type: Experiment type.
            backend_name: Name of the backend the experiment ran on.
            metadata: Experiment metadata.
            experiment_id: Experiment ID. It must be in the ``uuid4`` format.
                One will be generated if not supplied.
            parent_id: The experiment ID of the parent experiment.
                The parent experiment must exist, must be on the same backend as the child,
                and an experiment cannot be its own parent.
            job_ids: IDs of experiment jobs.
            tags: Tags to be associated with the experiment.
            notes: Freeform notes about the experiment.
            json_encoder: Custom JSON encoder to use to encode the experiment.
            kwargs: Additional keywords supported by the service provider.

        Returns:
            Experiment ID.

        Raises:
            DbExperimentEntryExists: If the experiment already exists.
        """
        pass

    @abstractmethod
    def update_experiment(
        self,
        experiment_id: str,
        metadata: Optional[Dict] = None,
        job_ids: Optional[List[str]] = None,
        notes: Optional[str] = None,
        tags: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> None:
        """Update an existing experiment.

        Args:
            experiment_id: Experiment ID.
            metadata: Experiment metadata.
            job_ids: IDs of experiment jobs.
            notes: Freeform notes about the experiment.
            tags: Tags to be associated with the experiment.
            kwargs: Additional keywords supported by the service provider.

        Raises:
            DbExperimentEntryNotFound: If the experiment does not exist.
        """
        pass

    @abstractmethod
    def experiment(
        self, experiment_id: str, json_decoder: Type[json.JSONDecoder] = json.JSONDecoder
    ) -> Dict:
        """Retrieve a previously stored experiment.

        Args:
            experiment_id: Experiment ID.
            json_decoder: Custom JSON decoder to use to decode the retrieved experiment.

        Returns:
            A dictionary containing the retrieved experiment data.

        Raises:
            DbExperimentEntryNotFound: If the experiment does not exist.
        """
        pass

    @abstractmethod
    def experiments(
        self,
        limit: Optional[int] = 10,
        json_decoder: Type[json.JSONDecoder] = json.JSONDecoder,
        device_components: Optional[Union[str, DeviceComponent]] = None,
        experiment_type: Optional[str] = None,
        backend_name: Optional[str] = None,
        tags: Optional[List[str]] = None,
        parent_id: Optional[str] = None,
        tags_operator: Optional[str] = "OR",
        **filters: Any,
    ) -> List[Dict]:
        """Retrieve all experiment data, with optional filtering.

        Args:
            limit: Number of experiment data entries to retrieve. ``None`` means no limit.
            json_decoder: Custom JSON decoder to use to decode the retrieved experiment.
            device_components: Filter by device components. An experiment must have analysis
                results with device components matching the given list exactly to be included.
            experiment_type: Experiment type used for filtering.
            backend_name: Backend name used for filtering.
            tags: Filter by tags assigned to experiments. This can be used
                with `tags_operator` for granular filtering.
            parent_id: Filter by parent experiment ID.
            tags_operator: Logical operator to use when filtering by tags. Valid
                values are "AND" and "OR":

                    * If "AND" is specified, then an experiment must have all of the tags
                      specified in `tags` to be included.
                    * If "OR" is specified, then an experiment only needs to have any
                      of the tags specified in `tags` to be included.

            **filters: Additional filtering keywords supported by the service provider.

        Returns:
            A list of experiments.
        """
        pass

    @abstractmethod
    def delete_experiment(self, experiment_id: str) -> None:
        """Delete an experiment.

        Args:
            experiment_id: Experiment ID.
        """
        pass

    @abstractmethod
    def create_analysis_result(
        self,
        experiment_id: str,
        result_data: Dict,
        result_type: str,
        device_components: Optional[Union[str, DeviceComponent]] = None,
        tags: Optional[List[str]] = None,
        quality: Optional[str] = None,
        verified: bool = False,
        result_id: Optional[str] = None,
        json_encoder: Type[json.JSONEncoder] = json.JSONEncoder,
        **kwargs: Any,
    ) -> str:
        """Create a new analysis result in the database.

        Args:
            experiment_id: ID of the experiment this result is for.
            result_data: Result data to be stored.
            result_type: Analysis result type.
            device_components: Target device components, such as qubits.
            tags: Tags to be associated with the analysis result.
            quality: Quality of this analysis.
            verified: Whether the result quality has been verified.
            result_id: Analysis result ID. It must be in the ``uuid4`` format.
                One will be generated if not supplied.
            json_encoder: Custom JSON encoder to use to encode the analysis result.
            kwargs: Additional keywords supported by the service provider.

        Returns:
            Analysis result ID.

        Raises:
            DbExperimentEntryExists: If the analysis result already exits.
        """
        pass

    @abstractmethod
    def update_analysis_result(
        self,
        result_id: str,
        result_data: Optional[Dict] = None,
        tags: Optional[List[str]] = None,
        quality: Optional[str] = None,
        verified: bool = None,
        **kwargs: Any,
    ) -> None:
        """Update an existing analysis result.

        Args:
            result_id: Analysis result ID.
            result_data: Result data to be stored.
            quality: Quality of this analysis.
            verified: Whether the result quality has been verified.
            tags: Tags to be associated with the analysis result.
            kwargs: Additional keywords supported by the service provider.

        Raises:
            DbExperimentEntryNotFound: If the analysis result does not exist.
        """
        pass

    @abstractmethod
    def analysis_result(
        self, result_id: str, json_decoder: Type[json.JSONDecoder] = json.JSONDecoder
    ) -> Dict:
        """Retrieve a previously stored experiment.

        Args:
            result_id: Analysis result ID.
            json_decoder: Custom JSON decoder to use to decode the retrieved analysis result.

        Returns:
            Retrieved analysis result.

        Raises:
            DbExperimentEntryNotFound: If the analysis result does not exist.
        """
        pass

    @abstractmethod
    def analysis_results(
        self,
        limit: Optional[int] = 10,
        json_decoder: Type[json.JSONDecoder] = json.JSONDecoder,
        device_components: Optional[Union[str, DeviceComponent]] = None,
        experiment_id: Optional[str] = None,
        result_type: Optional[str] = None,
        backend_name: Optional[str] = None,
        quality: Optional[str] = None,
        verified: Optional[bool] = None,
        tags: Optional[List[str]] = None,
        tags_operator: Optional[str] = "OR",
        **filters: Any,
    ) -> List[Dict]:
        """Retrieve all analysis results, with optional filtering.

        Args:
            limit: Number of analysis results to retrieve. ``None`` means no limit.
            json_decoder: Custom JSON decoder to use to decode the retrieved analysis results.
            device_components: Target device components, such as qubits.
            experiment_id: Experiment ID used for filtering.
            result_type: Analysis result type used for filtering.
            backend_name: Backend name used for filtering. If specified, analysis
                results associated with experiments on that backend are returned.
            quality: Quality value used for filtering.
            verified: Whether the result quality has been verified.
            tags: Filter by tags assigned to analysis results. This can be used
                with `tags_operator` for granular filtering.
            tags_operator: Logical operator to use when filtering by tags. Valid
                values are "AND" and "OR":

                    * If "AND" is specified, then an analysis result must have all of the tags
                      specified in `tags` to be included.
                    * If "OR" is specified, then an analysis result only needs to have any
                      of the tags specified in `tags` to be included.

            **filters: Additional filtering keywords supported by the service provider.

        Returns:
            A list of analysis results.
        """
        pass

    @abstractmethod
    def delete_analysis_result(self, result_id: str) -> None:
        """Delete an analysis result.

        Args:
            result_id: Analysis result ID.
        """
        pass

    @abstractmethod
    def create_figure(
        self, experiment_id: str, figure: Union[str, bytes], figure_name: Optional[str]
    ) -> Tuple[str, int]:
        """Store a new figure in the database.

        Args:
            experiment_id: ID of the experiment this figure is for.
            figure: Path of the figure file or figure data to store.
            figure_name: Name of the figure. If ``None``, the figure file name, if
                given, or a generated name is used.

        Returns:
            A tuple of the name and size of the saved figure.

        Raises:
            ExperimentEntryExists: If the figure already exits.
        """
        pass

    @abstractmethod
    def update_figure(
        self, experiment_id: str, figure: Union[str, bytes], figure_name: str
    ) -> Tuple[str, int]:
        """Update an existing figure.

        Args:
            experiment_id: Experiment ID.
            figure: Path of the figure file or figure data to store.
            figure_name: Name of the figure.

        Returns:
            A tuple of the name and size of the saved figure.

        Raises:
            ExperimentEntryNotFound: If the figure does not exist.
        """
        pass

    @abstractmethod
    def figure(
        self, experiment_id: str, figure_name: str, file_name: Optional[str] = None
    ) -> Union[int, bytes]:
        """Retrieve an existing figure.

        Args:
            experiment_id: Experiment ID.
            figure_name: Name of the figure.
            file_name: Name of the local file to save the figure to. If ``None``,
                the content of the figure is returned instead.

        Returns:
            The size of the figure if `file_name` is specified. Otherwise the
            content of the figure in bytes.

        Raises:
            ExperimentEntryNotFound: If the figure does not exist.
        """
        pass

    @abstractmethod
    def delete_figure(
        self,
        experiment_id: str,
        figure_name: str,
    ) -> None:
        """Delete an existing figure.

        Args:
            experiment_id: Experiment ID.
            figure_name: Name of the figure.
        """
        pass

    @property
    @abstractmethod
    def preferences(self) -> Dict:
        """Return the preferences for the service.

        Note:
            These are preferences passed to the applications that use this service
            and have no effect on the service itself. It is up to the application
            to implement the preferences.

        Returns:
            Dict: The experiment preferences.
        """
        pass
