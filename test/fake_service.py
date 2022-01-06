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

"""Fake service class for tests."""

from typing import Optional, List, Dict, Type, Any, Union, Tuple
import copy
import json

from test.fake_backend import FakeBackend

from qiskit_experiments.database_service import DatabaseServiceV1
from qiskit_experiments.database_service.device_component import DeviceComponent

# pylint:disable=missing-raises-doc


class FakeService(DatabaseServiceV1):
    """
    Extremely simple database for testing
    """

    def __init__(self):
        self.database = {}

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
        """

        self.database[experiment_id] = {
            "experiment_type": experiment_type,
            "experiment_id": experiment_id,
            "parent_id": parent_id,
            "backend_name": backend_name,
            "metadata": metadata,
            "job_ids": job_ids,
            "tags": tags,
            "notes": notes,
            "share_level": kwargs.get("share_level", None),
            "figure_names": kwargs.get("figure_names", None),
            "analysis": {},
        }

        return experiment_id

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
        """
        raise Exception("not implemented")

    def experiment(
        self, experiment_id: str, json_decoder: Type[json.JSONDecoder] = json.JSONDecoder
    ) -> Dict:
        """Retrieve a previously stored experiment.

        Args:
            experiment_id: Experiment ID.
            json_decoder: Custom JSON decoder to use to decode the retrieved experiment.

        Returns:
            A dictionary containing the retrieved experiment data.
        """

        db_entry = copy.deepcopy(self.database[experiment_id])
        db_entry["backend"] = FakeBackend()
        return db_entry

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
        raise Exception("not implemented")

    def delete_experiment(self, experiment_id: str) -> None:
        raise Exception("not implemented")

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
        self.database[experiment_id]["analysis"][result_id] = {
            "result_data": result_data,
            "result_id": result_id,
            "result_type": result_type,
            "device_components": device_components,
            "experiment_id": experiment_id,
            "quality": quality,
            "verified": verified,
            "tags": tags,
            "service": self,
        }

        return result_id

    def update_analysis_result(
        self,
        result_id: str,
        result_data: Optional[Dict] = None,
        tags: Optional[List[str]] = None,
        quality: Optional[str] = None,
        verified: bool = None,
        **kwargs: Any,
    ) -> None:
        raise Exception("not implemented")

    def analysis_result(
        self, result_id: str, json_decoder: Type[json.JSONDecoder] = json.JSONDecoder
    ) -> Dict:
        raise Exception("not implemented")

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
        return self.database[experiment_id]["analysis"].values()

    def delete_analysis_result(self, result_id: str) -> None:
        raise Exception("not implemented")

    def create_figure(
        self, experiment_id: str, figure: Union[str, bytes], figure_name: Optional[str]
    ) -> Tuple[str, int]:
        return

    def update_figure(
        self, experiment_id: str, figure: Union[str, bytes], figure_name: str
    ) -> Tuple[str, int]:
        raise Exception("not implemented")

    def figure(
        self, experiment_id: str, figure_name: str, file_name: Optional[str] = None
    ) -> Union[int, bytes]:
        raise Exception("not implemented")

    def delete_figure(
        self,
        experiment_id: str,
        figure_name: str,
    ) -> None:
        raise Exception("not implemented")

    @property
    def preferences(self) -> Dict:
        return {"auto_save": False}
