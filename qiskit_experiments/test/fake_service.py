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
import pandas as pd
from datetime import datetime, timedelta

from qiskit_experiments.test.fake_backend import FakeBackend

from qiskit_experiments.database_service import DatabaseServiceV1
from qiskit_experiments.database_service.device_component import DeviceComponent


class FakeService(DatabaseServiceV1):
    """
    Extremely simple database for testing
    """

    def __init__(self):
        self.exps = pd.DataFrame(columns=["experiment_type", "backend_name", "metadata", "experiment_id", "parent_id", "job_ids", "tags", "notes"])
        self.results = pd.DataFrame(columns=["experiment_id", "result_data", "result_type", "device_components", "tags", "quality", "verified", "result_id", "chisq"])

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

        if experiment_id is None:
            raise ValueError("The fake service requires the experiment id parameter")
        
        self.exps = self.exps.append({ 
            "experiment_type": experiment_type,
            "experiment_id": experiment_id,
            "parent_id": parent_id,
            "backend_name": backend_name,
            "metadata": metadata,
            "job_ids": job_ids,
            "tags": tags,
            "notes": notes,
            "share_level": kwargs.get("share_level", None),
            "device_components": [],
            "start_datetime": datetime(2022, 1, 1) + timedelta(hours=len(self.exps))
        }, ignore_index=True)

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
        row = (self.exps.experiment_id == experiment_id)
        if metadata is not None:
            self.exps.loc[row, "metadata"] = metadata
        if job_ids is not None:
            self.exps.loc[row, "job_ids"] = job_ids
        if tags is not None:
            self.exps.loc[row, "tags"] = tags
        if notes is not None:
            self.exps.loc[row, "notes"] = notes
        if "share_level" in kwargs:
            self.exps.loc[row, "share_level"] = kwargs["share_level"]
        if "parent_id" in kwargs:
            self.exps.loc[row, "parent_id"] = kwargs["parent_id"]

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
        db_entry = self.exps.loc[self.exps.experiment_id == experiment_id].to_dict("records")[0]
        db_entry["backend"] = FakeBackend(db_entry["backend_name"])
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
        df = self.exps
        
        if experiment_type is not None:
            df = df.loc[df.experiment_type == experiment_type]

        # TODO: do we have to return the backend itself, as in `experiment`?
        if backend_name is not None:
            df = df.loc[df.backend_name == backend_name]

        # Note a bug in the interface for all services:
        # It is impossible to filter by expeirments whose parent id is None
        # (i.e., root experiments)
        if parent_id is not None:
            df = df.loc[df.parent_id == parent_id]

        # Waiting for consistency between provider service and qiskit-experiments service,
        # currently they have different types for `device_components`
        if device_components is not None:
            raise ValueError("The fake service currently does not support filtering on device components")

        if tags is not None:
            if tags_operator == "OR":
                df = df.loc[df.tags.apply(lambda dftags: any([x in dftags for x in tags]))]
            elif tags_operator == "AND":
                df = df.loc[df.tags.apply(lambda dftags: all([x in dftags for x in tags]))]
            else:
                raise ValueError("Unrecognized tags operator")

        # These are parameters of IBMExperimentService.experiments
        if "start_datetime_before" in filters:
            df = df.loc[df.start_datetime <= filters["start_datetime_before"]]
        if "start_datetime_after" in filters:
            df = df.loc[df.start_datetime >= filters["start_datetime_after"]]

        # This is a parameter of IBMExperimentService.experiments
        if "sort_by" in filters:
            sort_by = filters["sort_by"]
        else:
            sort_by = "start_datetime:desc"

        if not isinstance(sort_by, list):
            sort_by = [sort_by]
            
        # TODO: support also experiment_type
        if len(sort_by) != 1:
            raise ValueError("The fake service currently supports only sorting by start_datetime")

        sortby_split = sort_by[0].split(":")
        # TODO: support also experiment_type
        if len(sortby_split) != 2 or sortby_split[0] != "start_datetime" or (sortby_split[1] != "asc" and sortby_split[1] != "desc"):
            raise ValueError("The fake service currently supports only sorting by start_datetime, which can be either asc or desc")

        df = df.sort_values(by="start_datetime", ascending=(sortby_split[1] == "asc"))

        df = df.iloc[:limit]
            
        return df.to_dict("records")

    def delete_experiment(self, experiment_id: str) -> None:
        index = self.exps[self.exps.experiment_id == experiment_id].index
        self.exps.drop(index, inplace=True)

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
        self.results = self.results.append({
            "result_data": result_data,
            "result_id": result_id,
            "result_type": result_type,
            "device_components": device_components,
            "experiment_id": experiment_id,
            "quality": quality,
            "verified": verified,
            "tags": tags,
            "backend_name": self.exps.loc[self.exps.experiment_id == experiment_id].iloc[0].backend_name,
            "chisq": kwargs.get("chisq", None)
        }, ignore_index=True)

        def add_new_components(expcomps):
            for dc in device_components:
                if dc not in expcomps:
                    expcomps.append(dc)

        self.exps.loc[self.exps.experiment_id==experiment_id, "device_components"].apply(add_new_components)

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
        row = (self.results.result_id == result_id)
        if result_data is not None:
            self.results.loc[row, "result_data"] = result_data
        if tags is not None:
            self.results.loc[row, "tags"] = tags
        if quality is not None:
            self.results.loc[row, "quality"] = quality
        if verified is not None:
            self.results.loc[row, "verified"] = verified
        if "chisq" in kwargs:
            self.result.loc[row, "chisq"] = chisq

    def analysis_result(
        self, result_id: str, json_decoder: Type[json.JSONDecoder] = json.JSONDecoder
    ) -> Dict:
        return self.results.loc[self.results.result_id == result_id].to_dict("records")[0]

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
        return self.results.loc[self.results.experiment_id == experiment_id].to_dict("records")

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
