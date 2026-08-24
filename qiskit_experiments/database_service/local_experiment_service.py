# This code is part of Qiskit.
#
# (C) Copyright IBM 2021, 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Local experiment service for storing experiment data locally."""

import json
import logging
import os
from dataclasses import fields
from datetime import datetime, timezone

import pandas as pd
import numpy as np

from qiskit_experiments.database_service import (
    DbAnalysisResultData,
    DbExperimentData,
    ExperimentEntryNotFound,
    ExperimentEntryExists,
    ResultQuality,
)

logger = logging.getLogger(__name__)


class LocalExperimentService:
    """Provides local experiment database services.

    This class provides a service for storing experiment data locally
    without connecting to a remote service. Data can be persisted to
    disk or kept only in memory.

    .. note::

        This class is designed for demonstration and testing purposes and will
        not scale well to storing many results. It stores all results in memory
        and writes all data out to disk at every save. It could serve as a
        reference for writing a more scalable system for saving experiments.
    """

    experiment_db_columns = [f.name for f in fields(DbExperimentData)]
    results_db_columns = [f.name for f in fields(DbAnalysisResultData)]

    def __init__(
        self,
        db_dir: str | None = None,
    ) -> None:
        """LocalExperimentService constructor.

        Args:
            db_dir: The directory in which to place the database files.
                If None, results are saved in memory only and lost when the
                Python process ends.
        """
        self._experiments = pd.DataFrame()
        self._results = pd.DataFrame()
        self._figures = None
        self._files = None
        self._files_list = {}
        self._options = {}

        self.db_dir = db_dir
        self.figures_dir = os.path.join(self.db_dir, "figures") if db_dir else None
        self.files_dir = os.path.join(self.db_dir, "files") if db_dir else None
        self.experiments_file = os.path.join(self.db_dir, "experiments.json") if db_dir else None
        self.results_file = os.path.join(self.db_dir, "results.json") if db_dir else None
        if db_dir:
            self._create_directories()

        self._init_db()

    def _create_directories(self):
        """Creates the directories needed for the DB if they do not exist (internal method)"""
        dirs_to_create = [self.db_dir, self.figures_dir, self.files_dir]
        for dir_to_create in dirs_to_create:
            if not os.path.exists(dir_to_create):
                os.makedirs(dir_to_create, exist_ok=True)

    def save(self):
        """Saves the db to disk"""
        if self.db_dir:
            self._experiments.to_json(self.experiments_file)
            self._results.to_json(self.results_file)
            self._save_figures()
            self._save_files()

    def _save_figures(self):
        """Saves the figures to disk"""
        for exp_id in self._figures:
            for figure_name, figure_data in self._figures[exp_id].items():
                filename = f"{exp_id}_{figure_name}"
                with open(os.path.join(self.figures_dir, filename), "wb") as file:
                    file.write(figure_data)

    def _save_files(self):
        """Saves the files to disk"""
        db_files = set()
        for exp_id in self._files:
            for file_name, file_data in self._files[exp_id].items():
                full_filename = f"{exp_id}_{file_name}"
                db_files.add(full_filename)
                file_ext = os.path.splitext(full_filename)[1]
                mode = "wb" if file_ext == ".zip" else "w"
                encoding = None if mode == "wb" else "utf-8"
                with open(
                    os.path.join(self.files_dir, full_filename), mode, encoding=encoding
                ) as file:
                    file.write(file_data)
        current_files = set(os.listdir(self.files_dir))
        stray_files = current_files.difference(db_files)
        for file in stray_files:
            try:
                os.unlink(os.path.join(self.files_dir, file))
            except (OSError, FileNotFoundError):
                pass

    def _init_db(self):
        """Initializes the db (internal method)"""
        if self.db_dir:
            if os.path.exists(self.experiments_file):
                self._experiments = pd.read_json(self.experiments_file)
            else:
                self._experiments = pd.DataFrame(columns=self.experiment_db_columns)

            if os.path.exists(self.results_file):
                self._results = pd.read_json(self.results_file)
            else:
                self._results = pd.DataFrame(columns=self.results_db_columns)

            if os.path.exists(self.figures_dir):
                self._figures = self._get_figure_list()
            else:
                self._figures = {}
            if os.path.exists(self.files_dir):
                self._files, self._files_list = self._get_files()
            else:
                self._files = {}
        else:
            self._experiments = pd.DataFrame(columns=self.experiment_db_columns)
            self._results = pd.DataFrame(columns=self.results_db_columns)
            self._figures = {}
            self._files = {}

        self.save()

    @property
    def options(self) -> dict:
        """Return service options dictionary."""
        return self._options

    def backends(self) -> dict:
        """Return the backend list from the experiment DB."""
        return self._experiments.backend.unique().tolist()

    def experiments(self) -> list[str]:
        """Retrieve experiment ids

        Returns:
            A list of experiment ids.
        """
        return self._experiments.experiment_id.unique().tolist()

    def experiment(
        self,
        experiment_id: str,
        json_decoder: type[json.JSONDecoder] = None,  # pylint: disable=unused-argument
    ) -> DbExperimentData:
        """Retrieve a single experiment from the database.

        Args:
            experiment_id: Experiment ID
            json_decoder: Custom JSON decoder (unused in local service)

        Returns:
            Retrieved experiment data

        Raises:
            ExperimentEntryNotFound: If the experiment is not found
        """
        exp = self._experiments.loc[self._experiments.experiment_id == experiment_id]
        if exp.empty:
            raise ExperimentEntryNotFound(f"Experiment {experiment_id} not found")

        # Convert the first (and only) row to DbExperimentData
        exp_dict = self._prepare_experiment_data(exp.iloc[0].to_dict())
        return DbExperimentData(**exp_dict)

    def create_or_update_experiment(
        self,
        data: "DbExperimentData",
        json_encoder: type[json.JSONEncoder] = json.JSONEncoder,  # pylint: disable=unused-argument
        create: bool = True,
        max_attempts: int = 3,
        **kwargs,  # pylint: disable=unused-argument
    ) -> str:
        """Creates a new experiment, or updates an existing one.

        Args:
            data: The experiment data to save
            json_encoder: Custom JSON encoder (unused)
            create: Whether to attempt create first
            max_attempts: Maximum number of attempts
            **kwargs: Additional parameters (ignored for local service)

        Returns:
            Experiment ID
        """

        # Convert DbExperimentData to API format
        api_data = {f.name: val for f in fields(data) if (val := getattr(data, f.name)) is not None}
        for field in ("creation_datetime", "start_datetime", "end_datetime", "updated_datetime"):
            if api_data.get(field):
                api_data[field] = api_data[field].isoformat()

        def create_exp():
            return self._experiment_create(api_data)

        def update_exp():
            # Remove fields that shouldn't be updated
            update_data = api_data.copy()
            for field in [
                "experiment_id",
                "device_name",
                "group_id",
                "hub_id",
                "project_id",
                "type",
                "start_time",
                "parent_id",
            ]:
                update_data.pop(field, None)
            return self._experiment_update(data.experiment_id, update_data)

        params = {}
        result = self._create_or_update(create_exp, update_exp, params, create, max_attempts)
        return DbExperimentData(**result)

    def delete_experiment(self, experiment_id: str) -> None:
        """Delete an experiment from the database.

        Args:
            experiment_id: Experiment ID to delete

        Raises:
            ExperimentEntryNotFound: If the experiment is not found
        """
        exp = self._experiments.loc[self._experiments.experiment_id == experiment_id]
        if exp.empty:
            raise ExperimentEntryNotFound(f"Experiment {experiment_id} not found")

        self._experiments.drop(
            self._experiments.loc[self._experiments.experiment_id == experiment_id].index,
            inplace=True,
        )
        self.save()

    def _prepare_experiment_data(self, row: dict) -> dict:
        """Prepare database entry fields for dataclass

        Args:
            row: Dataframe row containing experiment data

        Returns:
            Dictionary suitable for DbExperimentData initialization
        """
        data = row.copy()

        # Convert timestamps
        for field in ("creation_datetime", "start_datetime", "end_datetime", "updated_datetime"):
            if pd.notna(data.get(field)):
                data[field] = datetime.fromisoformat(data[field])

        list_fields = {"tags", "job_ids"}
        str_fields = {"notes", "hub", "group", "project", "owner"}
        dict_fields = {"metadata"}

        for key, val in data.items():
            if isinstance(val, float) and pd.isna(val):
                if key in list_fields:
                    data[key] = []
                elif key in str_fields:
                    data[key] = ""
                elif key in dict_fields:
                    data[key] = {}
                else:
                    data[key] = None

        return data

    def _experiment_create(self, data: dict) -> dict:
        """Create an experiment (internal method).

        Args:
            data: Experiment data.

        Returns:
            Experiment data.

        Raises:
            ExperimentEntryExists: If the experiment already exists

        """
        data_dict = data.copy()
        now = datetime.now(timezone.utc).isoformat()
        for time_field in ("start_datetime", "creation_datetime", "updated_datetime"):
            if time_field not in data_dict:
                data_dict[time_field] = now
        if "tags" not in data_dict:
            data_dict["tags"] = []
        if "figure_names" not in data_dict:
            data_dict["figure_names"] = []

        exp = self._experiments.loc[self._experiments.experiment_id == data_dict["experiment_id"]]
        if not exp.empty:
            raise ExperimentEntryExists

        new_df = pd.DataFrame([data_dict], columns=self._experiments.columns)
        self._experiments = pd.concat([self._experiments, new_df], ignore_index=True)
        self.save()
        exp = self._experiments.loc[self._experiments.experiment_id == data_dict["experiment_id"]]
        return self._prepare_experiment_data(exp.to_dict("records")[0])

    def _experiment_update(self, experiment_id: str, new_data: dict) -> dict:
        """Update an experiment (internal method).

        Args:
            experiment_id: Experiment UUID.
            new_data: New experiment data.

        Returns:
            Experiment data.

        Raises:
            ExperimentEntryNotFound: If the experiment is not found
        """
        new_data = new_data.copy()
        exp = self._experiments.loc[self._experiments.experiment_id == experiment_id]
        if exp.empty:
            raise ExperimentEntryNotFound
        exp_index = exp.index[0]
        new_data["updated_datetime"] = datetime.now(timezone.utc).isoformat()
        for key, value in new_data.items():
            self._experiments.at[exp_index, key] = value
        self.save()
        exp = self._experiments.loc[self._experiments.experiment_id == experiment_id]
        return self._prepare_experiment_data(exp.to_dict("records")[0])

    def analysis_results(
        self,
        limit: int | None = None,
        backend_name: str | None = None,
        device_components: list[str] | None = None,
        experiment_id: str | None = None,
        result_type: str | None = None,
        quality: str | list[str] | None = None,
        verified: bool | None = None,
        tags: list[str] | None = None,
        created_at: list | None = None,
        json_decoder: type[json.JSONDecoder] = None,  # pylint: disable=unused-argument
    ) -> list[DbAnalysisResultData]:
        """Return a list of analysis results.

        Args:
            limit: Number of analysis results to retrieve.
            backend_name: Name of the backend.
            device_components: A list of device components used for filtering.
            experiment_id: Experiment UUID used for filtering.
            result_type: Analysis result type used for filtering.
            quality: Quality value used for filtering.
            verified: Indicates whether this result has been verified.
            tags: Filter by tags assigned to analysis results.
            created_at: A list of timestamps used to filter by creation time.
            json_decoder: Custom JSON decoder (unused in local service)

        Returns:
            A list of analysis results.
        Raises:
            ValueError: If the parameters are unsuitable for filtering
        """
        # pylint: disable=unused-argument
        df = self._results

        # TODO: skipping device components for now until we conslidate more with the provider service
        # (in the qiskit-experiments service there is no operator for device components,
        # so the specification for filtering is not clearly defined)

        if experiment_id is not None:
            df = df.loc[df.experiment_id == experiment_id]
        if result_type is not None:
            df = df.loc[df.result_type == result_type]
        if backend_name is not None:
            df = df.loc[df.backend_name == backend_name]
        if quality is not None:
            df = df.loc[df.quality == quality]
        if verified is not None:
            df = df.loc[df.verified == verified]

        if tags is not None:
            tags = tags.split(",")
            df = df.loc[df.tags.apply(lambda dftags: any(x in dftags for x in tags))]

        df = df.sort_values(["creation_datetime", "experiment_id"], ascending=[False, True])

        if limit is not None:
            df = df.iloc[:limit]

        # Convert dataframe rows to DbAnalysisResultData objects
        results = []
        for _, row in df.iterrows():
            result_dict = self._prepare_analysis_result_data(row.to_dict())
            results.append(DbAnalysisResultData(**result_dict))

        return results

    def analysis_result(
        self,
        result_id: str,
        json_decoder: type[json.JSONDecoder] = None,  # pylint: disable=unused-argument
    ) -> DbAnalysisResultData:
        """Retrieve a single analysis result from the database.

        Args:
            result_id: Analysis result ID
            json_decoder: Custom JSON decoder (unused in local service)

        Returns:
            Retrieved analysis result data

        Raises:
            ExperimentEntryNotFound: If the analysis result is not found
        """
        result = self._results.loc[self._results.result_id == result_id]
        if result.empty:
            raise ExperimentEntryNotFound(f"Analysis result {result_id} not found")

        # Convert the first (and only) row to DbAnalysisResultData
        result_dict = self._prepare_analysis_result_data(result.iloc[0].to_dict())
        return DbAnalysisResultData(**result_dict)

    def create_or_update_analysis_result(
        self,
        data: "DbAnalysisResultData",
        json_encoder: type[json.JSONEncoder] = json.JSONEncoder,  # pylint: disable=unused-argument
        create: bool = True,
        max_attempts: int = 3,
    ) -> str:
        """Creates or updates an analysis result.

        Args:
            data: The analysis result data to save
            json_encoder: Custom JSON encoder (unused)
            create: Whether to attempt create first
            max_attempts: Maximum number of attempts

        Returns:
            Analysis result ID
        """

        # Convert DbAnalysisResultData to API format
        api_data = {f.name: val for f in fields(data) if (val := getattr(data, f.name)) is not None}
        if api_data.get("quality"):
            api_data["quality"] = ResultQuality.to_str(api_data["quality"])
        for field in ("creation_datetime", "updated_datetime"):
            if api_data.get(field):
                api_data[field] = api_data[field].isoformat()

        def create_result():
            return self._analysis_result_create(api_data)

        def update_result():
            # Remove fields that shouldn't be updated
            update_data = api_data.copy()
            for field in ["result_id", "experiment_id", "device_components", "type"]:
                update_data.pop(field, None)
            return self._analysis_result_update(data.result_id, update_data)

        params = {}
        result = self._create_or_update(create_result, update_result, params, create, max_attempts)
        # result is a dict from analysis_result_create or _analysis_result_update
        return result["result_id"]

    def create_analysis_results(
        self,
        data: list["DbAnalysisResultData"],
        blocking: bool = True,  # pylint: disable=unused-argument
        max_workers: int = 100,  # pylint: disable=unused-argument
        json_encoder: type[json.JSONEncoder] = json.JSONEncoder,
    ):
        """Create multiple analysis results (simplified without threading for local).

        Args:
            data: List of analysis result data to save
            blocking: Ignored for local service (always blocking)
            max_workers: Ignored for local service
            json_encoder: Custom JSON encoder

        Returns:
            Status dictionary with results
        """
        successful = []
        failed = []

        for result_data in data:
            try:
                self.create_or_update_analysis_result(
                    result_data, json_encoder=json_encoder, create=True, max_attempts=3
                )
                successful.append(result_data)
            except Exception as ex:  # pylint: disable=broad-exception-caught
                failed.append({"data": result_data, "exception": ex})

        return {
            "running": [],
            "done": successful,
            "fail": failed,
        }

    def delete_analysis_result(self, result_id: str) -> dict:
        """Delete an analysis result.

        Args:
            result_id: Analysis result ID.

        Raises:
            ExperimentEntryNotFound: If the analysis result is not found
        """
        result = self._results.loc[self._results.result_id == result_id]
        if result.empty:
            raise ExperimentEntryNotFound
        self._results.drop(
            self._results.loc[self._results.result_id == result_id].index, inplace=True
        )
        self.save()

    def _analysis_result_create(self, result: dict) -> dict:
        """Upload an analysis result.

        Args:
            result: The analysis result to upload

        Returns:
            Analysis result data.

        Raises:
            ValueError: If experiment id is missing
            ExperimentEntryNotFound: If experiment is not found
        """
        data_dict = result.copy()

        exp_id = data_dict.get("experiment_id")
        if exp_id is None:
            raise ValueError("Cannot create analysis result without experiment id")
        exp = self._experiments.loc[self._experiments.experiment_id == exp_id]
        if exp.empty:
            raise ExperimentEntryNotFound(f"Experiment {exp_id} not found")
        exp_index = exp.index[0]
        data_dict["backend_name"] = self._experiments.at[exp_index, "backend"]
        now = datetime.now(timezone.utc).isoformat()
        if data_dict.get("creation_datetime") is None:
            data_dict["creation_datetime"] = now
        if data_dict.get("updated_datetime") is None:
            data_dict["updated_datetime"] = now

        new_df = pd.DataFrame([data_dict], columns=self._results.columns)
        self._results = pd.concat([self._results, new_df], ignore_index=True)
        self.save()
        return data_dict

    def _analysis_result_update(self, result_id: str, new_data: dict) -> dict:
        """Update an analysis result (internal method).

        Args:
            result_id: Analysis result ID.
            new_data: New analysis result data.

        Returns:
            Analysis result data.

        Raises:
            ExperimentEntryNotFound: If the analysis result is not found
        """
        new_data = new_data.copy()
        result = self._results.loc[self._results.result_id == result_id]
        if result.empty:
            raise ExperimentEntryNotFound
        result_index = result.index[0]
        new_data["updated_datetime"] = datetime.now(timezone.utc).isoformat()
        for key, value in new_data.items():
            self._results.at[result_index, key] = value
        self.save()
        result = self._results.loc[self._results.result_id == result_id]
        return self._prepare_analysis_result_data(result.to_dict("records")[0])

    def _prepare_analysis_result_data(self, row: dict) -> dict:
        """Prepare row dict from database for analysis result dataclass.

        Args:
            row: Dataframe row containing analysis result data

        Returns:
            Dictionary suitable for DbAnalysisResultData initialization
        """
        data = row.copy()

        # Convert timestamps
        for field in ("creation_datetime", "updated_datetime"):
            if pd.notna(data.get(field)):
                data[field] = datetime.fromisoformat(data[field])

        list_fields = {"device_components", "tags"}
        str_fields = {"notes", "hub", "group", "project", "owner"}
        dict_fields = {"result_data"}
        bool_fields = {"verified"}

        for key, val in data.items():
            if isinstance(val, float) and pd.isna(val):
                if key in list_fields:
                    data[key] = []
                elif key in str_fields:
                    data[key] = ""
                elif key in dict_fields:
                    data[key] = {}
                elif key in bool_fields:
                    data[key] = False
                else:
                    data[key] = None

        if "quality" in data and isinstance(data["quality"], str):
            data["quality"] = ResultQuality.from_str(data["quality"])

        return data

    def figure(
        self, experiment_id: str, figure_name: str, file_name: str | None = None
    ) -> int | bytes:
        """Retrieve an existing figure.

        Args:
            experiment_id: Experiment ID
            figure_name: Name of the figure
            file_name: Local file to save to (if None, returns bytes)

        Returns:
            Size if file_name given, otherwise figure bytes
        """
        data = self._figure_get(experiment_id, figure_name)

        if file_name:
            with open(file_name, "wb") as file:
                num_bytes = file.write(data)
            return num_bytes

        return data

    def create_or_update_figure(
        self,
        experiment_id: str,
        figure: str | bytes,
        figure_name: str | None = None,
        create: bool = True,
        max_attempts: int = 3,
    ) -> tuple:
        """Creates a figure if it doesn't exist, otherwise updates it.

        Args:
            experiment_id: Experiment ID
            figure: Figure file name or figure data
            figure_name: Name of the figure
            create: Whether to attempt create first
            max_attempts: Maximum number of attempts

        Returns:
            Tuple of (figure_name, size)
        """
        params = {
            "experiment_id": experiment_id,
            "figure": figure,
            "figure_name": figure_name,
        }
        return self._create_or_update(
            self._figure_create, self._figure_update, params, create, max_attempts
        )

    def create_figures(
        self,
        experiment_id: str,
        figure_list: list[tuple],
        blocking: bool = True,  # pylint: disable=unused-argument
        max_workers: int = 100,  # pylint: disable=unused-argument
    ):
        """Create multiple figures (simplified without threading for local).

        Args:
            experiment_id: ID of the experiment
            figure_list: List of (figure, name) tuples
            blocking: Ignored for local service
            max_workers: Ignored for local service

        Returns:
            Status dictionary with results
        """
        successful = []
        failed = []

        for figure, figure_name in figure_list:
            try:
                self.create_or_update_figure(
                    experiment_id=experiment_id,
                    figure=figure,
                    figure_name=figure_name,
                    create=True,
                    max_attempts=3,
                )
                successful.append((figure, figure_name))
            except Exception as ex:  # pylint: disable=broad-exception-caught
                failed.append({"data": (figure, figure_name), "exception": ex})

        return {
            "running": [],
            "done": successful,
            "fail": failed,
        }

    def delete_figure(self, experiment_id: str, figure_name: str) -> None:
        """Delete an experiment plot.

        Args:
            experiment_id: Experiment ID
            figure_name: Name of the figure
        """
        try:
            self._figure_delete(experiment_id, figure_name)
        except ExperimentEntryNotFound:
            logger.warning("Figure %s not found.", figure_name)

    def _get_figure_list(self):
        """Generates the figure dictionary based on stored data on disk"""
        figures = {}
        for exp_id in self._experiments.experiment_id:
            # exp_id should be str to begin with, so just in case
            exp_id_string = str(exp_id)
            figures_for_exp = {}
            for filename in os.listdir(self.figures_dir):
                if filename.startswith(exp_id_string):
                    with open(os.path.join(self.figures_dir, filename), "rb") as file:
                        figure_data = file.read()
                    figure_name = filename[len(exp_id_string) + 1 :]
                    figures_for_exp[figure_name] = figure_data
            figures[exp_id] = figures_for_exp
        return figures

    def _figure_create(
        self,
        experiment_id: str,
        figure: str | bytes,
        figure_name: str | None = None,
    ) -> tuple:
        """Store a new figure in the database (internal method).

        Args:
            experiment_id: ID of the experiment
            figure: Figure file name or figure data
            figure_name: Name of the figure

        Returns:
            Tuple of (figure_name, size)
        """
        if figure_name is None:
            if isinstance(figure, str):
                figure_name = figure
            else:
                figure_name = f"figure_{datetime.now(timezone.utc).isoformat()}.svg"

        if not figure_name.endswith(".svg"):
            figure_name += ".svg"

        if isinstance(figure, str):
            with open(figure, "rb") as file:
                figure = file.read()

        if experiment_id not in self._figures:
            self._figures[experiment_id] = {}
        exp_figures = self._figures[experiment_id]
        if figure_name in exp_figures:
            raise ExperimentEntryExists(f"Figure {figure_name} already exists")
        exp_figures[figure_name] = figure
        self.save()

        return figure_name, len(figure)

    def _figure_get(self, experiment_id: str, plot_name: str) -> bytes:
        """Retrieve an experiment plot (internal method).

        Args:
            experiment_id: Experiment UUID.
            plot_name: Name of the plot.

        Returns:
            Retrieved experiment plot.

        Raises:
            ExperimentEntryNotFound: If the figure is not found
        """

        exp_figures = self._figures[experiment_id]
        if plot_name not in exp_figures:
            raise ExperimentEntryNotFound(f"Figure {plot_name} not found")
        return exp_figures[plot_name]

    def _figure_update(
        self,
        experiment_id: str,
        figure: str | bytes,
        figure_name: str,
    ) -> tuple:
        """Update an existing figure (internal method).

        Args:
            experiment_id: Experiment ID
            figure: Figure file name or figure data
            figure_name: Name of the figure

        Returns:
            Tuple of (figure_name, size)
        """
        if not figure_name.endswith(".svg"):
            figure_name += ".svg"

        if isinstance(figure, str):
            with open(figure, "rb") as file:
                figure = file.read()

        exp_figures = self._figures[experiment_id]
        if figure_name not in exp_figures:
            raise ExperimentEntryNotFound(f"Figure {figure_name} not found")
        exp_figures[figure_name] = figure
        self.save()

        return figure_name, len(figure)

    def _figure_delete(self, experiment_id: str, plot_name: str) -> None:
        """Delete an experiment plot (internal method).

        Args:
            experiment_id: Experiment UUID.
            plot_name: Plot file name.

        Raises:
            ExperimentEntryNotFound: If the figure is not found
        """
        exp_figures = self._figures[experiment_id]
        if plot_name not in exp_figures:
            raise ExperimentEntryNotFound(f"Figure {plot_name} not found")
        del exp_figures[plot_name]

    def files(self, experiment_id: str) -> dict:
        """Retrieve the file list for an experiment.

        Args:
            experiment_id: Experiment ID

        Returns:
            Dictionary with file list metadata (format: {"files": [...]})
        """
        return {"files": self._files_list.get(experiment_id, [])}

    def experiment_has_file(self, experiment_id: str, filename: str) -> bool:
        """Check if an experiment has a specific file.

        Args:
            experiment_id: Experiment ID
            filename: Name of the file to check

        Returns:
            True if the file exists, False otherwise
        """
        if experiment_id not in self._files:
            return False
        return filename in self._files[experiment_id]

    def file_upload(
        self,
        experiment_id: str,
        file_name: str,
        file_data: dict | str | bytes,
        json_encoder: type[json.JSONEncoder] = json.JSONEncoder,
    ):
        """Uploads a data file to the DB.

        Args:
            experiment_id: The experiment the file belongs to
            file_name: The expected filename
            file_data: Dictionary or JSON string or bytes to save
            json_encoder: Custom JSON encoder

        Raises:
            RuntimeError: pyyaml not available and a yaml file requested
        """
        # Ensure proper file extension
        if not (
            file_name.endswith(".json") or file_name.endswith(".yaml") or file_name.endswith(".zip")
        ):
            file_name += ".json"

        if isinstance(file_data, dict):
            if file_name.endswith(".yaml"):
                try:
                    import yaml
                except ImportError as err:
                    raise RuntimeError("pyyaml required to store yaml file!") from err
                file_data = yaml.dump(file_data)
            elif file_name.endswith(".json"):
                file_data = json.dumps(file_data, cls=json_encoder)

        if experiment_id not in self._files_list:
            self._files_list[experiment_id] = []
        if experiment_id not in self._files:
            self._files[experiment_id] = {}
        size = len(file_data)
        new_file_element = {
            "Key": file_name,
            "Size": size,
            "LastModified": datetime.now(timezone.utc).isoformat(),
        }
        self._files_list[experiment_id].append(new_file_element)
        self._files[experiment_id][file_name] = file_data
        self.save()

    def file_delete(
        self,
        experiment_id: str,
        file_name: str,
    ):
        """Delete a file from the database"""
        if not (
            file_name.endswith(".json") or file_name.endswith(".yaml") or file_name.endswith(".zip")
        ):
            file_name += ".json"

        if experiment_id not in self._files:
            raise ExperimentEntryNotFound
        if file_name not in self._files[experiment_id]:
            raise ExperimentEntryNotFound

        del self._files[experiment_id][file_name]
        self._files_list[experiment_id] = [
            e for e in self._files_list[experiment_id] if e["Key"] != file_name
        ]
        self.save()

    def file_download(
        self,
        experiment_id: str,
        file_name: str,
        json_decoder: type[json.JSONDecoder] = json.JSONDecoder,
    ) -> dict:
        """Downloads a data file from the DB.

        Args:
            experiment_id: The experiment the file belongs to
            file_name: The filename
            json_decoder: Custom JSON decoder

        Returns:
            Deserialized file data

        Raises:
            ExperimentEntryNotFound: File not found
            RuntimeError: pyyaml not available and a yaml file requested
        """
        if not (
            file_name.endswith(".json") or file_name.endswith(".yaml") or file_name.endswith(".zip")
        ):
            file_name += ".json"

        if experiment_id not in self._files:
            raise ExperimentEntryNotFound
        if file_name not in self._files[experiment_id]:
            raise ExperimentEntryNotFound
        if file_name.endswith(".yaml"):
            try:
                import yaml
            except ImportError as err:
                raise RuntimeError("pyyaml required to load yaml file!") from err
            return yaml.safe_load(self._files[experiment_id][file_name])
        elif file_name.endswith(".json"):
            return json.loads(self._files[experiment_id][file_name], cls=json_decoder)
        return self._files[experiment_id][file_name]

    def _get_files(self):
        """Generates the figure dictionary based on stored data on disk"""
        files = {}
        files_list = {}
        for exp_id in self._experiments.experiment_id:
            # exp_id should be str to begin with, so just in case
            exp_id_string = str(exp_id)
            file_list_for_exp = []
            files_for_exp = {}
            for filename in os.listdir(self.files_dir):
                if filename.startswith(exp_id_string):
                    file_full_path = os.path.join(self.files_dir, filename)
                    file_ext = os.path.splitext(filename)[1]
                    mode = "rb" if file_ext == ".zip" else "r"
                    encoding = None if mode == "rb" else "utf-8"
                    with open(file_full_path, mode, encoding=encoding) as file:
                        file_data = file.read()
                    file_size = len(file_data)
                    file_name = filename[len(exp_id_string) + 1 :]
                    files_for_exp[file_name] = file_data
                    new_file_element = {
                        "Key": file_name,
                        "Size": file_size,
                        "LastModified": os.path.getmtime(file_full_path),
                    }
                    file_list_for_exp.append(new_file_element)
            files_list[exp_id] = file_list_for_exp
            files[exp_id] = files_for_exp
        return files, files_list

    def _experiment_files_get(self, experiment_id: str) -> dict[str, list[str]]:
        """Retrieve experiment related files (internal method).

        Args:
            experiment_id: Experiment ID.

        Returns:
            Experiment files.
        """
        return {"files": self._files_list.get(experiment_id, [])}

    def _experiment_file_download_impl(
        self, experiment_id: str, file_name: str, json_decoder: type[json.JSONDecoder]
    ) -> dict:
        """Downloads a data file from the DB (internal implementation)

        Args:
            experiment_id: Experiment ID.
            file_name: The name of the data file
            json_decoder: Custom decoder to use to decode the retrieved experiment.

        Returns:
            The Dictionary of contents of the file

        Raises:
            ExperimentEntryNotFound: if experiment or file not found
        """
        if experiment_id not in self._files:
            raise ExperimentEntryNotFound
        if file_name not in self._files[experiment_id]:
            raise ExperimentEntryNotFound
        if file_name.endswith(".yaml"):
            try:
                import yaml
            except ImportError as err:
                raise RuntimeError("pyyaml required to load yaml file!") from err
            return yaml.safe_load(self._files[experiment_id][file_name])
        elif file_name.endswith(".json"):
            return json.loads(self._files[experiment_id][file_name], cls=json_decoder)
        return self._files[experiment_id][file_name]

    def _create_or_update(
        self,
        create_func,
        update_func,
        params,
        create: bool = True,
        max_attempts: int = 3,
    ):
        """Creates or updates a database entry using the given functions.

        Args:
            create_func: Function to create new entry
            update_func: Function to update existing entry
            params: Parameters to pass to the functions
            create: Whether to attempt create first
            max_attempts: Maximum number of attempts

        Returns:
            Result from the successful function call
        """
        attempts = 0
        success = False
        result = None
        while attempts < max_attempts and not success:
            attempts += 1
            if create:
                try:
                    result = create_func(**params)
                    success = True
                except ExperimentEntryExists:
                    create = False
            else:
                try:
                    result = update_func(**params)
                    success = True
                except ExperimentEntryNotFound:
                    create = True
        return result

    def _convert_db_to_dict(self, dataframe: pd.DataFrame):
        """Prepares db values for dataclasses"""
        result = dataframe.replace({np.nan: None}).to_dict("records")[0]
        return result
