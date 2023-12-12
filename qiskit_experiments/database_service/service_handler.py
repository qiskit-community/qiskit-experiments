# This code is part of Qiskit.
#
# (C) Copyright IBM 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""IBM Experiment service handler."""

from __future__ import annotations

import json
import logging
import re
import contextlib
import sys
import traceback
from collections.abc import Iterator
from datetime import datetime, timezone
from typing import Any, Type, TYPE_CHECKING
from json import JSONEncoder, JSONDecoder

from qiskit_ibm_experiment.service.experiment_dataclasses import ExperimentData as ServiceExpData
from qiskit_ibm_experiment import IBMExperimentService
from qiskit.providers import Provider, Backend, Job
from qiskit.exceptions import QiskitError

from qiskit_experiments.framework import AnalysisResultTable, AnalysisResult
from qiskit_experiments.framework.json import ExperimentEncoder, ExperimentDecoder
from qiskit_experiments.database_service.exceptions import ExperimentDataError
from qiskit_experiments.database_service.device_component import to_component, DeviceComponent
from qiskit_experiments.database_service.utils import qiskit_version

from .utils import ThreadSafeOrderedDict


LOG = logging.getLogger(__name__)

if TYPE_CHECKING:
    # There is a cyclical dependency here, but the name needs to exist for
    # Sphinx on Python 3.9+ to link type hints correctly.  The gating on
    # `TYPE_CHECKING` means that the import will never be resolved by an actual
    # interpreter, only static analysis.
    from qiskit_experiments.framework.experiment_data import ExperimentData


def parse_utc_datetime(dt_str: str | None) -> datetime | None:
    """Parses UTC datetime from a string"""
    if dt_str is None:
        return None

    db_datetime_format = "%Y-%m-%dT%H:%M:%S.%fZ"
    dt_utc = datetime.strptime(dt_str, db_datetime_format)
    dt_utc = dt_utc.replace(tzinfo=timezone.utc)
    return dt_utc


class ExperimentServiceFrontend:
    """Frontend class for working with the experiment service.

    Raises:
        QiskitError: _description_
        ExperimentDataError: _description_
        ExperimentDataError: _description_
        QiskitError: _description_

    Returns:
        _type_: _description_

    Yields:
        _type_: _description_
    """

    valid_uid_regex = re.compile(r"\A(?P<short_id>\w{8})-\w{4}-\w{4}-\w{4}-\w{12}\Z")

    db_url = "https://auth.quantum-computing.ibm.com/api"
    json_encoder = ExperimentEncoder
    json_decoder = ExperimentDecoder

    metadata_version = 1
    metadata_filename = "metadata.json"
    metadata_size_limit = 10000

    def __init__(
        self,
        provider: Provider | None = None,
        service: IBMExperimentService | None = None,
    ):
        """Create new service frontend.

        Args:
            provider: Backend provider instance.
            service: Database service instance.
        """
        self._provider: Provider | None = None
        self._service: IBMExperimentService | None = None
        self._share_level: str | None = None
        self._auto_save: bool = False
        self._already_saved: bool = False

        self.backend_name: str | None = None
        self.hub: str | None = None
        self.group: str | None = None
        self.project: str | None = None

        self.created_datetime: datetime | None = None
        self.updated_datetime: datetime | None = None

        self.provider = provider
        self.service = service or self.get_service_from_provider(provider)

        # Separately track entries to sync with database.
        # Job result data is not saved.
        # Experiment service saves job IDs and retrieves the result data from IQX server.
        self.job_ids: list[str] = []
        self.figure_names: list[str] = []
        self.artifact_ids: list[str] = []
        self.analysis_result_ids: list[str] = []

    @classmethod
    def validate_uid(
        cls,
        uid: str,
    ):
        """Validate experiment data UID against defined format.

        Args:
            uid: ID to test.
        """
        if not cls.valid_uid_regex.match(uid):
            LOG.warning(
                "The experiment data ID %s is not a valid UID string in the database service. "
                "This entry might fail in saving in the database.",
                uid,
            )

    @classmethod
    def get_service_from_provider(
        cls,
        provider: Provider,
    ) -> IBMExperimentService | None:
        """A helper method to get database service instance from provider.

        Args:
            provider: Provider instance.

        Returns:
            Database service instance tied to the provider.
        """
        if provider is None:
            return None
        try:
            # qiskit-ibmq-provider style
            if hasattr(provider, "credentials"):
                token = provider.credentials.token
            # qiskit-ibm-provider style
            elif hasattr(provider, "_account"):
                token = provider._account.token
            else:
                return None
            return IBMExperimentService(token=token, url=cls.db_url)
        except Exception:  # pylint: disable=broad-except
            return None

    def set_hgp_from_provider(
        self,
        provider: Provider,
    ):
        if provider is None:
            return
        try:
            # qiskit-ibmq-provider style
            if hasattr(provider, "credentials"):
                creds = provider.credentials
                self.hgp = f"{creds.hub}/{creds.group}/{creds.project}"
            # qiskit-ibm-provider style
            elif hasattr(provider, "_get_hgp"):
                self.hgp = provider._get_hgp(backend_name=self.backend_name).name
        except (AttributeError, IndexError, QiskitError):
            pass

    @property
    def hgp(self) -> str:
        """Account information in the formatted hub/group/project string."""
        return f"{self.hub}/{self.group}/{self.project}"

    @hgp.setter
    def hgp(self, new_hgp: str):
        if re.match(r"[^/]*/[^/]*/[^/]*$", new_hgp) is None:
            raise QiskitError("hgp can be only given in a <hub>/<group>/<project> format")
        self.hub, self.group, self.project = new_hgp.split("/")

    @property
    def service(self) -> IBMExperimentService | None:
        """Experiment database service instance to store experiment data."""
        return self._service

    @service.setter
    def service(self, new_service: IBMExperimentService):
        if self._service:
            raise ExperimentDataError("An experiment service is already being used.")
        self._service = new_service
        with contextlib.suppress(Exception):
            self.auto_save = new_service.options.get("auto_save", False)

    @property
    def provider(self):
        """Backend provider."""
        return self._provider

    @provider.setter
    def provider(self, new_provider: Provider):
        self._provider = new_provider
        self.set_hgp_from_provider(new_provider)
        if self._service is None:
            self.service = self.get_service_from_provider(new_provider)

    @property
    def share_level(self):
        """Share level for this experiment data."""
        return self._share_level

    @share_level.setter
    def share_level(self, new_level: str):
        if new_level not in ("public", "hub", "group", "project", "private"):
            raise ExperimentDataError(f"Invalid share level {new_level}.")
        self._share_level = new_level
        if self.auto_save:
            self.save_metadata()

    @property
    def auto_save(self) -> bool:
        """If auto-save is enabled."""
        return self._auto_save

    @auto_save.setter
    def auto_save(self, new_val: bool):
        if new_val is True:
            self.save(save_children=False)
        self._auto_save = new_val

    def save_metadata(
        self,
        experiment_data: "ExperimentData",
        suppress_errors: bool = True
    ):
        """Save this experiment metadata to a database service.

        .. note::
            This method does not save analysis results nor figures.
            Use :meth:`save` for general saving of all experiment data.

            See :meth:`qiskit.providers.experiment.IBMExperimentService.create_experiment`
            for fields that are saved.

        Args:
            experiment_data: ExperimentData instance to save.
            suppress_errors: Set True to catch exceptions.

        Raises:
            QiskitError: When save fails.
        """
        metadata = experiment_data.metadata

        if not self._service:
            LOG.warning(
                "Experiment cannot be saved because no experiment service is available. "
                "An experiment service is available, for example, when using an IBM Quantum backend."
            )
            return
        try:
            total_size = sys.getsizeof(json.dumps(metadata, cls=self.json_encoder))
            use_artifact = total_size > self.metadata_size_limit

            payload = self._make_service_metadata_payload(
                experiment_data=experiment_data,
                remove_metadata=use_artifact,
            )
            res = self.service.create_or_update_experiment(
                payload,
                json_encoder=self.json_encoder,
                create=not self._already_saved,
            )
            if isinstance(res, dict):
                self.created_datetime = parse_utc_datetime(res.get("created_at", None))
                self.updated_datetime = parse_utc_datetime(res.get("updated_at", None))
            self._already_saved = True

            if use_artifact:
                self.service.file_upload(
                    experiment_id=experiment_data.experiment_id,
                    file_name=self.metadata_filename,
                    file_data=experiment_data.metadata,
                    json_encoder=self.json_encoder,
                )

        except Exception as ex:  # pylint: disable=broad-except
            LOG.error("Unable to save the experiment data: %s", traceback.format_exc())
            if not suppress_errors:
                raise QiskitError(f"Experiment data save failed\nError Message:\n{str(ex)}") from ex

    def save(
        self,
        experiment_data: ExperimentData,
    ):
        pass

    def retrieve_job_data(
        self,
        experiment_jobs: dict[str, Job],
    ) -> Iterator[Job]:
        """A helper method to retrieve job instances from provider.

        Args:
            experiment_jobs: Collection of related jobs to mutate.

        Yields:
            Retrieved job instances.
        """
        if self.provider is None or all(experiment_jobs.values()):
            return

        # first find which jobs are listed in the `job_ids` field of the experiment data
        for jid in self.job_ids:
            if jid in experiment_jobs or experiment_jobs[jid] is None:
                continue
            try:
                LOG.debug("Retrieving job [Job ID: %s]", jid)
                retrieved_job = self.provider.retrieve_job(jid)
                experiment_jobs[jid] = retrieved_job
                yield retrieved_job
            except Exception:  # pylint: disable=broad-except
                LOG.warning(
                    "Unable to retrieve data from job [Job ID: %s]",
                    jid,
                )

    def retrieve_analysis_results(
        self,
        experiment_id: str,
        analysis_results: AnalysisResultTable,
        refresh: bool = False,
    ):
        """A helper method to get stored analysis results from database service.

        Args:
            experiment_id: UID of experiment entry storing analysis results.
            analysis_results: Table of analysis results to mutate.
            refresh: Set True to overwrite existing local entries.
        """
        if self.service and (len(analysis_results) == 0 or refresh):
            retrieved_results = self.service.analysis_results(
                experiment_id=experiment_id,
                limit=None,
                json_decoder=self.json_decoder,
            )
            for result in retrieved_results:
                # Canonicalize into IBM specific data structure.
                # TODO define proper data schema on frontend and delegate this to service.
                cano_quality = AnalysisResult.RESULT_QUALITY_TO_TEXT.get(result.quality, "unknown")
                cano_components = [to_component(c) for c in result.device_components]
                extra = result.result_data["_extra"]
                if result.chisq is not None:
                    extra["chisq"] = result.chisq
                analysis_results.add_entry(
                    result_id=result.result_id,
                    name=result.result_type,
                    value=result.result_data["_value"],
                    quality=cano_quality,
                    components=cano_components,
                    experiment_id=result.experiment_id,
                    tags=result.tags,
                    backend=result.backend_name,
                    created_time=result.creation_datetime,
                    **extra,
                )

    def _make_service_metadata_payload(
        self,
        experiment_data: "ExperimentData",
        remove_metadata: bool = False,
    ):
        """A helper method to create experiment service payload from experiment data."""
        out = ServiceExpData(
            experiment_id=experiment_data.experiment_id,
            parent_id=experiment_data.parent_id,
            experiment_type=experiment_data.experiment_type,
            backend=experiment_data.backend_name or str(experiment_data.backend),
            tags=experiment_data.tags,
            job_ids=self.job_ids,
            share_level=self.share_level,
            metadata=experiment_data.metadata if not remove_metadata else {},
            figure_names=self.figure_names,
            notes=experiment_data.notes,
            hub=self.hub,
            group=self.group,
            project=self.project,
            owner=None,
            creation_datetime=self.created_datetime,
            start_datetime=experiment_data.start_datetime,
            end_datetime=experiment_data.end_datetime,
            updated_datetime=self.updated_datetime,
        )
        return out

    @classmethod
    def from_service_experiment_data(
        cls,
        experiment_data: ServiceExpData,
        provider: Provider | None = None,
        service: IBMExperimentService | None = None,
    ) -> "ExperimentServiceFrontend":

        instance = ExperimentServiceFrontend(
            provider=provider,
            service=service,
        )
        instance.created_datetime = experiment_data.creation_datetime
        instance.updated_datetime = experiment_data.updated_datetime
        instance.job_ids = experiment_data.job_ids
        instance.figure_names = experiment_data.figure_names

        instance.backend_name = experiment_data.backend
        instance.hub = experiment_data.hub
        instance.project = experiment_data.project
        instance.group = experiment_data.group
        instance.share_level = experiment_data.share_level

        return instance



#
# class ServiceExtensionMixIn:
#     """An extension of ExperimentData interface for database service APIs."""
#
#     _service_frontend: ExperimentServiceFrontend = None
#     _child_data: ThreadSafeOrderedDict
#
#     @property
#     def hub(self) -> str:
#         """Hub name of your quantum service provider."""
#         return self._service_frontend.hub
#
#     @property
#     def group(self) -> str:
#         """Group name of your quantum service provider."""
#         return self._service_frontend.group
#
#     @property
#     def project(self) -> str:
#         """Project name of your quantum service provider."""
#         return self._service_frontend.project
#
#     @property
#     def hgp(self) -> str:
#         """Account information in the formatted hub/group/project string."""
#         return self._service_frontend.hgp
#
#     @hgp.setter
#     def hgp(self, new_hgp):
#         self._service_frontend.hgp = new_hgp
#
#     @property
#     def service(self) -> IBMExperimentService | None:
#         """Experiment database service instance to store experiment data."""
#         return self._service_frontend.service
#
#     @service.setter
#     def service(self, new_service: IBMExperimentService):
#         self._service_frontend.service = new_service
#
#     @property
#     def provider(self):
#         """Backend provider."""
#         return self._service_frontend.provider
#
#     @provider.setter
#     def provider(self, new_provider: Provider):
#         self._service_frontend.provider = new_provider
#
#     @property
#     def share_level(self):
#         """Share level for this experiment data."""
#         return self._service_frontend.share_level
#
#     @share_level.setter
#     def share_level(self, new_level: str):
#         self._service_frontend.share_level = new_level
#
#     @property
#     def auto_save(self) -> bool:
#         """If auto-save is enabled."""
#         return self._service_frontend.auto_save
#
#     @auto_save.setter
#     def auto_save(self, new_val: bool):
#         self._service_frontend.auto_save = new_val
#
#     def save_metadata(
#         self,
#         suppress_errors: bool = True
#     ):
#         """Save this experiment metadata to a database service.
#
#         .. note::
#             This method does not save analysis results nor figures.
#             Use :meth:`save` for general saving of all experiment data.
#
#             See :meth:`qiskit.providers.experiment.IBMExperimentService.create_experiment`
#             for fields that are saved.
#
#         Args:
#             suppress_errors: Set True to catch exceptions.
#
#         Raises:
#             QiskitError: When save fails.
#         """
#         self._service_frontend.save_metadata(
#             experiment_data=self,
#             suppress_errors=suppress_errors,
#         )
#         for data in self._child_data.values():
#             data.save_metadata()
#
#     # def source(self) -> dict:
#     #     if not self.service_frontend.source:
#     #
#     #
#     #     return {
#     #         "class": f"{self.__class__.__module__}.{self.__class__.__name__}",
#     #         "metadata_version": self._metadata_version,
#     #         "qiskit_version": qiskit_version(),
#     #     }
