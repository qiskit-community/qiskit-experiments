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

"""Analysis result abstract interface."""

import logging
from typing import Optional, List, Union, Dict, Callable, Any
import uuid
import json
import copy
from functools import wraps

from .database_service import DatabaseServiceV1
from .json import ExperimentEncoder, ExperimentDecoder
from .utils import save_data, qiskit_version
from .exceptions import DbExperimentDataError
from .device_component import DeviceComponent, to_component

LOG = logging.getLogger(__name__)


def do_auto_save(func: Callable):
    """Decorate the input function to auto save data."""

    @wraps(func)
    def _wrapped(self, *args, **kwargs):
        return_val = func(self, *args, **kwargs)
        if self.auto_save:
            self.save()
        return return_val

    return _wrapped


class DbAnalysisResult:
    """Base common type for all versioned DbAnalysisResult abstract classes.

    Note this class should not be inherited from directly, it is intended
    to be used for type checking. When implementing a custom DbAnalysisResult
    you should use the versioned classes as the parent class and not this class
    directly.
    """

    version = 0


class DbAnalysisResultV1(DbAnalysisResult):
    """Class representing an analysis result for an experiment.

    Analysis results can also be stored in a database.
    """

    version = 1
    _data_version = 1

    _json_encoder = ExperimentEncoder
    _json_decoder = ExperimentDecoder

    _extra_data = {}

    def __init__(
        self,
        result_data: Dict,
        result_type: str,
        device_components: List[Union[DeviceComponent, str]],
        experiment_id: str,
        result_id: Optional[str] = None,
        chisq: Optional[float] = None,
        quality: Optional[str] = None,
        verified: bool = False,
        tags: Optional[List[str]] = None,
        service: Optional[DatabaseServiceV1] = None,
        **kwargs,
    ):
        """AnalysisResult constructor.

        Args:
            result_data: Analysis result data.
            result_type: Analysis result type.
            device_components: Target device components this analysis is for.
            experiment_id: ID of the experiment.
            result_id: Result ID. If ``None``, one is generated.
            chisq: Reduced chi squared of the fit.
            quality: Quality of the analysis. Refer to the experiment service
                provider for valid values.
            verified: Whether the result quality has been verified.
            tags: Tags for this analysis result.
            service: Experiment service to be used to store result in database.
            **kwargs: Additional analysis result attributes.
        """
        result_data = result_data or {}
        self._result_data = copy.deepcopy(result_data)
        self._source = self._result_data.pop(
            "_source",
            {
                "class": f"{self.__class__.__module__}.{self.__class__.__name__}",
                "data_version": self._data_version,
                "qiskit_version": qiskit_version(),
            },
        )

        # Data to be stored in DB.
        self._experiment_id = experiment_id
        self._id = result_id or str(uuid.uuid4())
        self._type = result_type
        self._device_components = []
        for comp in device_components:
            if isinstance(comp, str):
                comp = to_component(comp)
            self._device_components.append(comp)

        self._chisq = chisq
        self._quality = quality
        self._quality_verified = verified
        self._tags = tags or []

        # Other attributes.
        self._service = service
        self._created_in_db = False
        self._auto_save = False
        if self._service:
            try:
                self.auto_save = self._service.option("auto_save")
            except AttributeError:
                pass
        self._extra_data = kwargs

    @classmethod
    def load(cls, result_id: str, service: DatabaseServiceV1) -> "DbAnalysisResultV1":
        """Load a saved analysis result from a database service.

        Args:
            result_id: Analysis result ID.
            service: the database service.

        Returns:
            The loaded analysis result.
        """
        # Load data from the service
        instance = cls._from_service_data(service.analysis_result(result_id))
        instance._created_in_db = True
        return instance

    def save(self) -> None:
        """Save this analysis result in the database.

        Raises:
            DbExperimentDataError: If the analysis result contains invalid data.
        """
        if not self._service:
            LOG.warning(
                "Analysis result cannot be saved because no experiment service is available."
            )
            return

        _result_data = copy.deepcopy(self._result_data)
        _result_data["_source"] = self._source

        new_data = {
            "experiment_id": self._experiment_id,
            "result_type": self.result_type,
            "device_components": self.device_components,
        }
        update_data = {
            "result_id": self.result_id,
            "result_data": _result_data,
            "tags": self.tags(),
            "chisq": self._chisq,
            "quality": self.quality,
            "verified": self.verified,
        }

        self._created_in_db, _ = save_data(
            is_new=(not self._created_in_db),
            new_func=self._service.create_analysis_result,
            update_func=self._service.update_analysis_result,
            new_data=new_data,
            update_data=update_data,
            json_encoder=self._json_encoder
        )

    @classmethod
    def _from_service_data(cls, service_data: Dict) -> "DbAnalysisResultV1":
        """Construct an analysis result from saved database service data.

        Args:
            service_data: Analysis result data.

        Returns:
            The loaded analysis result.
        """
        # Parse serialized data
        result_data = service_data.pop("result_data")
        if result_data:
            result_data = json.loads(json.dumps(result_data), cls=cls._json_decoder)
        # Initialize the result object
        return cls(
            result_data,
            result_type=service_data.pop("result_type"),
            device_components=service_data.pop("device_components"),
            experiment_id=service_data.pop("experiment_id"),
            result_id=service_data.pop("result_id"),
            quality=service_data.pop("quality"),
            verified=service_data.pop("verified"),
            tags=service_data.pop("tags"),
            service=service_data.pop("service"),
            **service_data,
        )

    def data(self) -> Dict:
        """Return analysis result data.

        Returns:
            Analysis result data.
        """
        return self._result_data

    @do_auto_save
    def set_data(self, new_data: Dict) -> None:
        """Set result data.

        Args:
            new_data: New analysis result data.
        """
        self._result_data = new_data

    def tags(self):
        """Return tags associated with this result."""
        return self._tags

    @do_auto_save
    def set_tags(self, new_tags: List[str]) -> None:
        """Set tags for this result.

        Args:
            new_tags: New tags for the result.
        """
        self._tags = new_tags

    @property
    def result_id(self) -> str:
        """Return analysis result ID.

        Returns:
            ID for this analysis result.
        """
        return self._id

    @property
    def result_type(self) -> str:
        """Return analysis result type.

        Returns:
            Analysis result type.
        """
        return self._type

    @property
    def source(self) -> Dict:
        """Return the class name and version."""
        return self._source

    @property
    def chisq(self) -> Optional[float]:
        """Return the reduced χ² of this analysis."""
        return self._chisq

    @chisq.setter
    def chisq(self, new_chisq: float) -> None:
        """Set the reduced χ² of this analysis."""
        self._chisq = new_chisq
        if self._auto_save:
            self.save()

    @property
    def quality(self) -> str:
        """Return the quality of this analysis.

        Returns:
            Quality of this analysis.
        """
        return self._quality

    @quality.setter
    def quality(self, new_quality: str) -> None:
        """Set the quality of this analysis.

        Args:
            new_quality: New analysis quality.
        """
        self._quality = new_quality
        if self.auto_save:
            self.save()

    @property
    def verified(self) -> bool:
        """Return the verified flag.

        The ``verified`` flag is intended to indicate whether the quality
        value has been verified by a human.

        Returns:
            Whether the quality has been verified.
        """
        return self._quality_verified

    @verified.setter
    def verified(self, verified: bool) -> None:
        """Set the verified flag.

        Args:
            verified: Whether the quality is verified.
        """
        self._quality_verified = verified
        if self.auto_save:
            self.save()

    @property
    def experiment_id(self) -> str:
        """Return the ID of the experiment associated with this analysis result.

        Returns:
            ID of experiment associated with this analysis result.
        """
        return self._experiment_id

    @property
    def service(self) -> Optional[DatabaseServiceV1]:
        """Return the database service.

        Returns:
            Service that can be used to store this analysis result in a database.
            ``None`` if not available.
        """
        return self._service

    @service.setter
    def service(self, service: DatabaseServiceV1) -> None:
        """Set the service to be used for storing result data in a database.

        Args:
            service: Service to be used.

        Raises:
            DbExperimentDataError: If an experiment service is already being used.
        """
        if self._service:
            raise DbExperimentDataError("An experiment service is already being used.")
        self._service = service

    @property
    def auto_save(self) -> bool:
        """Return current auto-save option.

        Returns:
            Whether changes will be automatically saved.
        """
        return self._auto_save

    @auto_save.setter
    def auto_save(self, save_val: bool) -> None:
        """Set auto save preference.

        Args:
            save_val: Whether to do auto-save.
        """
        if save_val and not self._auto_save:
            self.save()
        self._auto_save = save_val

    @property
    def device_components(self) -> List[DeviceComponent]:
        """Return target device components for this analysis result.

        Returns:
            Target device components.
        """
        return self._device_components

    def __getattr__(self, name: str) -> Any:
        try:
            return self._extra_data[name]
        except KeyError:
            # pylint: disable=raise-missing-from
            raise AttributeError("Attribute %s is not defined" % name)

    def __str__(self):
        ret = f"\nAnalysis Result: {self.result_type}"
        ret += f"\nAnalysis Result ID: {self.result_id}"
        ret += f"\nExperiment ID: {self.experiment_id}"
        ret += f"\nDevice Components: {self.device_components}"
        if self.chisq:
            ret += f"\nχ²: {self.chisq}"
        ret += f"\nQuality: {self.quality}"
        ret += f"\nVerified: {self.verified}"
        if self.tags():
            ret += f"\nTags: {self.tags()}"
        ret += "\nResult Data:"
        ret += str(self.data())

        return ret
