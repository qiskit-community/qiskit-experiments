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

import copy
import logging
import math
import uuid
import traceback
from typing import Optional, List, Union, Dict, Any

import uncertainties

from qiskit_ibm_experiment import IBMExperimentService, AnalysisResultData
from qiskit_ibm_experiment import ResultQuality
from qiskit_ibm_experiment.exceptions import IBMExperimentEntryExists, IBMExperimentEntryNotFound

from qiskit_experiments.framework.json import (
    ExperimentEncoder,
    ExperimentDecoder,
    _serialize_safe_float,
)

from .db_fitval import FitVal
from .device_component import DeviceComponent, to_component
from .exceptions import DbExperimentDataError
from .utils import qiskit_version

LOG = logging.getLogger(__name__)


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
        data: AnalysisResultData,
        service: Optional[IBMExperimentService] = None,
    ):
        """AnalysisResult constructor.

        Args:
            data: analysis data
            service: Experiment service to be used to store result in database.
        """
        # Data to be stored in DB.
        self._data = copy.deepcopy(data)
        if self.source is None:
            self._data.result_data["_source"] = self.default_source()
        self._service = service
        self._created_in_db = False
        self._auto_save = False
        if self._service:
            try:
                self.auto_save = self._service.preferences["auto_save"]
            except AttributeError:
                pass

    @classmethod
    def from_values(
        cls,
        name: str,
        value: Any,
        device_components: List[Union[DeviceComponent, str]],
        experiment_id: str,
        result_id: Optional[str] = None,
        chisq: Optional[float] = None,
        quality: Optional[ResultQuality] = ResultQuality.UNKNOWN,
        extra: Optional[Dict[str, Any]] = None,
        verified: bool = False,
        tags: Optional[List[str]] = None,
        service: Optional[IBMExperimentService] = None,
        source: Optional[Dict[str, str]] = None,
    ) -> "DbAnalysisResultV1":
        """AnalysisResult constructor.

        Args:
            name: analysis result name.
            value: main analysis result value.
            device_components: Target device components this analysis is for.
            experiment_id: ID of the experiment.
            result_id: Result ID. If ``None``, one is generated.
            chisq: Reduced chi squared of the fit.
            quality: Quality of the analysis. Refer to the experiment service
                provider for valid values.
            extra: Dictionary of extra analysis result data
            verified: Whether the result quality has been verified.
            tags: Tags for this analysis result.
            service: Experiment service to be used to store result in database.
            source: Class and qiskit version information when loading from an
                experiment service.
        Returns:
            The Analysis result object
        """
        # Data to be stored in DB.
        data = AnalysisResultData(
            experiment_id=experiment_id,
            result_id=result_id or str(uuid.uuid4()),
            result_type=name,
            chisq=chisq,
            quality=quality,
            verified=verified,
            tags=tags or [],
        )
        data.result_data = cls.format_result_data(value, extra, chisq, source)
        for comp in device_components:
            if isinstance(comp, str):
                comp = to_component(comp)
            data.device_components.append(comp)
        return cls(data, service)

    @classmethod
    def default_source(cls) -> Dict[str, str]:
        """The default source dictionary to generate"""
        return {
            "class": f"{cls.__module__}.{cls.__name__}",
            "data_version": cls._data_version,
            "qiskit_version": qiskit_version(),
        }

    @staticmethod
    def format_result_data(value, extra, chisq, source):
        """Formats the result data from the given arguments"""
        if source is None:
            source = DbAnalysisResultV1.default_source()
        result_data = {
            "_value": copy.deepcopy(value),
            "_chisq": chisq,
            "_extra": copy.deepcopy(extra or {}),
            "_source": source,
        }

        # Format special DB display fields
        if isinstance(value, FitVal):
            db_value = DbAnalysisResultV1._display_format(value.value)
            if db_value is not None:
                result_data["value"] = db_value
            if isinstance(value.stderr, (int, float)):
                result_data["variance"] = DbAnalysisResultV1._display_format(value.stderr**2)
            if isinstance(value.unit, str):
                result_data["unit"] = value.unit
        elif isinstance(value, uncertainties.UFloat):
            db_value = DbAnalysisResultV1._display_format(value.nominal_value)
            if db_value is not None:
                result_data["value"] = db_value
            if isinstance(value.std_dev, (int, float)):
                result_data["variance"] = DbAnalysisResultV1._display_format(value.std_dev**2)
            if "unit" in extra:
                result_data["unit"] = extra["unit"]
        else:
            db_value = DbAnalysisResultV1._display_format(value)
            if db_value is not None:
                result_data["value"] = db_value
        return result_data

    @classmethod
    def load(cls, result_id: str, service: IBMExperimentService) -> "DbAnalysisResultV1":
        """Load a saved analysis result from a database service.

        Args:
            result_id: Analysis result ID.
            service: the database service.

        Returns:
            The loaded analysis result.
        """
        # Load data from the service
        data = service.analysis_result(result_id, json_decoder=cls._json_decoder)
        result = cls(data)
        result._created_in_db = True
        return result

    def save(self) -> None:
        """Save this analysis result in the database.

        Raises:
            DbExperimentDataError: If the analysis result contains invalid data.
        """
        if not self.service:
            LOG.warning(
                "Analysis result cannot be saved because no experiment service is available."
            )
            return
        attempts = 0
        success = False
        is_new = not self._created_in_db
        try:
            while attempts < 3 and not success:
                attempts += 1
                if is_new:
                    try:
                        self.service.create_analysis_result(
                            self._data, json_encoder=self._json_encoder
                        )
                        success = True
                        self._created_in_db = True
                    except IBMExperimentEntryExists:
                        is_new = False
                else:
                    try:
                        self.service.update_analysis_result(
                            self._data, json_encoder=self._json_encoder
                        )
                        success = True
                    except IBMExperimentEntryNotFound:
                        is_new = True
        except Exception:  # pylint: disable=broad-except
            # Don't fail the experiment just because its data cannot be saved.
            LOG.error("Unable to save the experiment data: %s", traceback.format_exc())

        if not success:
            LOG.error("Unable to save the experiment data:")

    def copy(self) -> "DbAnalysisResultV1":
        """Return a copy of the result with a new result ID"""
        return DbAnalysisResultV1(data=self._data.copy(), service=self.service)

    @property
    def name(self) -> str:
        """Return analysis result name.

        Returns:
            Analysis result name.
        """
        return self._data.result_type

    @property
    def value(self) -> Any:
        """Return analysis result value.

        Returns:
            Analysis result value.
        """
        return self._data.result_data["_value"]

    @value.setter
    def value(self, new_value: Any) -> None:
        """Set the analysis result value."""
        self._data.result_data["_value"] = new_value
        if self.auto_save:
            self.save()

    @property
    def extra(self) -> Dict[str, Any]:
        """Return extra analysis result data.

        Returns:
            Additional analysis result data.
        """
        return self._extra

    @extra.setter
    def extra(self, new_value: Dict[str, Any]) -> None:
        """Set the analysis result value."""
        if not isinstance(new_value, dict):
            raise DbExperimentDataError(
                f"The `extra` field of {type(self).__name__} must be a dict."
            )
        self._extra = new_value
        if self.auto_save:
            self.save()

    @property
    def device_components(self) -> List[DeviceComponent]:
        """Return target device components for this analysis result.

        Returns:
            Target device components.
        """
        return self._data.device_components

    @device_components.setter
    def device_components(self, components: List[Union[DeviceComponent, str]]):
        """Set the device components"""
        self._data.device_components = []
        for comp in components:
            if isinstance(comp, str):
                comp = to_component(comp)
            self._data.device_components.append(comp)

    @property
    def result_id(self) -> str:
        """Return analysis result ID.

        Returns:
            ID for this analysis result.
        """
        return self._data.result_id

    @property
    def experiment_id(self) -> str:
        """Return the ID of the experiment associated with this analysis result.

        Returns:
            ID of experiment associated with this analysis result.
        """
        return self._data.experiment_id

    @property
    def chisq(self) -> Optional[float]:
        """Return the reduced χ² of this analysis."""
        return self._data.chisq

    @chisq.setter
    def chisq(self, new_chisq: float) -> None:
        """Set the reduced χ² of this analysis."""
        self._data.chisq = new_chisq
        if self.auto_save:
            self.save()

    @property
    def quality(self) -> str:
        """Return the quality of this analysis.

        Returns:
            Quality of this analysis.
        """
        return self._data.quality

    @quality.setter
    def quality(self, new_quality: str) -> None:
        """Set the quality of this analysis.

        Args:
            new_quality: New analysis quality.
        """
        self._data.quality = new_quality
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
        return self._data.verified

    @verified.setter
    def verified(self, verified: bool) -> None:
        """Set the verified flag.

        Args:
            verified: Whether the quality is verified.
        """
        self._data.verified = verified
        if self.auto_save:
            self.save()

    @property
    def tags(self):
        """Return tags associated with this result."""
        return self._data.tags

    @tags.setter
    def tags(self, new_tags: List[str]) -> None:
        """Set tags for this result."""
        if not isinstance(new_tags, list):
            raise DbExperimentDataError(
                f"The `tags` field of {type(self).__name__} must be a list."
            )
        self._data.tags = new_tags
        if self.auto_save:
            self.save()

    @property
    def service(self) -> Optional[IBMExperimentService]:
        """Return the database service.

        Returns:
            Service that can be used to store this analysis result in a database.
            ``None`` if not available.
        """
        return self._service

    @service.setter
    def service(self, service: IBMExperimentService) -> None:
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
    def source(self) -> Dict:
        """Return the class name and version."""
        if "_source" in self._data.result_data:
            return self._data.result_data["_source"]
        return None

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

    @staticmethod
    def _display_format(value):
        """Format values for supported types for display in database service"""
        if value is None or isinstance(value, (int, bool, str)):
            # Pass supported value types directly
            return value
        if isinstance(value, float):
            # Safe handling on NaN float values that serialize to invalid JSON
            if math.isfinite(value):
                return value
            else:
                return _serialize_safe_float(value)["__value__"]
        if isinstance(value, complex):
            # Convert complex floats to strings for display
            return f"{value}"
        # For all other value types that cannot be natively displayed
        # we return the class name
        return f"({type(value).__name__})"

    def __str__(self):
        ret = f"{type(self).__name__}"
        ret += f"\n- name: {self.name}"
        ret += f"\n- value: {str(self.value)}"
        if self.chisq is not None:
            ret += f"\n- χ²: {str(self.chisq)}"
        if self.quality is not None:
            ret += f"\n- quality: {self.quality}"
        if self.extra:
            ret += f"\n- extra: <{len(self.extra)} items>"
        ret += f"\n- device_components: {[str(i) for i in self.device_components]}"
        ret += f"\n- verified: {self.verified}"
        return ret

    def __repr__(self):
        out = f"{type(self).__name__}("
        out += f"name={self.name}"
        out += f", value={repr(self.value)}"
        out += f", device_components={repr(self.device_components)}"
        out += f", experiment_id={self.experiment_id}"
        out += f", result_id={self.result_id}"
        out += f", chisq={self.chisq}"
        out += f", quality={self.quality}"
        out += f", verified={self.verified}"
        out += f", extra={repr(self.extra)}"
        out += f", tags={self.tags}"
        out += f", service={repr(self.experiment_id)}"
        for key, val in self._extra_data.items():
            out += f", {key}={repr(val)}"
        out += ")"
        return out

    def __json_encode__(self):
        return {
            "data": {
                "name": self._data.name,
                "value": self._data.value,
                "device_components": self._data.device_components,
                "experiment_id": self._data.experiment_id,
                "result_id": self._data.result_id,
                "chisq": self._data.chisq,
                "quality": self._data.quality,
                "extra": self._data.extra,
                "verified": self._data.verified,
                "tags": self._data.tags,
            }
        }
