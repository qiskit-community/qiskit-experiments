# This code is part of Qiskit.
#
# (C) Copyright IBM 2021-2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Analysis result class."""

import copy
import logging
import math
import uuid
import traceback
from typing import Optional, List, Union, Dict, Any

import uncertainties

from qiskit_ibm_experiment import IBMExperimentService, AnalysisResultData
from qiskit_ibm_experiment import ResultQuality
from qiskit.exceptions import QiskitError

from qiskit_experiments.framework.json import (
    ExperimentEncoder,
    ExperimentDecoder,
    _serialize_safe_float,
)

from qiskit_experiments.database_service.device_component import DeviceComponent, to_component
from qiskit_experiments.database_service.exceptions import ExperimentDataError
from qiskit_experiments.framework.package_deps import qiskit_version

LOG = logging.getLogger(__name__)


class AnalysisResult:
    """Class representing an analysis result for an experiment.

    Analysis results can also be stored using the experiments service.

    The field ``db_data`` is a dataclass (`ExperimentDataclass`) containing
    all the data that can be stored with the service and loaded from it, and
    as such is subject to strict conventions.

    Other data fields can be added and used freely, but they won't be saved
    to the database.

    Note that the ``result_data`` field of the dataclass is by itself a dictionary
    capable of holding arbitrary values (in a dictionary indexed by a string).

    The data fields in the ``db_data`` dataclass are:

    * ``experiment_id``: ``str``
    * ``result_id``: ``str``
    * ``result_type``: ``str``
    * ``device_components``: ``List[str]``
    * ``quality``: ``str``
    * ``verified``: ``bool``
    * ``tags``: ``List[str]``
    * ``backend_name``: ``str``
    * ``chisq``: ``float``
    * ``result_data``: ``Dict[str]``

    Analysis data that does not fit into the other fields should be added to
    the ``result_data`` dict, e.g. curve parameters in experiments doing a curve fit.
    """

    version = 1
    _data_version = 1

    _json_encoder = ExperimentEncoder
    _json_decoder = ExperimentDecoder

    _extra_data = {}

    RESULT_QUALITY_TO_TEXT = {
        ResultQuality.GOOD: "good",
        ResultQuality.BAD: "bad",
        ResultQuality.UNKNOWN: "unknown",
    }

    def __init__(
        self,
        name: str = None,
        value: Any = None,
        device_components: List[Union[DeviceComponent, str]] = None,
        experiment_id: str = None,
        result_id: Optional[str] = None,
        chisq: Optional[float] = None,
        quality: Optional[str] = RESULT_QUALITY_TO_TEXT[ResultQuality.UNKNOWN],
        extra: Optional[Dict[str, Any]] = None,
        verified: bool = False,
        tags: Optional[List[str]] = None,
        service: Optional[IBMExperimentService] = None,
        source: Optional[Dict[str, str]] = None,
    ) -> "AnalysisResult":
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
            source: Class and Qiskit version information when loading from an
                experiment service.
        Returns:
            The AnalysisResult object.
        """
        # Data to be stored in DB.
        self._db_data = AnalysisResultData(
            experiment_id=experiment_id,
            result_id=result_id or str(uuid.uuid4()),
            result_type=name,
            chisq=chisq,
            quality=quality,
            verified=verified,
            tags=tags or [],
        )
        self._db_data.result_data = self.format_result_data(value, extra, chisq, source)
        if device_components is not None:
            for comp in device_components:
                if isinstance(comp, str):
                    comp = to_component(comp)
                self._db_data.device_components.append(comp)

        self._service = service
        self._created_in_db = False
        self._auto_save = False
        if self._service:
            try:
                self.auto_save = self._service.preferences["auto_save"]
            except AttributeError:
                pass

    def set_data(self, data: AnalysisResultData):
        """Sets the analysis data stored in the class"""
        self._db_data = data
        new_device_components = [to_component(comp) for comp in self._db_data.device_components]
        self._db_data.device_components = new_device_components
        self._db_data.quality = self.RESULT_QUALITY_TO_TEXT.get(self._db_data.quality, "unknown")

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
            source = AnalysisResult.default_source()
        result_data = {
            "_value": copy.deepcopy(value),
            "_chisq": chisq,
            "_extra": copy.deepcopy(extra or {}),
            "_source": source,
        }

        # Format special DB display fields
        if isinstance(value, uncertainties.UFloat):
            db_value = AnalysisResult._display_format(value.nominal_value)
            if db_value is not None:
                result_data["value"] = db_value
            if isinstance(value.std_dev, (int, float)):
                result_data["variance"] = AnalysisResult._display_format(value.std_dev**2)
            if "unit" in result_data["_extra"]:
                result_data["unit"] = result_data["_extra"]["unit"]
        else:
            db_value = AnalysisResult._display_format(value)
            if db_value is not None:
                result_data["value"] = db_value
        return result_data

    @classmethod
    def load(cls, result_id: str, service: IBMExperimentService) -> "AnalysisResult":
        """Load a saved analysis result from a database service.

        Args:
            result_id: Analysis result ID.
            service: the database service.

        Returns:
            The loaded analysis result.
        """
        # Load data from the service
        data = service.analysis_result(result_id, json_decoder=cls._json_decoder)
        result = cls()
        result.set_data(data)
        result._created_in_db = True
        result.service = service
        return result

    def save(self, suppress_errors: bool = True) -> None:
        """Save this analysis result in the database.

        Args:
            suppress_errors: should the method catch exceptions (true) or
                pass them on, potentially aborting the experiment (false).
        Raises:
            ExperimentDataError: If the analysis result contains invalid data.
            QiskitError: If the save to the database failed.
        """
        if not self.service:
            LOG.warning(
                "Analysis result cannot be saved because no experiment service is available."
            )
            return
        try:
            self.service.create_or_update_analysis_result(
                self._db_data, json_encoder=self._json_encoder, create=not self._created_in_db
            )
            self._created_in_db = True
        except Exception as ex:  # pylint: disable=broad-except
            # Don't automatically fail the experiment just because its data cannot be saved.
            LOG.error("Unable to save the experiment data: %s", traceback.format_exc())
            if not suppress_errors:
                raise QiskitError(f"Analysis result save failed\nError Message:\n{str(ex)}") from ex

    def copy(self) -> "AnalysisResult":
        """Return a copy of the result with a new result ID"""
        new_instance = AnalysisResult(
            name=self.name,
            value=self.value,
            device_components=self.device_components,
            experiment_id=self.experiment_id,
        )
        new_instance._db_data = self._db_data.copy()
        new_instance._db_data.result_id = str(uuid.uuid4())
        return new_instance

    @property
    def name(self) -> str:
        """Return analysis result name.

        Returns:
            Analysis result name.
        """
        return self._db_data.result_type

    @property
    def value(self) -> Any:
        """Return analysis result value.

        Returns:
            Analysis result value.
        """
        return self._db_data.result_data["_value"]

    @value.setter
    def value(self, new_value: Any) -> None:
        """Set the analysis result value."""
        self._db_data.result_data["_value"] = new_value
        if self.auto_save:
            self.save()

    @property
    def extra(self) -> Dict[str, Any]:
        """Return extra analysis result data.

        Returns:
            Additional analysis result data.
        """
        return self._db_data.result_data["_extra"]

    @extra.setter
    def extra(self, new_value: Dict[str, Any]) -> None:
        """Set the analysis result value."""
        if not isinstance(new_value, dict):
            raise ExperimentDataError(f"The `extra` field of {type(self).__name__} must be a dict.")
        self._db_data.result_data["_extra"] = new_value
        if self.auto_save:
            self.save()

    @property
    def device_components(self) -> List[DeviceComponent]:
        """Return target device components for this analysis result.

        Returns:
            Target device components.
        """
        return self._db_data.device_components

    @device_components.setter
    def device_components(self, components: List[Union[DeviceComponent, str]]):
        """Set the device components"""
        self._db_data.device_components = []
        for comp in components:
            if isinstance(comp, str):
                comp = to_component(comp)
            self._db_data.device_components.append(comp)

    @property
    def result_id(self) -> str:
        """Return analysis result ID.

        Returns:
            ID for this analysis result.
        """
        return self._db_data.result_id

    @property
    def experiment_id(self) -> str:
        """Return the ID of the experiment associated with this analysis result.

        Returns:
            ID of experiment associated with this analysis result.
        """
        return self._db_data.experiment_id

    @experiment_id.setter
    def experiment_id(self, new_id: str) -> None:
        """Sets the experiment id"""
        self._db_data.experiment_id = new_id

    @property
    def chisq(self) -> Optional[float]:
        """Return the reduced χ² of this analysis."""
        return self._db_data.chisq

    @chisq.setter
    def chisq(self, new_chisq: float) -> None:
        """Set the reduced χ² of this analysis."""
        self._db_data.chisq = new_chisq
        if self.auto_save:
            self.save()

    @property
    def quality(self) -> str:
        """Return the quality of this analysis.

        Returns:
            Quality of this analysis.
        """
        return self._db_data.quality

    @quality.setter
    def quality(self, new_quality: str) -> None:
        """Set the quality of this analysis.

        Args:
            new_quality: New analysis quality.
        """
        self._db_data.quality = new_quality
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
        return self._db_data.verified

    @verified.setter
    def verified(self, verified: bool) -> None:
        """Set the verified flag.

        Args:
            verified: Whether the quality is verified.
        """
        self._db_data.verified = verified
        if self.auto_save:
            self.save()

    @property
    def tags(self):
        """Return tags associated with this result."""
        return self._db_data.tags

    @tags.setter
    def tags(self, new_tags: List[str]) -> None:
        """Set tags for this result."""
        if not isinstance(new_tags, list):
            raise ExperimentDataError(f"The `tags` field of {type(self).__name__} must be a list.")
        self._db_data.tags = new_tags
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
            ExperimentDataError: If an experiment service is already being used.
        """
        if self._service:
            raise ExperimentDataError("An experiment service is already being used.")
        self._service = service

    @property
    def source(self) -> Dict:
        """Return the class name and version."""
        if "_source" in self._db_data.result_data:
            return self._db_data.result_data["_source"]
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
        out += f", experiment id={repr(self.experiment_id)}"
        for key, val in self._extra_data.items():
            out += f", {key}={repr(val)}"
        out += ")"
        return out

    def __json_encode__(self):
        return {
            "name": self._db_data.result_type,
            "value": self._db_data.result_data.get("_value", None),
            "device_components": self._db_data.device_components,
            "experiment_id": self._db_data.experiment_id,
            "result_id": self._db_data.result_id,
            "chisq": self._db_data.chisq,
            "quality": self._db_data.quality,
            "extra": self.extra,
            "verified": self._db_data.verified,
            "tags": self._db_data.tags,
            "source": self.source,
        }
