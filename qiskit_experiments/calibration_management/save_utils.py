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

"""A helper module to save calibration data to local storage.

These model version must be upgraded when :class:`.Calibrations` data structure changes
and current data model no longer supports the upgraded data structure.
"""

from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import List, Dict, Any

from qiskit.pulse import ScheduleBlock

from .calibrations import Calibrations
from .control_channel_map import ControlChannelMap
from .parameter_value import ParameterValue


@dataclass(frozen=True)
class ParameterModelV1:
    """A data schema of a single calibrated parameter.

    .. note::
        This is intentionally agnostic to the data structure of
        Qiskit Experiments Calibrations for portability.

    """

    param_name: str
    qubits: List[int]
    schedule: str = ""
    value: float = 0.0
    datetime: datetime = datetime.now()
    valid: bool = True
    exp_id: str = ""
    group: str = ""

    @classmethod
    def apply_schema(cls, data: Dict[str, Any]):
        """Consume dictionary and returns canonical data model."""
        return ParameterModelV1(**data)


@dataclass(frozen=True)
class ScheduleModelV1:
    """A data schema of a single templated schedule.

    .. note::
        This is intentionally agnostic to the data structure of
        Qiskit Experiments Calibrations for portability.

    """

    data: ScheduleBlock
    qubits: List[int] = field(default_factory=list)
    num_qubits: int = 0

    @classmethod
    def apply_schema(cls, data: Dict[str, Any]):
        """Consume dictionary and returns canonical data model."""
        return ScheduleModelV1(**data)


@dataclass
class CalibrationModelV1:
    """A data schema of whole device representation.

    .. note::
        This is intentionally agnostic to the data structure of
        Qiskit Experiments Calibrations for portability.

    """

    backend_name: str
    backend_version: str
    device_coupling_graph: List[List[int]]
    control_channel_map: ControlChannelMap
    schedules: List[ScheduleModelV1] = field(default_factory=list)
    parameters: List[ParameterModelV1] = field(default_factory=list)
    schema_version: str = "1.0"

    @classmethod
    def apply_schema(cls, data: Dict[str, Any]):
        """Consume dictionary and returns canonical data model."""
        in_data = {}
        for key, value in data.items():
            if key == "schedules":
                value = list(map(ScheduleModelV1.apply_schema, value))
            if key == "parameters":
                value = list(map(ParameterModelV1.apply_schema, value))
            in_data[key] = value
        return CalibrationModelV1(**in_data)


def to_dict(
    cals: Calibrations,
    most_recent_only: bool = True,
) -> Dict[str, Any]:
    """A helper function to convert calibration data into dictionary.

    Args:
        cals: A calibration instance to save.
        most_recent_only: Set True to save calibration parameters with most recent time stamps.

    Returns:
        Canonicalized calibration data in dictionary format.
    """
    # This can dispatch canonicalize function as version evolves.
    model = _canonicalize_calibration_data_v1(cals, most_recent_only)
    return asdict(model)


def from_dict(
    cal_data: Dict[str, Any],
) -> Calibrations:
    """A helper function to build calibration instance from canonical dictionary.

    Args:
        cal_data: Calibration data that conforms to the predefined data schema.

    Returns:
        Calibration instance.

    Raises:
        ValueError: When input data model version is no longer supported.
        KeyError: When input data model doesn't conform to data schema.
    """
    # Apply schema for data field validation
    try:
        version = cal_data["schema_version"]
        if version == "1.0":
            model = CalibrationModelV1.apply_schema(cal_data)
        else:
            raise ValueError(
                f"Loading calibration data with schema version {version} is no longer supported. "
                "Use the same version of Qiskit Experiments at the time of saving."
            )
    except (KeyError, TypeError) as ex:
        raise KeyError(
            "Loaded data doesn't match with the defined data schema. "
            "Check if this object is dumped from the Calibrations instance."
        ) from ex

    # This can dispatch loading mechanism depending on schema version
    cals = Calibrations(
        coupling_map=model.device_coupling_graph,
        control_channel_map=model.control_channel_map.chan_map,
        backend_name=model.backend_name,
        backend_version=model.backend_version,
    )

    # Add schedules
    for sched in model.schedules:
        cals.add_schedule(
            schedule=sched.data,
            qubits=sched.qubits if len(sched.qubits) != 0 else None,
            num_qubits=sched.num_qubits,
        )

    # Add parameters
    for param in model.parameters:
        param_value = ParameterValue(
            value=param.value,
            date_time=param.datetime,
            valid=param.valid,
            exp_id=param.exp_id,
            group=param.group,
        )
        cals.add_parameter_value(
            value=param_value,
            param=param.param_name,
            qubits=tuple(param.qubits),
            schedule=param.schedule,
            update_inst_map=False,
        )
    cals.update_inst_map()

    return cals


def _canonicalize_calibration_data_v1(
    cals: Calibrations,
    most_recent_only: bool = True,
) -> CalibrationModelV1:
    """A helper function to canonicalize calibration data.

    Args:
        cals: Calibration instance to apply data model.
        most_recent_only: Set True to save calibration parameters with most recent time stamps.

    Returns:
        Canonical calibration data.
    """
    schedules = getattr(cals, "_schedules")
    num_qubits = getattr(cals, "_schedules_qubits")
    parameters = getattr(cals, "_params")
    if most_recent_only:
        # Get values with most recent time stamps.
        parameters = {k: max(parameters[k], key=lambda x: x.date_time) for k in parameters}

    data_entries = []
    for param_key, param_value in parameters.items():
        entry = ParameterModelV1(
            param_name=param_key.parameter,
            qubits=param_key.qubits,
            schedule=param_key.schedule,
            value=param_value.value,
            datetime=param_value.date_time,
            valid=param_value.valid,
            exp_id=param_value.exp_id,
            group=param_value.group,
        )
        data_entries.append(entry)

    sched_entries = []
    for sched_key, sched_obj in schedules.items():
        entry = ScheduleModelV1(
            data=sched_obj,
            qubits=sched_key.qubits or [],
            num_qubits=num_qubits[sched_key],
        )
        sched_entries.append(entry)

    calibration_data = CalibrationModelV1(
        backend_name=cals.backend_name,
        backend_version=cals.backend_version,
        device_coupling_graph=getattr(cals, "_coupling_map"),
        control_channel_map=ControlChannelMap(getattr(cals, "_control_channel_map")),
        schedules=sched_entries,
        parameters=data_entries,
    )

    return calibration_data
