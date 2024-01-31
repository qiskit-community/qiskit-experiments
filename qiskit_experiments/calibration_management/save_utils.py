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

.. warning::

    This module is expected to be internal and is not intended as a stable user-facing API.

.. note::

    Because a locally saved :class:`.Calibrations` instance may not conform to the
    data model of the latest Qiskit Experiments implementation, the calibration loader
    must be aware of the data model version.
    CalibrationModel classes representing the data model must have
    the version suffix, e.g., `CalibrationModelV1` and the `schema_version` field.
    This helps the loader to raise user-friendly error rather than being crashed by
    incompatible data, and possibly to dispatch the loader function based on the version number.

    When a developer refactors the :class:`.Calibrations` class to a new data model,
    the developer must also define a corresponding CalibrationModel class with new version number.
    Existing CalibrationModel classes should be preserved for backward compatibility.


.. note::

    We don't guarantee the portability of stored data across different Qiskit Experiments
    versions. We allow the calibration loader to raise an error for non-supported
    data models.

"""

from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import List, Dict, Any

from qiskit import QuantumCircuit
from qiskit.circuit import Instruction
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
    """Name of the parameter."""

    qubits: List[int]
    """List of associated qubits."""

    schedule: str = ""
    """Associated schedule name."""

    value: float = 0.0
    """Parameter value."""

    datetime: datetime = None
    """A date time at which this value is obtained."""

    valid: bool = True
    """If this parameter is valid."""

    exp_id: str = ""
    """Associated experiment ID which is used to obtain this value."""

    group: str = ""
    """Name of calibration group in which this calibration parameter belongs to."""

    @classmethod
    def apply_schema(cls, data: Dict[str, Any]):
        """Consume dictionary and returns canonical data model."""
        return ParameterModelV1(**data)


@dataclass
class CalibrationModelV1:
    """A data schema to represent instances of Calibrations.

    .. note::
        This is intentionally agnostic to the data structure of
        Qiskit Experiments Calibrations for portability.

    """

    backend_name: str
    """Name of the backend."""

    backend_version: str
    """Version of the backend."""

    device_coupling_graph: List[List[int]]
    """Qubit coupling graph of the device."""

    control_channel_map: ControlChannelMap
    """Mapping of ControlChannel to qubit index."""

    schedules: List[ScheduleBlock] = field(default_factory=list)
    """Template schedules. It must contain the metadata for qubits and num_qubits."""

    parameters: List[ParameterModelV1] = field(default_factory=list)
    """List of calibrated pulse parameters."""

    schedule_free_parameters: QuantumCircuit = field(default_factory=lambda: QuantumCircuit(1))
    """Placeholder circuit for parameters not associated with a schedule

    The circuit contains placeholder instructions which have the Parameter
    objects attached and operate on the qubits that the parameter is associated
    with in the calibrations.
    """

    schema_version: str = "1.0"
    """Version of this data model. This must be static."""

    @classmethod
    def apply_schema(cls, data: Dict[str, Any]):
        """Consume dictionary and returns canonical data model."""
        in_data = {}
        for key, value in data.items():
            if key == "parameters":
                value = list(map(ParameterModelV1.apply_schema, value))
            in_data[key] = value
        return CalibrationModelV1(**in_data)


def calibrations_to_dict(
    cals: Calibrations,
    most_recent_only: bool = True,
) -> Dict[str, Any]:
    """A helper function to convert calibration data into dictionary.

    Args:
        cals: A calibration instance to save.
        most_recent_only: Set True to save calibration parameters with most recent time stamps.

    Returns:
        Canonical calibration data in dictionary format.
    """
    schedules = getattr(cals, "_schedules")
    num_qubits = getattr(cals, "_schedules_qubits")
    parameters = getattr(cals, "_params")
    if most_recent_only:
        # Get values with most recent time stamps.
        parameters = {k: [max(parameters[k], key=lambda x: x.date_time)] for k in parameters}

    data_entries = []
    for param_key, param_values in parameters.items():
        for param_value in param_values:
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
        if "qubits" not in sched_obj.metadata or "num_qubits" not in sched_obj.metadata:
            qubit_metadata = {
                "qubits": sched_key.qubits,
                "num_qubits": num_qubits[sched_key],
            }
            sched_obj.metadata.update(qubit_metadata)
        sched_entries.append(sched_obj)

    max_qubit = max(
        (max(k.qubits or (0,)) for k in cals._parameter_map if k.schedule is None),
        default=0,
    )
    schedule_free_parameters = QuantumCircuit(max_qubit + 1)
    for sched_key, param in cals._parameter_map.items():
        if sched_key.schedule is None:
            schedule_free_parameters.append(
                Instruction("parameter_container", len(sched_key.qubits), 0, [param]),
                sched_key.qubits,
            )

    model = CalibrationModelV1(
        backend_name=cals.backend_name,
        backend_version=cals.backend_version,
        device_coupling_graph=getattr(cals, "_coupling_map"),
        control_channel_map=ControlChannelMap(getattr(cals, "_control_channel_map")),
        schedules=sched_entries,
        parameters=data_entries,
        schedule_free_parameters=schedule_free_parameters,
    )

    return asdict(model)


def calibrations_from_dict(
    cal_data: Dict[str, Any],
) -> Calibrations:
    """A helper function to build calibration instance from canonical dictionary.

    Args:
        cal_data: Calibration data dictionary which is formatted according to the
            predefined data schema provided by Qiskit Experiments.
            This formatting is implicitly performed when the calibration data is
            dumped into dictionary with the :func:`calibrations_to_dict` function.

    Returns:
        Calibration instance.

    Raises:
        ValueError: When input data model version is not supported.
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
        qubits = sched.metadata.pop("qubits", tuple())
        num_qubits = sched.metadata.pop("num_qubits", None)
        cals.add_schedule(
            schedule=sched,
            qubits=qubits if qubits and len(qubits) != 0 else None,
            num_qubits=num_qubits,
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

    for instruction in model.schedule_free_parameters.data:
        # For some reason, pylint thinks the items in data are tuples instead
        # of CircuitInstruction. Remove the following line if it ever stops
        # thinking that:
        # pylint: disable=no-member
        for param in instruction.operation.params:
            cals._register_parameter(param, instruction.qubits)

    cals.update_inst_map()

    return cals
