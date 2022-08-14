# This code is part of Qiskit.
#
# (C) Copyright IBM 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Utility functions for calibration management."""

from typing import Dict, Set, Tuple
import regex as re

from qiskit.circuit import ParameterExpression, Parameter
from qiskit.pulse import ScheduleBlock

from qiskit_experiments.exceptions import CalibrationError
from qiskit_experiments.calibration_management.calibration_key_types import ParameterKey


def validate_channels(schedule: ScheduleBlock) -> Set[Parameter]:
    """Validate amd get the parameters in the channels of the schedule.

    Note that channels implicitly defined in references will be ignored.

    Args:
        schedule: The schedule for which to get the parameters in the channels.

    Returns:
        The set of parameters explicitly defined in the schedule.

    Raises:
        CalibrationError: If a channel is parameterized by more than one parameter.
        CalibrationError: If the parameterized channel index is not formatted properly.
    """

    # The channel indices need to be parameterized following this regex.
    __channel_pattern__ = r"^ch\d[\.\d]*\${0,1}[\d]*$"

    param_indices = set()

    # Schedules with references do not explicitly have channels. This needs special handling.
    if schedule.is_referenced():
        for block in schedule.blocks:
            if isinstance(block, ScheduleBlock):
                param_indices.update(validate_channels(block))

        return param_indices

    for ch in schedule.channels:
        if isinstance(ch.index, ParameterExpression):
            if len(ch.index.parameters) != 1:
                raise CalibrationError(f"Channel {ch} can only have one parameter.")

            param_indices.add(ch.index)
            if re.compile(__channel_pattern__).match(ch.index.name) is None:
                raise CalibrationError(
                    f"Parameterized channel must correspond to {__channel_pattern__}"
                )

    return param_indices


def standardize_assignment_parameters(assign_params, qubits, name) -> Tuple[Dict, Dict]:
    """Standardize the format of manually specified parameter assignments.

    TODO

    Raises:
        CalibrationError: If the assign_params contains parameter assignments for
            schedule references and the key is not a tuple of reference name and
            corresponding qubits.
    """
    assign_params_ = dict()
    ref_assign_params_ = dict()

    if assign_params:

        for assign_param, value in assign_params.items():
            if isinstance(value, dict):
                # Test that the key is a string followed by qubits tuple.
                if not isinstance(assign_param[0], str) and not isinstance(assign_param[0], tuple):
                    raise CalibrationError(
                        "Parameter reference assignment should be tuple of reference name "
                        f"and qubits. Found {assign_param}"
                    )

                ref_assign_params_[assign_param] = value
            elif isinstance(assign_param, str):
                assign_params_[ParameterKey(assign_param, qubits, name)] = value
            else:
                assign_params_[ParameterKey(*assign_param)] = value

        assign_params = assign_params_

    return assign_params_, ref_assign_params_
