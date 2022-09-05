# This code is part of Qiskit.
#
# (C) Copyright IBM 2019-2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Calibration helper functions"""

from typing import Optional, Set, Tuple
from functools import lru_cache
import regex as re
import retworkx as rx

from qiskit.circuit import ParameterExpression, Parameter
from qiskit.pulse import ScheduleBlock

from qiskit_experiments.exceptions import CalibrationError


def update_schedule_dependency(schedule: ScheduleBlock, dag: rx.PyDiGraph):
    """Update a DAG of schedule dependencies.

    Args:
        schedule: A ScheduleBlock that potentially has references to other schedules
            that are already present in the dag.
        dag: A directed acyclic graph that encodes schedule dependencies using references.
    """
    parent_idx = dag.add_node(schedule.name)
    for reference in schedule.references:
        dag.add_edge(parent_idx, dag.nodes().index(reference[0]), None)


def used_in_references(schedule_names: Set[str], dag: rx.PyDiGraph) -> Set[str]:
    """Find all the schedules in the DAG that reference the given schedules.

    Args:
        schedule_names: A list of schedules to which references may exist.
        dag: The dag that represents the dependencies between schedule references.

    Returns:
        A set of schedules that reference the given schedules.
    """
    callers = set()

    for name in schedule_names:
        callers.update(_referred_by(name, dag))

    return callers


def _referred_by(schedule_name: str, dag: rx.PyDiGraph) -> Set[str]:
    """Return all the schedules that refer to this schedule by name."""
    referred_by = set()
    for predecessor in dag.predecessors(dag.nodes().index(schedule_name)):
        referred_by.add(predecessor)
        referred_by.update(_referred_by(predecessor, dag))

    return referred_by


def validate_channels(schedule: ScheduleBlock) -> Set[Parameter]:
    """Validate and get the parameters in the channels of the schedule.

    Channels implicitly defined in references are ignored.

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

    regex = re.compile(__channel_pattern__)
    for ch in schedule.channels:
        if isinstance(ch.index, ParameterExpression):
            if len(ch.index.parameters) != 1:
                raise CalibrationError(f"Channel {ch} can only have one parameter.")

            param_indices.add(ch.index)
            if regex.match(ch.index.name) is None:
                raise CalibrationError(
                    f"Parameterized channel must correspond to {__channel_pattern__}"
                )

    return param_indices


@lru_cache
def reference_info(
    reference: Tuple[str, ...],
    qubits: Optional[Tuple[int, ...]] = None,
) -> Tuple[str, Tuple[int, ...]]:
    """Extract reference information from the reference tuple.

    Args:
        reference: The reference of a Reference instruction in a ScheduleBlock.
        qubits: Optional argument to reorder the references.

    Returns:
        A string corresponding to the name of the referenced schedule and the qubits that
        this schedule applies to.

    Raises:
        CalibrationError: If ``reference`` is not a tuple.
        CalibrationError: If ``reference`` is not a tuple of reference name and the qubits that
            that the schedule applies to.
    """
    if not isinstance(reference, tuple):
        raise CalibrationError(f"A schedule reference must be a tuple. Found {reference}.")

    ref_schedule_name, ref_qubits = reference[0], reference[1:]

    if not isinstance(ref_schedule_name, str) and not isinstance(ref_qubits, tuple):
        raise CalibrationError(
            f"A schedule reference is a name and qubits tuple. Found {reference}"
        )

    ref_qubits = tuple(int(qubit[1:]) for qubit in ref_qubits)

    # get the qubit indices for which we are getting the schedules
    if qubits is not None and len(qubits) >= len(ref_qubits):
        ref_qubits = tuple(qubits[idx] for idx in ref_qubits)

    return ref_schedule_name, ref_qubits
