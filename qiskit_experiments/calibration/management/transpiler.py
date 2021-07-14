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

"""Transpiler functionality for calibrations."""

from typing import Dict, List, Optional, Tuple, Union

from qiskit.transpiler.passes import SetLayout
from qiskit.transpiler.passes import (
    TrivialLayout,
    FullAncillaAllocation,
    EnlargeWithAncilla,
    ApplyLayout
)
from qiskit.transpiler.layout import Layout
from qiskit.transpiler.passmanager import PassManager
from qiskit.transpiler.coupling import CouplingMap
from qiskit.circuit.quantumregister import QuantumRegister
from qiskit.dagcircuit import DAGCircuit
from qiskit.pulse.schedule import ScheduleBlock
from qiskit.transpiler.basepasses import TransformationPass

from qiskit_experiments.calibration.management.calibrations import Calibrations

# TODO: need to show how this might work in an experiment. E.g. Rabi?

class CalibrationAdder(TransformationPass):
    """Transformation pass to inject calibrations into circuits."""

    def __init__(
            self,
            calibrations: Calibrations,
            gate_schedule_map: Optional[Dict[str, str]] = None,
    ):
        """
        TODO Discuss: we could give calibrations as Dict[str, Dict[Tuple, ScheduleBlock]]
        TODO but this means that we need to export all the calibrations which may not scale well
        TODO using cas.get_schedule(name, qubits) on an as needed basis seems better.

        Args:
            calibrations:
            gate_schedule_map:
        """
        super().__init__()
        self._cals = calibrations
        self._gate_schedule_map = gate_schedule_map or dict()

    def get_calibration(
            self,
            gate_name: str,
            qubits: Tuple[int, ...]
    ) -> Union[ScheduleBlock, None]:
        """Gets the calibrated schedule

        Args:
            gate_name: Name of the gate for which to get the schedule.
            params: The parameters of the gate if any.
            qubits: The qubits for which to get the parameters.

        Returns:
            The schedule if one is found otherwise return None.
        """
        name = self._gate_schedule_map.get(gate_name, gate_name)
        try:
            return self._cals.get_schedule(name, qubits)
        except KeyError:
            return None

    def run(self, dag: DAGCircuit) -> DAGCircuit:
        """Run the calibration adder pass on `dag`.

        Args:
            dag: DAG to schedule.

        Returns:
            A DAG with calibrations added to it.
        """
        bit_indices = {bit: index for index, bit in enumerate(dag.qubits)}

        for node in dag.nodes():
            if node.type == "op":
                params = node.op.params
                qubits = tuple(bit_indices[qarg] for qarg in node.qargs)

                schedule = self.get_calibration(node.op.name, qubits)

                if schedule is not None:
                    dag.add_calibration(node.op, qubits, schedule, params=params)

        return dag


def get_calibration_pass_manager(
    initial_layout: List[int],
    coupling_map: List[List[int]],
    calibrations: Optional[Calibrations],
    gate_schedule_map: Optional[Dict[str, str]],
) -> PassManager:
    """Get a calibrations experiment pass manager.

    Args:
        initial_layout:
        coupling_map:
        calibrations:
        gate_schedule_map:

    Returns:
         An instance of :class:`PassManager` tailored to calibration experiments.
    """
    initial_layout = Layout.from_intlist(initial_layout, QuantumRegister(len(initial_layout), "q"))
    coupling_map = CouplingMap(coupling_map)

    def _choose_layout_condition(property_set):
        return not property_set["layout"]

    pm = PassManager()
    pm.append(SetLayout(initial_layout))
    pm.append(TrivialLayout(coupling_map), condition=_choose_layout_condition)
    pm.append([FullAncillaAllocation(coupling_map), EnlargeWithAncilla(), ApplyLayout()])

    if calibrations is not None:
        pm.append(CalibrationAdder(calibrations, gate_schedule_map))

    return pm
