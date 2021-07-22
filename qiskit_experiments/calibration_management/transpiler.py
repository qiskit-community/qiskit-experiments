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

from qiskit import QuantumCircuit
from qiskit.dagcircuit import DAGCircuit
from qiskit.pulse.schedule import ScheduleBlock
from qiskit.transpiler.basepasses import TransformationPass
from qiskit.transpiler.passmanager import PassManager

from qiskit_experiments.calibration.management.calibrations import Calibrations
from qiskit_experiments.exceptions import CalibrationError
from qiskit_experiments.calibration_management.calibration_key_types import InstructionMap
from qiskit_experiments.framework.base_experiment import BaseExperiment


class CalAdder(TransformationPass):
    """Transformation pass to inject calibrations into circuits."""

    def __init__(
            self,
            calibrations: Calibrations,
            instruction_maps: Optional[List[InstructionMap]] = None,
            qubit_layout: Optional[Dict[int, int]] = None
    ):
        """

        This transpiler pass is intended to be run in the :meth:`circuits` method of the
        experiment classes before the main transpiler pass.

        Args:
            calibrations: An instance of calibration from which to fetch the schedules.
            instruction_maps: A list of instruction maps to map gate names in the circuit to
                schedule names in the calibrations. If this is not provided the transpiler pass
                will assume that the schedule has the same name as the gate. Each instruction map
                may also specify parameters that should be left free in the schedule.
            qubit_layout: The initial layout that will be used. This remaps the qubits
                in the added calibrations. For instance, if {0: 3} is given and use this pass
                on a circuit then any gates on qubit 0 will add calibrations for qubit 3.
        """
        super().__init__()
        self._cals = calibrations

        self._instruction_maps = dict()
        for inst_map in instruction_maps:
            self._intruction_maps[inst_map.inst] = inst_map

        self._qubit_layout = qubit_layout

    def get_calibration(
            self,
            gate_name: str,
            qubits: Tuple[int, ...],
    ) -> Union[ScheduleBlock, None]:
        """Gets the calibrated schedule

        Args:
            gate_name: Name of the gate for which to get the schedule.
            qubits: The qubits for which to get the parameters.

        Returns:
            The schedule if one is found otherwise return None.
        """
        name = gate_name
        assign_params = None

        # check for a non-trivial instruction to schedule mapping.
        if gate_name in self._instruction_maps:
            inst_map = self._instruction_maps[gate_name]
            name = inst_map.schedule
            assign_params = {param.name: param for param in inst_map.free_params}

        try:
            return self._cals.get_schedule(name, qubits, assign_params=assign_params)
        except CalibrationError:
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

                if self._qubit_layout is not None:
                    try:
                        qubits = tuple(self._qubit_layout[qubit] for qubit in qubits)
                    except KeyError:
                        pass

                schedule = self.get_calibration(node.op.name, qubits)

                if schedule is not None:
                    dag.add_calibration(node.op, qubits, schedule, params=params)

        return dag


def inject_calibrations(circuit: QuantumCircuit, experiment: BaseExperiment) -> QuantumCircuit:
    """Inject calibrations from a :class:`Calibrations` instance into a circuit.

    This function requires that the experiment has a list of InstructionMaps in its
    experiment options.

    Args:
        circuit: The circuit into which to inject calibrations.
        experiment: The experiment object that

    Returns:
         A quantum circuit with the relevant calibrations inject into it.
    """
    calibrations = experiment.experiment_options.calibrations
    inst_maps = experiment.experiment_options.instruction_name_maps
    layout = {idx: qubit  for idx, qubit in enumerate(experiment.physical_qubits)}

    return PassManager(CalAdder(calibrations, inst_maps, layout)).run(circuit)
