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

"""Transpiler pass for calibration experiments."""

from typing import Dict, List, Tuple, Union

from qiskit import QuantumCircuit
from qiskit.dagcircuit import DAGCircuit
from qiskit.pulse.schedule import ScheduleBlock
from qiskit.transpiler.basepasses import TransformationPass
from qiskit.transpiler.passmanager import PassManager

from qiskit_experiments.calibration_management.calibrations import Calibrations
from qiskit_experiments.exceptions import CalibrationError
from qiskit_experiments.framework.base_experiment import BaseExperiment


class CalibrationsAdder(TransformationPass):
    """A transformation pass to add calibrations for standard circuit instructions.

    This transpiler pass is intended to be run in the :meth:`circuits` method of the
    experiment classes before the main transpiler pass. It's only goal is to extract
    the needed pulse schedules from an instance of Calibrations and attach them to the
    template circuit. This has a couple of challenges. Note that if no mapping is provided
    this transpiler pass assumes that the name of the schedule in the calibrations is the
    same as the name of the gate instruction.
    """

    def __init__(
            self,
            calibrations: Calibrations,
            instructions_map: Dict[str, str],
            qubit_layout: Dict[int, int],
    ):
        """Initialize the pass.

        Args:
            calibrations: An instance of calibration from which to fetch the schedules.
            instructions_map: A map of circuit instruction names (keys) to schedule names stored
                in the calibrations (values). If an entry is not found the pass will assume that
                the instruction in the circuit and the schedule have the same name.
            qubit_layout: The initial layout that will be used.
        """
        super().__init__()
        self._qubit_layout = qubit_layout
        self._cals = calibrations
        self._instructions_map = instructions_map

    def _get_calibration(self, gate: str, qubits: Tuple[int, ...]) -> Union[ScheduleBlock, None]:
        """Get a schedule from the internally stored calibrations.

        Args:
            gate: Name of the gate for which to get the schedule.
            qubits: The qubits for which to get the parameters.

        Returns:
            The schedule if one is found otherwise return None.
        """

        # Extract the gate to schedule and any parameter name mappings.
        sched_name = self._instructions_map.get(gate, gate)

        # Try and get a schedule, if there is none then return None.
        try:
            return self._cals.get_schedule(sched_name, qubits)
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

                # Get the qubit indices in the circuit.
                qubits = tuple(bit_indices[qarg] for qarg in node.qargs)

                # Get the physical qubits that they will remap to.
                qubits = tuple(self._qubit_layout[qubit] for qubit in qubits)

                schedule = self._get_calibration(node.op.name, qubits)

                # Permissive stance: if we don't find a schedule we continue.
                # The call to the transpiler that happens before running the
                # experiment will either complain or force us to use the
                # backend gates.
                if schedule is None:
                    continue

                if len(set(qubits) & set(ch.index for ch in schedule.channels)) == 0:
                    raise CalibrationError(
                        f"None of the qubits {qubits} are contained in the channels of "
                        f"the schedule named {schedule.name} for gate {node.op}."
                    )

                dag.add_calibration(node.op, qubits, schedule)

        return dag


def inject_calibrations(
    circuits: Union[QuantumCircuit, List[QuantumCircuit]],
    experiment: BaseExperiment
) -> Union[QuantumCircuit, List[QuantumCircuit]]:
    """Inject calibrations from a :class:`Calibrations` instance into a circuit.

    This function only adds calibrations if it can find them in the calibrations.

    Args:
        circuits: The circuit or list of circuits into which to inject calibrations.
        experiment: The experiment object that

    Returns:
         A quantum circuit with the relevant calibrations added to it.
    """
    layout = {idx: qubit for idx, qubit in enumerate(experiment.physical_qubits)}

    # Identify the available schedule data.
    calibrations = experiment.experiment_options.get("calibrations", None)

    # Run the transpiler pass according to the available data
    if calibrations is not None:
        inst_maps = experiment.experiment_options.get("instruction_name_map", None) or dict()
        return PassManager(CalibrationsAdder(calibrations, inst_maps, layout)).run(circuits)

    return circuits
