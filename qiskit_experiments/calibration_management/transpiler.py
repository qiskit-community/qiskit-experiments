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

from abc import abstractmethod
from typing import Dict, List, Optional, Tuple, Union

from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.dagcircuit import DAGCircuit
from qiskit.pulse.schedule import ScheduleBlock
from qiskit.transpiler.basepasses import TransformationPass
from qiskit.transpiler.passmanager import PassManager

from qiskit_experiments.calibration_management.calibrations import Calibrations
from qiskit_experiments.exceptions import CalibrationError
from qiskit_experiments.framework.base_experiment import BaseExperiment


class CalibrationsMap:

    def __init__(self):
        """"""
        self._map = dict()

    def add(self, gate_name: str, schedule_name: str, parameter_map: Dict[str, str]):
        """
        Args:
            gate_name:
            schedule_name:
            parameter_map:
        """
        self._map[gate_name] = (schedule_name, parameter_map)

    def get(self, gate_name: str) -> Tuple[str, Dict]:
        """"""
        if gate_name in self._map:
            return self._map[gate_name]

        # Return the trivial map
        return gate_name, {}


class BaseCalibrationAdder(TransformationPass):
    """Transformation pass to inject calibrations into circuits of calibration experiments."""

    def __init__(self, qubit_layout: Optional[Dict[int, int]] = None):
        """Initialize the pass.

        Args:
            qubit_layout: The initial layout to be used in the transpilation called before
                running the experiment. This initial layout is needed here since the qubits
                in the circuits on which this transpiler pass is run will always be [0, 1, ...].
                If the initial layout is, for example, {0: 3} then any gates on qubit 0 will
                add calibrations for qubit 3.
        """
        super().__init__()
        self._qubit_layout = qubit_layout

    @abstractmethod
    def _get_calibration(
        self, gate: str, qubits: Tuple[int, ...], params: List[Parameter]
    ) -> Union[ScheduleBlock, None]:
        """Get a schedule from the internally stored schedules."""

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

                # Get the qubit indices in the circuit.
                qubits = tuple(bit_indices[qarg] for qarg in node.qargs)

                # Get the physical qubits that they will remap to.
                qubits = tuple(self._qubit_layout[qubit] for qubit in qubits)

                schedule = self._get_calibration(node.op.name, qubits, params)

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

                # Check the consistency between the circuit and schedule parameters
                for param in params:
                    if isinstance(param, Parameter) and param not in schedule.parameters:
                        raise CalibrationError(
                            f"Gate {node.op.name} has parameter {param} that is not contained "
                            f"in schedule {schedule}."
                        )

                dag.add_calibration(node.op, qubits, schedule, params=params)

        return dag


class CalibrationsAdder(BaseCalibrationAdder):
    """This BaseCalibrationAdder stores schedules in the calibrations.

    This transpiler pass is intended to be run in the :meth:`circuits` method of the
    experiment classes before the main transpiler pass. It's only goal is to extract
    the needed pulse schedules from an instance of Calibrations and attach them to the
    template circuit. This has a couple of challenges.

    * First, the same pulse in the calibrations can be attached to different gates.
      For example, an X-gate "x" may need to be attached to a Rabi gate in a
      :class:`Rabi` experiment while in an :class:`EFSpectroscopy` experiment it will
      be attached to the X-gate.

    * Second, the gate may sometimes be attached with parameters and sometimes not.
      In a Rabi experiment the "x" schedule will have a parametric amplitude while in
      the :class:`FineXAmplitude` the gate will not have any free parameters.

    These two issues are solved by adding an InstructionMap which is a named tuple of
    instruction name in the circuit, the schedule name in the calibrations and any
    parameter instance that needs to be unassigned when getting the schedule from the
    :class:`Calibrations` instance. Consider the following examples.

    .. code-block::python

        # Instruction mapping for a Rabi experiment
        InstructionMap("Rabi", "x", [Parameter("amp")])

        # Instruction mapping for a Drag experiment
        beta = Parameter("β")
        InstructionMap("Rp", "x", [beta])
        InstructionMap("Rm", "xm", [beta])

    Note that if no mapping is provided this transpiler pass assumes that the name of
    the schedule in the calibrations is the same as the name of the gate instruction.
    """

    def __init__(
            self,
            calibrations: Calibrations,
            calibrations_map: Optional[CalibrationsMap] = None,
            qubit_layout: Optional[Dict[int, int]] = None,
    ):
        """Initialize the pass.

        Args:
            calibrations: An instance of calibration from which to fetch the schedules.
            calibrations_map: A list of instruction maps to map gate names in the circuit to
                schedule names in the calibrations. If this is not provided the transpiler pass
                will assume that the schedule has the same name as the gate. Each instruction map
                may also specify parameters that should be left free in the schedule.
            qubit_layout: The initial layout that will be used.
        """
        super().__init__(qubit_layout)
        self._cals = calibrations
        self._calibrations_map = calibrations_map or dict()

    def _get_calibration(
        self, gate: str,
        qubits: Tuple[int, ...],
        node_params: List[Parameter]
    ) -> Union[ScheduleBlock, None]:
        """Get a schedule from the internally stored calibrations.

        Args:
            gate: Name of the gate for which to get the schedule.
            qubits: The qubits for which to get the parameters.

        Returns:
            The schedule if one is found otherwise return None.
        """

        # Extract the gate to schedule and any parameter name mappings.
        sched_name, params_map = self._calibrations_map.get(gate)

        assign_params = {}
        for param in node_params:
            if isinstance(param, Parameter):
                assign_params[params_map.get(param.name, param.name)] = param

        # Try and get a schedule, if there is none then return None.
        try:
            return self._cals.get_schedule(sched_name, qubits, assign_params=assign_params)
        except CalibrationError:
            return None


class ScheduleAdder(BaseCalibrationAdder):
    """A naive calibrations adder for lists of schedules.

    This calibration adder is used for cases when users provide schedules for the
    experiment as a dict of schedules.
    """

    def __init__(
        self,
        schedules: Dict[Tuple[str, Tuple[int, ...]], ScheduleBlock],
        qubit_layout: Optional[Dict[int, int]] = None,
    ):
        """Initialize the :class:`ScheduleAdder` from a dict of schedules.

        Args:
            schedules: The schedules are provided as a dict. Here, the keys correspond to a
                tuple of the name of the instruction in the quantum circuits and the qubits
                while the values are the schedules that will be added to the calibrations of
                the QuantumCircuit.
            qubit_layout: The initial layout that will be used.
        """
        super().__init__(qubit_layout)
        self._schedules = schedules

    def _get_calibration(
        self,
        gate: str,
        qubits: Tuple[int, ...],
        node_params: List[Parameter]
    ) -> Union[ScheduleBlock, None]:
        """Get a schedule from the internally stored schedules.

        Args:
            gate: Name of the gate for which to get the schedule.
            qubits: The qubits for which to get the parameters.

        Returns:
            The schedule if one is found otherwise return None.
        """
        return self._schedules.get((gate, qubits), None)


def inject_calibrations(
    circuits: Union[QuantumCircuit, List[QuantumCircuit]],
    experiment: BaseExperiment
) -> Union[QuantumCircuit, List[QuantumCircuit]]:
    """Inject calibrations from a :class:`Calibrations` instance into a circuit.

    This function requires that the experiment has a list of InstructionMaps in its
    experiment options as well as the calibrations.

    Args:
        circuits: The circuit or list of circuits into which to inject calibrations.
        experiment: The experiment object that

    Returns:
         A quantum circuit with the relevant calibrations inject into it.
    """
    layout = {idx: qubit for idx, qubit in enumerate(experiment.physical_qubits)}

    calibrations = experiment.experiment_options.get("calibrations", None)

    if calibrations is None:
        user_schedule_config = experiment.experiment_options.get("schedules_config", None)

        if user_schedule_config is None:
            return circuits

        return PassManager(ScheduleAdder(user_schedule_config, layout)).run(circuits)

    else:
        inst_maps = experiment.experiment_options.instruction_name_maps

        return PassManager(CalibrationsAdder(calibrations, inst_maps, layout)).run(circuits)
