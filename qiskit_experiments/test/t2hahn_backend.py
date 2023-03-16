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
"""
T2HahnBackend class.
Temporary backend to be used for t2hahn experiment
"""
import copy
from typing import Sequence, Union

import numpy as np

from qiskit import QiskitError, QuantumCircuit
from qiskit.circuit import Delay, Reset, Parameter
from qiskit.circuit.library import Measure, RZGate, SXGate, XGate
from qiskit.dagcircuit import DAGCircuit
from qiskit.providers import BackendV2, Job, Options
from qiskit.transpiler import InstructionProperties, PassManager, Target, TransformationPass
from qiskit.utils.units import apply_prefix

from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, ReadoutError, RelaxationNoisePass, reset_error

from qiskit_experiments.test.utils import FakeJob


# These pylint directives should be removed when the version of pylint is
# updated from 2.7.1
# pylint: disable=isinstance-second-argument-not-valid-type
# pylint: disable=arguments-differ


class ResetQubits(TransformationPass):
    """Pass to inject reset instructions for each qubit

    The resets are used to add qubit initialization error.
    """

    def run(self, dag: DAGCircuit):
        new_dag = copy.deepcopy(dag)

        for qreg in new_dag.qregs.values():
            new_dag.apply_operation_front(Reset(), qreg, [])

        return new_dag


class QubitDrift(TransformationPass):
    """Pass to rotate qubits during delays

    This pass adds rotations that mimic the qubit being detuned from the drive
    frequency (while assuming that the gate times are negligible; rotations are
    only added for delays).
    """

    def __init__(self, qubit_frequencies: Sequence[float], dt: float):
        super().__init__()
        self.qubit_frequencies = qubit_frequencies
        self.dt = dt

    def run(self, dag: DAGCircuit):
        qubit_indices = {bit: index for index, bit in enumerate(dag.qubits)}

        new_dag = dag.copy_empty_like()

        for node in dag.topological_op_nodes():
            new_dag.apply_operation_back(node.op, node.qargs, node.cargs)

            if node.name == "delay":
                q0 = qubit_indices[node.qargs[0]]
                if self.qubit_frequencies[q0] is None:
                    continue
                if node.op.unit == "dt":
                    duration = node.op.duration * self.dt
                elif node.op.unit != "s":
                    duration = apply_prefix(node.op.duration, node.op.unit)
                angle = 2 * np.pi * self.qubit_frequencies[q0] * duration
                angle = angle % (2 * np.pi)
                new_dag.apply_operation_back(RZGate(angle), [node.qargs[0]], [])

        return new_dag


class T2HahnBackend(BackendV2):
    """
    A simple and primitive backend, to be run by the T2Hahn tests
    """

    def __init__(
        self,
        t2hahn: Union[None, float, Sequence[Union[float, None]]] = None,
        frequency: Union[None, float, Sequence[Union[float, None]]] = None,
        initialization_error: Union[None, float, Sequence[Union[float, None]]] = None,
        readout0to1: Union[None, float, Sequence[Union[float, None]]] = None,
        readout1to0: Union[None, float, Sequence[Union[float, None]]] = None,
        seed: int = 9000,
        dt: float = 1 / 4.5e9,
        num_qubits: Union[None, int] = None,
    ):
        """
        Initialize the T2Hahn backend
        """

        super().__init__(
            name="T2Hahn_simulator",
            backend_version="0",
        )

        if num_qubits is None:
            for arg in (t2hahn, frequency, initialization_error, readout0to1, readout1to0):
                if isinstance(arg, Sequence):
                    num_qubits = len(arg)
                    break
            else:
                num_qubits = 1

        self._t2hahn = (
            t2hahn if isinstance(t2hahn, (Sequence, type(None))) else [t2hahn] * num_qubits
        )
        self._frequency = (
            frequency if isinstance(frequency, (Sequence, type(None))) else [frequency] * num_qubits
        )
        self._initialization_error = (
            initialization_error
            if isinstance(initialization_error, (Sequence, type(None)))
            else [initialization_error] * num_qubits
        )
        self._readout0to1 = (
            readout0to1
            if isinstance(readout0to1, (Sequence, type(None)))
            else [readout0to1] * num_qubits
        )
        self._readout1to0 = (
            readout1to0
            if isinstance(readout1to0, (Sequence, type(None)))
            else [readout1to0] * num_qubits
        )
        self._seed = seed

        self._target = Target(dt=dt, num_qubits=num_qubits)
        for instruction in (Measure(), Reset(), RZGate(Parameter("angle")), SXGate(), XGate()):
            self.target.add_instruction(
                instruction,
                properties={(q,): InstructionProperties(duration=0) for q in range(num_qubits)},
            )
        self.target.add_instruction(Delay(Parameter("duration")))

    @property
    def target(self) -> Target:
        return self._target

    @property
    def max_circuits(self) -> None:
        return None

    @classmethod
    def _default_options(cls) -> Options:
        return Options()

    def run(self, circuits, shots: int = 1024, **options) -> Job:

        passes = []

        if isinstance(circuits, QuantumCircuit):
            circuits = [circuits]

        for circuit in circuits:
            if circuit.num_qubits > self.num_qubits:
                raise QiskitError(
                    f"{self.__class__} can only run circuits that match its num_qubits"
                )

        noise_model = NoiseModel()

        if self._initialization_error is not None:
            passes.append(ResetQubits())
            for qubit, error in enumerate(self._initialization_error):
                if error is None:
                    continue
                noise_model.add_quantum_error(reset_error(1 - error, error), ["reset"], [qubit])

        readout0to1 = self._readout0to1
        readout1to0 = self._readout1to0
        if readout0to1 and not readout1to0:
            readout1to0 = [0] * self.num_qubits
        elif readout1to0 and not readout0to1:
            readout0to1 = [0] * self.num_qubits
        if readout0to1:
            for qubit, (err0to1, err1to0) in enumerate(zip(readout0to1, readout1to0)):
                if err0to1 is None:
                    err0to1 = 0
                if err1to0 is None:
                    err1to0 = 0
                error = ReadoutError([[1 - err0to1, err0to1], [err1to0, 1 - err1to0]])
                noise_model.add_readout_error(error, [qubit])
        if self._t2hahn:
            t2s = [t if t is not None else 1 for t in self._t2hahn]
            # Make T1 huge so only T2 matters
            passes.append(RelaxationNoisePass([t * 100 for t in t2s], t2s, self.dt, Delay))
        if self._frequency:
            passes.append(QubitDrift(self._frequency, self.dt))

        pm = PassManager(passes)
        new_circuits = pm.run(circuits)

        sim = AerSimulator(noise_model=noise_model, seed_simulator=self._seed)

        job = sim.run(new_circuits, shots=shots)

        return FakeJob(self, job.result())
