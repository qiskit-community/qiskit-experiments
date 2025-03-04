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
ZZRamseyTestBackend class.

Backend for testing the ZZRamsey experiment
"""
import copy
from typing import List, Optional, Sequence, Union

import numpy as np

from qiskit import QiskitError, QuantumCircuit
from qiskit.circuit import Delay, Reset, Parameter
from qiskit.circuit.library import Measure, RZGate, RZZGate, SXGate, XGate
from qiskit.dagcircuit import DAGCircuit
from qiskit.providers import BackendV2, Job, Options
from qiskit.transpiler import InstructionProperties, PassManager, Target, TransformationPass
from qiskit.utils.units import apply_prefix

from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, ReadoutError, RelaxationNoisePass, reset_error

from qiskit_experiments.test.utils import FakeJob


class ResetQubits(TransformationPass):
    """Pass to inject reset instructions for each qubit

    The resets are used to add qubit initialization error.
    """

    def run(self, dag: DAGCircuit):
        new_dag = copy.deepcopy(dag)

        for qreg in new_dag.qregs.values():
            new_dag.apply_operation_front(Reset(), qreg, [])

        return new_dag


class ZZDrift(TransformationPass):
    """Pass to ZZ rotate qubits during delays

    This pass adds rotations that mimic the qubits undergoing ZZ rotations
    during delays in the circuit.

    .. note::

        This pass only adds ZZ for qubits [0, 1]. Also, it just adds an RZZ
        after each delay on 0 or 1. If there are simultaneous delays, it will
        double rotate.
    """

    def __init__(self, zz_frequency: float, dt: float):
        super().__init__()
        self.zz_frequency = zz_frequency
        self.dt = dt

    def run(self, dag: DAGCircuit):
        if len(dag.qubits) < 2:
            return dag

        zz_qubits = dag.qubits[:2]

        new_dag = dag.copy_empty_like()

        for node in dag.topological_op_nodes():
            new_dag.apply_operation_back(node.op, node.qargs, node.cargs)

            if node.name == "delay" and node.qargs[0] in zz_qubits:
                if node.op.unit == "dt":
                    duration = node.op.duration * self.dt
                else:
                    duration = apply_prefix(node.op.duration, node.op.unit)
                angle = 2 * np.pi * self.zz_frequency / 2 * duration
                angle = angle % (2 * np.pi)
                new_dag.apply_operation_back(RZZGate(angle), zz_qubits, [])

        return new_dag


class ZZRamseyTestBackend(BackendV2):
    """
    A simple and primitive backend, to be run by the ZZRamsey tests

    .. note::

        See the note on ZZDrift. This class is only intended for testing
        ZZRamsey and only (kind of) works for qubits [0, 1].
    """

    def __init__(
        self,
        t2hahn: Union[float, Sequence[float]] = float("inf"),
        zz_frequency: float = 0.0,
        initialization_error: Union[float, Sequence[float]] = 0.0,
        readout0to1: Union[float, Sequence[float]] = 0.0,
        readout1to0: Union[float, Sequence[float]] = 0.0,
        seed: int = 9000,
        dt: float = 1 / 4.5e9,
        num_qubits: Optional[int] = None,
    ):
        """
        Initialize the T2Hahn backend
        """
        super().__init__(
            name="ZZRamsey_simulator",
            backend_version="0",
        )

        for arg in (t2hahn, initialization_error, readout0to1, readout1to0):
            if isinstance(arg, Sequence):
                if num_qubits is None:
                    num_qubits = len(arg)
                elif len(arg) != num_qubits:
                    raise ValueError(
                        f"Input lengths are not consistent: {num_qubits} != {len(arg)}"
                    )

        if num_qubits is None:
            num_qubits = 2

        self._t2hahn = t2hahn if isinstance(t2hahn, Sequence) else [t2hahn] * num_qubits
        self._zz_frequency = zz_frequency
        self._initialization_error = (
            initialization_error
            if isinstance(initialization_error, Sequence)
            else [initialization_error] * num_qubits
        )
        self._readout0to1 = (
            readout0to1 if isinstance(readout0to1, Sequence) else [readout0to1] * num_qubits
        )
        self._readout1to0 = (
            readout1to0 if isinstance(readout1to0, Sequence) else [readout1to0] * num_qubits
        )
        self._seed = seed

        self._target = Target(dt=dt, num_qubits=num_qubits)
        for instruction in (Measure(), Reset(), RZGate(Parameter("angle")), SXGate(), XGate()):
            self.target.add_instruction(
                instruction,
                properties={(q,): InstructionProperties(duration=0) for q in range(num_qubits)},
            )
        self.target.add_instruction(
            RZZGate(Parameter("angle")),
            properties={(0, 1): InstructionProperties(duration=0)},
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

    def run(
        self, run_input: Union[QuantumCircuit, List[QuantumCircuit]], shots: int = 1024, **options
    ) -> Job:
        passes = []

        if isinstance(run_input, QuantumCircuit):
            circuits = [run_input]
        else:
            circuits = run_input

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

        if any(self._readout0to1) or any(self._readout1to0):
            for qubit, (err0to1, err1to0) in enumerate(zip(self._readout0to1, self._readout1to0)):
                error = ReadoutError([[1 - err0to1, err0to1], [err1to0, 1 - err1to0]])
                noise_model.add_readout_error(error, [qubit])
        if any(t2 != float("inf") for t2 in self._t2hahn):
            # Make T1 huge so only T2 matters
            passes.append(
                RelaxationNoisePass([t * 100 for t in self._t2hahn], self._t2hahn, self.dt, Delay)
            )
        if self._zz_frequency:
            passes.append(ZZDrift(self._zz_frequency, self.dt))

        pm = PassManager(passes)
        new_circuits = pm.run(circuits)

        sim = AerSimulator(noise_model=noise_model, seed_simulator=self._seed)

        job = sim.run(new_circuits, shots=shots, **options)

        return FakeJob(self, job.result())
