# This code is part of Qiskit.
#
# (C) Copyright IBM 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
"""
Layer Fidelity RB Experiment class.
"""
import functools
import logging
from collections import defaultdict
from typing import Union, Iterable, Optional, List, Sequence, Tuple

import numpy as np
from numpy.random import Generator, default_rng
from numpy.random.bit_generator import BitGenerator, SeedSequence

from qiskit.circuit import QuantumCircuit, CircuitInstruction, Barrier, Gate
from qiskit.circuit.library import get_standard_gate_name_mapping
from qiskit.exceptions import QiskitError
from qiskit.providers import BackendV2Converter
from qiskit.providers.backend import Backend, BackendV1, BackendV2
from qiskit.quantum_info import Clifford
from qiskit.pulse.instruction_schedule_map import CalibrationPublisher

from qiskit_experiments.framework import BaseExperiment, Options
from qiskit_experiments.framework.restless_mixin import RestlessMixin

from .clifford_utils import (
    CliffordUtils,
    DEFAULT_SYNTHESIS_METHOD,
    compose_1q,
    compose_2q,
    inverse_1q,
    inverse_2q,
    num_from_2q_circuit,
    _product_1q_nums,
    _clifford_1q_int_to_instruction,
    _clifford_2q_int_to_instruction,
    _decompose_clifford_ops,
)
from .layer_fidelity_analysis import LayerFidelityAnalysis

LOG = logging.getLogger(__name__)


NUM_1Q_CLIFFORD = CliffordUtils.NUM_CLIFFORD_1_QUBIT


class LayerFidelity(BaseExperiment, RestlessMixin):
    """TODO

    # section: overview

    TODO

    # section: analysis_ref
        :class:`LayerFidelityAnalysis`

    # section: reference
        .. ref_arxiv:: 1 2311.05933
    """

    def __init__(
        self,
        physical_qubits: Sequence[int],
        two_qubit_layers: Sequence[Sequence[Tuple[int, int]]],
        lengths: Iterable[int],
        backend: Optional[Backend] = None,
        num_samples: int = 6,
        seed: Optional[Union[int, SeedSequence, BitGenerator, Generator]] = None,
        two_qubit_gate: Optional[str] = None,
        one_qubit_basis_gates: Optional[Sequence[str]] = None,
        replicate_in_parallel: bool = False,  # TODO: remove this option
    ):
        """Initialize a standard randomized benchmarking experiment.

        Args:
            physical_qubits: List of physical qubits for the experiment.
            two_qubit_layers: List of pairs of qubits to run on, will use the direction given here.
            lengths: A list of RB sequences lengths.
            backend: The backend to run the experiment on.
            num_samples: Number of samples to generate for each sequence length.
            seed: Optional, seed used to initialize ``numpy.random.default_rng``.
                  when generating circuits. The ``default_rng`` will be initialized
                  with this seed value every time :meth:`circuits` is called.
            two_qubit_gate: Two-qubit gate name (e.g. "cx", "cz", "ecr")
                            of which the two qubit layers consist.
            one_qubit_basis_gates: One-qubit gates to use for implementing 1q Clifford operations.
            replicate_in_parallel: Use a common 1Q/2Q direct RB sequence
                                   for all quibit pairs in a layer or not.

        Raises:
            QiskitError: If any invalid argument is supplied.
        """
        # Compute full layers
        full_layers = []
        for two_q_layer in two_qubit_layers:
            qubits_in_layer = {q for qpair in two_q_layer for q in qpair}
            for q in qubits_in_layer:
                if q not in physical_qubits:
                    raise QiskitError(f"Qubit {q} in two_qubit_layers is not in physical_qubits")
            layer = two_q_layer + [(q,) for q in physical_qubits if q not in qubits_in_layer]
            full_layers.append(layer)

        # Initialize base experiment
        super().__init__(
            physical_qubits, analysis=LayerFidelityAnalysis(full_layers), backend=backend
        )
        # assert isinstance(backend, BackendV2)

        # Verify parameters
        if len(set(lengths)) != len(lengths):
            raise QiskitError(f"The lengths list {lengths} should not contain duplicate elements.")
        if num_samples <= 0:
            raise QiskitError(f"The number of samples {num_samples} should be positive.")

        if two_qubit_gate:
            if self.backend:
                if two_qubit_gate not in self.backend.target.operation_names:
                    raise QiskitError(f"two_qubit_gate {two_qubit_gate} is not in backend.target")
                for two_q_layer in two_qubit_layers:
                    for qpair in two_q_layer:
                        if not self.backend.target.instruction_supported(two_qubit_gate, qpair):
                            raise QiskitError(f"{two_qubit_gate}{qpair} is not in backend.target")
        else:
            if self.backend:
                # Try to set default two_qubit_gate from backend
                for op in self.backend.target.operations:
                    if isinstance(op, Gate) and op.num_qubits == 2:
                        two_qubit_gate = op.name
                        LOG.info("%s is set for two_qubit_gate", op.name)
                        break
            if not two_qubit_gate:
                raise QiskitError(f"two_qubit_gate is not provided and failed to set from backend.")

        if one_qubit_basis_gates:
            for gate in one_qubit_basis_gates:
                if gate not in self.backend.target.operation_names:
                    raise QiskitError(f"{gate} in one_qubit_basis_gates is not in backend.target")
            for gate in one_qubit_basis_gates:
                for q in self.physical_qubits:
                    if not self.backend.target.instruction_supported(gate, (q,)):
                        raise QiskitError(f"{gate}({q}) is not in backend.target")
        else:
            if self.backend:
                # Try to set default one_qubit_basis_gates from backend
                one_qubit_basis_gates = []
                for op in self.backend.target.operations:
                    if isinstance(op, Gate) and op.num_qubits == 1:
                        one_qubit_basis_gates.append(op.name)
                LOG.info("%s is set for one_qubit_basis_gates", str(one_qubit_basis_gates))
            if not one_qubit_basis_gates:
                raise QiskitError(
                    f"one_qubit_basis_gates is not provided and failed to set from backend."
                )

        # Set configurable options
        self.set_experiment_options(
            lengths=sorted(lengths),
            num_samples=num_samples,
            seed=seed,
            two_qubit_layers=two_qubit_layers,
            two_qubit_gate=two_qubit_gate,
            one_qubit_basis_gates=tuple(one_qubit_basis_gates),
            replicate_in_parallel=replicate_in_parallel,
        )

    @classmethod
    def _default_experiment_options(cls) -> Options:
        """Default experiment options.

        Experiment Options:
            two_qubit_layers (List[List[Tuple[int, int]]]): List of pairs of qubits to run on.
            lengths (List[int]): A list of RB sequences lengths.
            num_samples (int): Number of samples to generate for each sequence length.
            seed (None or int or SeedSequence or BitGenerator or Generator): A seed
                used to initialize ``numpy.random.default_rng`` when generating circuits.
                The ``default_rng`` will be initialized with this seed value every time
                :meth:`circuits` is called.
            two_qubit_gate (str): Two-qubit gate name (e.g. "cx", "cz", "ecr")
                of which the two qubit layers consist.
            one_qubit_basis_gates (Tuple[str]): One-qubit gates to use for implementing 1q Cliffords.
            replicate_in_parallel (bool): Use a common 1Q/2Q direct RB sequence
                for all quibit pairs in a layer or not.
            clifford_synthesis_method (str): The name of the Clifford synthesis plugin to use
                for building circuits of RB sequences.
        """
        options = super()._default_experiment_options()
        options.update_options(
            lengths=None,
            num_samples=None,
            seed=None,
            two_qubit_layers=None,
            two_qubit_gate=None,
            one_qubit_basis_gates=(),
            clifford_synthesis_method=DEFAULT_SYNTHESIS_METHOD,
            replicate_in_parallel=True,
        )
        return options

    def set_experiment_options(self, **fields):
        """Set the experiment options.

        Args:
            fields: The fields to update the options

        Raises:
            AttributeError: If the field passed in is not a supported options
        """
        for field in fields:
            if field in {"two_qubit_layers"}:
                if (
                    hasattr(self._experiment_options, field)
                    and self._experiment_options[field] is not None
                ):
                    raise AttributeError(f"Options field {field} is not allowed to update.")
        super().set_experiment_options(**fields)

    @classmethod
    def _default_transpile_options(cls) -> Options:
        """Default transpiler options for transpiling RB circuits."""
        return Options(optimization_level=1)

    def set_transpile_options(self, **fields):
        """Transpile options is not supported for LayerFidelity experiments.

        Raises:
            QiskitError: If `set_transpile_options` is called.
        """
        raise QiskitError(
            "Custom transpile options is not supported for LayerFidelity experiments."
        )

    def _set_backend(self, backend: Backend):
        """Set the backend V2 for RB experiments since RB experiments only support BackendV2.
        If BackendV1 is provided, it is converted to V2 and stored.
        """
        if isinstance(backend, BackendV1):
            super()._set_backend(BackendV2Converter(backend, add_delay=True))
        else:
            super()._set_backend(backend)

    def __residual_qubits(self, two_qubit_layer):
        qubits_in_layer = {q for qpair in two_qubit_layer for q in qpair}
        return [q for q in self.physical_qubits if q not in qubits_in_layer]

    def circuits(self) -> List[QuantumCircuit]:
        """Return a list of physical circuits to measure layer fidelity.

        Returns:
            A list of :class:`QuantumCircuit`.
        """
        return list(self.circuits_generator())

    def circuits_generator(self) -> Iterable[QuantumCircuit]:
        """Generate physical circuits to measure layer fidelity.

        Returns:
            A generator of :class:`QuantumCircuit`s.
        """
        opts = self.experiment_options
        residal_qubits_by_layer = [self.__residual_qubits(layer) for layer in opts.two_qubit_layers]
        rng = default_rng(seed=opts.seed)
        # define functions and variables for speed
        _to_gate_1q = functools.partial(
            _clifford_1q_int_to_instruction,
            basis_gates=opts.one_qubit_basis_gates,
            synthesis_method=opts.clifford_synthesis_method,
        )
        _to_gate_2q = functools.partial(
            _clifford_2q_int_to_instruction,
            basis_gates=(opts.two_qubit_gate,) + opts.one_qubit_basis_gates,
            coupling_tuple=((0, 1),),
            synthesis_method=opts.clifford_synthesis_method,
        )
        GATE2Q = self.backend.target.operation_from_name(opts.two_qubit_gate)
        GATE2Q_CLIFF = num_from_2q_circuit(Clifford(GATE2Q).to_circuit())
        # Circuit generation
        circuits = []
        num_qubits = max(self.physical_qubits) + 1
        for i_sample in range(opts.num_samples):
            for i_set, (two_qubit_layer, one_qubits) in enumerate(
                zip(opts.two_qubit_layers, residal_qubits_by_layer)
            ):
                num_2q_gates = len(two_qubit_layer)
                num_1q_gates = len(one_qubits)
                composite_qubits = two_qubit_layer + [(q,) for q in one_qubits]
                composite_clbits = [(2 * c, 2 * c + 1) for c in range(num_2q_gates)]
                composite_clbits.extend(
                    [(c,) for c in range(2 * num_2q_gates, 2 * num_2q_gates + num_1q_gates)]
                )
                for length in opts.lengths:
                    circ = QuantumCircuit(num_qubits, num_qubits)
                    BARRIER_INST = CircuitInstruction(Barrier(num_qubits), circ.qubits)
                    # add the main body of the circuit switching implementation for speed
                    if opts.replicate_in_parallel:
                        self.__circuit_body_with_replication(
                            circ,
                            length,
                            two_qubit_layer,
                            one_qubits,
                            rng,
                            _to_gate_1q,
                            _to_gate_2q,
                            GATE2Q,
                            GATE2Q_CLIFF,
                            BARRIER_INST,
                        )
                    else:
                        self.__circuit_body(
                            circ,
                            length,
                            two_qubit_layer,
                            one_qubits,
                            rng,
                            _to_gate_1q,
                            _to_gate_2q,
                            GATE2Q,
                            GATE2Q_CLIFF,
                            BARRIER_INST,
                        )
                    # add the measurements
                    circ._append(BARRIER_INST)
                    for qubits, clbits in zip(composite_qubits, composite_clbits):
                        circ.measure(qubits, clbits)
                    # store composite structure in metadata
                    circ.metadata = {
                        "experiment_type": "BatchExperiment",
                        "composite_metadata": [
                            {
                                "experiment_type": "ParallelExperiment",
                                "composite_index": list(range(len(composite_qubits))),
                                "composite_metadata": [
                                    {
                                        "experiment_type": "SubLayerFidelity",
                                        "physical_qubits": qpair,
                                        "sample": i_sample,
                                        "xval": length,
                                    }
                                    for qpair in two_qubit_layer
                                ]
                                + [
                                    {
                                        "experiment_type": "SubLayerFidelity",
                                        "physical_qubits": (q,),
                                        "sample": i_sample,
                                        "xval": length,
                                    }
                                    for q in one_qubits
                                ],
                                "composite_qubits": composite_qubits,
                                "composite_clbits": composite_clbits,
                            }
                        ],
                        "composite_index": [i_set],
                    }
                    yield circ

    @staticmethod
    def __circuit_body(
        circ,
        length,
        two_qubit_layer,
        one_qubits,
        rng,
        _to_gate_1q,
        _to_gate_2q,
        GATE2Q,
        GATE2Q_CLIFF,
        BARRIER_INST,
    ):
        # initialize cliffords and a ciruit (0: identity clifford)
        cliffs_2q = [0] * len(two_qubit_layer)
        cliffs_1q = [0] * len(one_qubits)
        for _ in range(length):
            # sample random 1q-Clifford layer
            for j, qpair in enumerate(two_qubit_layer):
                # sample product of two 1q-Cliffords as 2q interger Clifford
                samples = rng.integers(NUM_1Q_CLIFFORD, size=2)
                cliffs_2q[j] = compose_2q(cliffs_2q[j], _product_1q_nums(*samples))
                for sample, q in zip(samples, qpair):
                    circ._append(_to_gate_1q(sample), (circ.qubits[q],), ())
            for k, q in enumerate(one_qubits):
                sample = rng.integers(NUM_1Q_CLIFFORD)
                cliffs_1q[k] = compose_1q(cliffs_1q[k], sample)
                circ._append(_to_gate_1q(sample), (circ.qubits[q],), ())
            circ._append(BARRIER_INST)
            # add two qubit gates
            for j, qpair in enumerate(two_qubit_layer):
                circ._append(GATE2Q, tuple(circ.qubits[q] for q in qpair), ())
                cliffs_2q[j] = compose_2q(cliffs_2q[j], GATE2Q_CLIFF)
                # TODO: add dd if necessary
            for k, q in enumerate(one_qubits):
                # TODO: add dd if necessary
                pass
            circ._append(BARRIER_INST)
        # add the last inverse
        for j, qpair in enumerate(two_qubit_layer):
            inv = inverse_2q(cliffs_2q[j])
            circ._append(_to_gate_2q(inv), tuple(circ.qubits[q] for q in qpair), ())
        for k, q in enumerate(one_qubits):
            inv = inverse_1q(cliffs_1q[k])
            circ._append(_to_gate_1q(inv), (circ.qubits[q],), ())
        return circ

    @staticmethod
    def __circuit_body_with_replication(
        circ,
        length,
        two_qubit_layer,
        one_qubits,
        rng,
        _to_gate_1q,
        _to_gate_2q,
        GATE2Q,
        GATE2Q_CLIFF,
        BARRIER_INST,
    ):
        # initialize cliffords and a ciruit (0: identity clifford)
        cliff_2q = 0
        cliff_1q = 0
        for _ in range(length):
            # sample random 1q-Clifford layer
            samples = rng.integers(NUM_1Q_CLIFFORD, size=2)
            cliff_2q = compose_2q(cliff_2q, _product_1q_nums(*samples))
            for qpair in two_qubit_layer:
                for sample, q in zip(samples, qpair):
                    circ._append(_to_gate_1q(sample), (circ.qubits[q],), ())
            if one_qubits:
                sample = rng.integers(NUM_1Q_CLIFFORD)
                cliff_1q = compose_1q(cliff_1q, sample)
                for q in one_qubits:
                    circ._append(_to_gate_1q(sample), (circ.qubits[q],), ())
            circ._append(BARRIER_INST)
            # add two qubit gates
            cliff_2q = compose_2q(cliff_2q, GATE2Q_CLIFF)
            for qpair in two_qubit_layer:
                circ._append(GATE2Q, tuple(circ.qubits[q] for q in qpair), ())
                # TODO: add dd if necessary
            for q in one_qubits:
                # TODO: add dd if necessary
                pass
            circ._append(BARRIER_INST)
        # add the last inverse
        inv = inverse_2q(cliff_2q)
        for qpair in two_qubit_layer:
            circ._append(_to_gate_2q(inv), tuple(circ.qubits[q] for q in qpair), ())
        inv = inverse_1q(cliff_1q)
        for q in one_qubits:
            circ._append(_to_gate_1q(inv), (circ.qubits[q],), ())
        return circ

    def _transpiled_circuits(self) -> List[QuantumCircuit]:
        """Return a list of experiment circuits, transpiled."""
        transpiled = [_decompose_clifford_ops(circ) for circ in self.circuits()]
        # Set custom calibrations provided in backend
        if isinstance(self.backend, BackendV2):
            instructions = []  # (op_name, qargs) for each element where qargs means qubit tuple
            for two_qubit_layer in self.experiment_options.two_qubit_layers:
                for qpair in two_qubit_layer:
                    instructions.append((self.experiment_options.two_qubit_gate, tuple(qpair)))
                for q in self.__residual_qubits(two_qubit_layer):
                    for gate_1q in self.experiment_options.one_qubit_basis_gates:
                        instructions.append((gate_1q, (q,)))

            common_calibrations = defaultdict(dict)
            for op_name, qargs in instructions:
                inst_prop = self.backend.target[op_name].get(qargs, None)
                if inst_prop is None:
                    continue
                try:
                    schedule = inst_prop.calibration
                except:  # TODO remove after qiskit #11397
                    continue
                if schedule is None:
                    continue
                publisher = schedule.metadata.get("publisher", CalibrationPublisher.QISKIT)
                if publisher != CalibrationPublisher.BACKEND_PROVIDER:
                    common_calibrations[op_name][(qargs, tuple())] = schedule

            for circ in transpiled:
                circ.calibrations = common_calibrations

        return transpiled

    def _metadata(self):
        metadata = super()._metadata()
        metadata["two_qubit_layers"] = self.experiment_options.two_qubit_layers
        return metadata
