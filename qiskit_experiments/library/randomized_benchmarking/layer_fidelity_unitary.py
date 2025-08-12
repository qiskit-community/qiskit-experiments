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
Layer Fidelity Unitary RB Experiment class.
"""
import functools
import logging
import warnings
from typing import Union, Iterable, Optional, List, Sequence, Tuple, Dict

import numpy as np
from numpy.random import Generator, default_rng
from numpy.random.bit_generator import BitGenerator, SeedSequence

from qiskit.circuit import QuantumCircuit, Instruction, CircuitInstruction, Barrier, Gate
from qiskit.circuit.library import get_standard_gate_name_mapping
from qiskit.transpiler import CouplingMap, generate_preset_pass_manager, Target
from qiskit.exceptions import QiskitError
from qiskit.providers.backend import Backend
from qiskit.quantum_info import Operator

from qiskit_experiments.framework import BaseExperiment, Options
from qiskit_experiments.framework.configs import ExperimentConfig

from .clifford_utils import (
    CliffordUtils,
    DEFAULT_SYNTHESIS_METHOD,
    _clifford_1q_int_to_instruction,
    _decompose_clifford_ops,
)
from .layer_fidelity_analysis import LayerFidelityAnalysis

LOG = logging.getLogger(__name__)


GATE_NAME_MAP = get_standard_gate_name_mapping()
NUM_1Q_CLIFFORD = CliffordUtils.NUM_CLIFFORD_1_QUBIT


class LayerFidelityUnitary(BaseExperiment):
    r"""A holistic benchmarking experiment to characterize the full quality of the devices at scale.

    # section: overview
        Unitary Layer Fidelity (ULF) is a method to estimate the fidelity of
        a connecting set of arbitrary two-qubit gates over :math:`N` qubits by measuring gate errors
        using simultaneous direct unitary randomized benchmarking (RB) in disjoint layers.
        LF can easily be expressed as a layer size independent quantity, error per layered gate (EPLG):
        :math:`EPLG = 1 - LF^{1/N_{2Q}}` where :math:`N_{2Q}` is number of 2-qubit gates in the layers.

        Each of the 2-qubit (or 1-qubit) direct RBs yields the decaying probabilities
        to get back to the ground state for an increasing sequence length (i.e. number of layers),
        fits the exponential curve to estimate the decay rate, and calculates
        the process fidelity of the subsystem from the rate.
        LF is calculated as the product of the 2-qubit (or 1-qubit) process fidelities.
        See Ref. [1] for details.

        This unitary version allows artibrary 2Q gates

    # section: analysis_ref
        :class:`LayerFidelityAnalysis`

    # section: reference
        .. ref_arxiv:: 1 2311.05933

    # section: example
        .. jupyter-execute::
            :hide-code:

            # backend
            from qiskit_aer import AerSimulator
            from qiskit_aer.noise import NoiseModel, depolarizing_error

            noise_model = NoiseModel()
            noise_model.add_all_qubit_quantum_error(depolarizing_error(5e-3, 1), ["sx", "x"])
            noise_model.add_all_qubit_quantum_error(depolarizing_error(0, 1), ["rz"])
            noise_model.add_all_qubit_quantum_error(depolarizing_error(5e-2, 2), ["rzz"])
            backend = AerSimulator(noise_model=noise_model)

        .. jupyter-execute::

            import numpy as np
            from qiskit import QuantumCircuit
            from qiskit.circuit.library import RZZGate
            from qiskit_experiments.library.randomized_benchmarking import LayerFidelityUnitary

            lengths = np.arange(1, 80, 10)
            two_qubit_layers=[[(0, 1), (3, 5)], [(1, 3), (5, 6)]]

            num_samples = 3
            seed = 106

            # Can load this way if benchmarking a generic circuit
            # qc = QuantumCircuit(2)
            # qc.rzz(0.5,0,1)
            # two_qubit_gates=[qc.to_instruction()]

            exp = LayerFidelityUnitary(
                    physical_qubits=[0, 1, 3, 5, 6],
                    two_qubit_layers=two_qubit_layers,
                    lengths=lengths,
                    backend=backend,
                    num_samples=num_samples,
                    seed=seed,
                    two_qubit_gates=[RZZGate(0.5)],
            )

            exp_data = exp.run().block_for_results()
            results = exp_data.analysis_results(dataframe=True)

            display(exp_data.figure(0)) # one of 6 figures
            display(exp_data.analysis_results("EPLG", dataframe=True))

            print(f"Available results: {set(results.name)}")
    """

    def __init__(
        self,
        physical_qubits: Sequence[int],
        two_qubit_layers: Sequence[Sequence[Tuple[int, int]]],
        lengths: Iterable[int],
        two_qubit_gates: Sequence[Union[Instruction, Gate]],
        num_samples: int = 6,
        backend: Optional[Backend] = None,
        seed: Optional[Union[int, SeedSequence, BitGenerator, Generator]] = None,
        two_qubit_basis_gates: Optional[Sequence[str]] = None,
        one_qubit_basis_gates: Optional[Sequence[str]] = None,
        layer_barrier: Optional[bool] = True,
        min_delay: Optional[Sequence[int]] = None,
        benchmark_suffix: Optional[str] = "",
    ):
        """Initialize a unitary layer fidelity experiment.

        Args:
            physical_qubits: List of physical qubits for the experiment.
            two_qubit_layers: List of two-qubit gate layers to run on. Each two-qubit
                    gate layer must be given as a list of directed qubit pairs.
            lengths: A list of layer lengths (the number of depth points).
            two_qubit_gates: A list of two qubit circuit instructions or gates that will be in the
                            entangling layer. If more than one than they are sampled from this list.
                            These are assumed to be the backend ISA already.
            num_samples: Number of samples (i.e. circuits) to generate for each layer length.
            backend: Optional, the backend to run the experiment on. Note that either ``backend`` or
                    ``two_qubit_gate`` and ``one_qubit_basis_gates`` must be set at instantiation.
            seed: Optional, seed used to initialize ``numpy.random.default_rng``.
                  when generating circuits. The ``default_rng`` will be initialized
                  with this seed value every time :meth:~.LayerFidelity.circuits` is called.
            two_qubit_basis_gates: Optional, 2q-gates to use for transpiling the inverse.
                            If not specified (but ``backend`` is supplied),
                            all 2q-gates supported in the backend are automatically set.
            one_qubit_basis_gates: Optional, 1q-gates to use for implementing 1q-Clifford operations.
                            If not specified (but ``backend`` is supplied),
                            all 1q-gates supported in the backend are automatically set.
            layer_barrier (bool): Optional, enforce a barrier across the whole layer.
                Default is True, which is the defined protocol for layer fidelity.
                If this is set to false the code runs
                simultaneous direct 1+2Q RB without a barrier across all qubits.
            min_delay: Optional. Define a minimum delay in each 2Q layer in units of dt. This
                delay operation will be applied in any 1Q edge of the layer during the 2Q gate layer
                in order to enforce a minimum duration of the 2Q layer. This enables some crosstalk
                testing by removing a gate from the layer without changing the layer duration. If not
                None then is a list equal in length to the number of two_qubit_layers.  Note that
                this options requires at least one 1Q edge (a qubit in physical_qubits but
                not in two_qubit_layers) to be applied. Also will not have an impact on the 2Q gates
                if layer_barrier=False.
            benchmark_suffix (str): Optional. Suffix string to be appended to the end of the names of the
                associated analysis results. Intended to allow for easier bookeeping if multiple types of
                gates are being tracked.

        Raises:
            QiskitError: If any invalid argument is supplied.
        """
        # Compute full layers
        full_layers = []
        for two_q_layer in two_qubit_layers:
            qubits_in_layer = {q for qpair in two_q_layer for q in qpair}
            if len(qubits_in_layer) != 2 * len(two_q_layer):
                raise QiskitError("two_qubit_layers have a layer with gates on non-disjoint qubits")
            for q in qubits_in_layer:
                if q not in physical_qubits:
                    raise QiskitError(f"Qubit {q} in two_qubit_layers is not in physical_qubits")
            layer = two_q_layer + [(q,) for q in physical_qubits if q not in qubits_in_layer]
            full_layers.append(layer)

        # Initialize base experiment
        super().__init__(
            physical_qubits, analysis=LayerFidelityAnalysis(full_layers), backend=backend
        )

        self.analysis.benchmark_suffix = benchmark_suffix
        # assert isinstance(backend, BackendV2)

        # Verify parameters
        if len(set(lengths)) != len(lengths):
            raise QiskitError(f"The lengths list {lengths} should not contain duplicate elements.")
        if num_samples <= 0:
            raise QiskitError(f"The number of samples {num_samples} should be positive.")

        if two_qubit_gates is None:
            raise QiskitError("Must specify a set of two qubit gate circuits.")

        if two_qubit_basis_gates is None:
            if self.backend is None:
                raise QiskitError("two_qubit_basis_gates or backend must be supplied.")
            # Try to set default two qubit basis gate from backend
            two_qubit_basis_gates = []
            for op in self.backend.target.operations:
                if isinstance(op, Gate) and op.num_qubits == 2:
                    two_qubit_basis_gates.append(op.name)
                    LOG.info("%s is set for two_qubit_gate", op.name)
                    break
            if not two_qubit_basis_gates:
                raise QiskitError("two_qubit_gate is not provided and failed to set from backend.")

        if one_qubit_basis_gates is None:
            if self.backend is None:
                raise QiskitError("one_qubit_basis_gates or backend must be supplied.")
            # Try to set default one_qubit_basis_gates from backend
            one_qubit_basis_gates = []
            for op in self.backend.target.operations:
                if isinstance(op, Gate) and op.num_qubits == 1:
                    if op.name in GATE_NAME_MAP:
                        one_qubit_basis_gates.append(op.name)
                    else:
                        warnings.warn(
                            f'Not using single qubit gate "{op.name}". Please '
                            "open an issue if support for using gates outside "
                            "of Qiskit's standard gates is needed for layer "
                            "fidelity."
                        )
            LOG.info("%s is set for one_qubit_basis_gates", str(one_qubit_basis_gates))
            if not one_qubit_basis_gates:
                raise QiskitError(
                    "one_qubit_basis_gates is not provided and failed to set from backend."
                )
        else:
            if self.backend is None:
                for gate in one_qubit_basis_gates:
                    if gate not in GATE_NAME_MAP:
                        raise QiskitError(f"Unknown gate in one_qubit_basis_gates: {gate}.")

        # Set configurable options
        self.set_experiment_options(
            lengths=sorted(lengths),
            num_samples=num_samples,
            seed=seed,
            two_qubit_layers=two_qubit_layers,
            two_qubit_gates=two_qubit_gates,
            two_qubit_basis_gates=tuple(two_qubit_basis_gates),
            one_qubit_basis_gates=tuple(one_qubit_basis_gates),
            layer_barrier=layer_barrier,
            min_delay=min_delay,
        )

        # Verify two_qubit_gate and one_qubit_basis_gates
        self.__validate_basis_gates()

    @classmethod
    def _default_experiment_options(cls) -> Options:
        """Default experiment options.

        Experiment Options:
            two_qubit_layers (List[List[Tuple[int, int]]]): List of two-qubit gate layers to run on.
                Each two-qubit gate layer must be given as a list of directed qubit pairs.
            lengths (List[int]): A list of layer lengths.
            num_samples (int): Number of samples to generate for each layer length.
            seed (None or int or SeedSequence or BitGenerator or Generator): A seed
                used to initialize ``numpy.random.default_rng`` when generating circuits.
                The ``default_rng`` will be initialized with this seed value every time
                :meth:`circuits` is called.
            two_qubit_gates (list of gates or circuit instructions): Two qubit circuits
            two_qubit_basis_gates (Tuple[str]): Two-qubit gates to use for implementing inverse.
            one_qubit_basis_gates (Tuple[str]): One-qubit gates to use for implementing 1q Cliffords.
            clifford_synthesis_method (str): The name of the Clifford synthesis plugin to use
                for building circuits of RB sequences. See :ref:`synth-methods-lbl`.
            layer_barrier (bool): Optional, enforce a barrier across the whole layer.
                Default is True, which is the defined protocol for layer fidelity.
                If this is set to false the code runs
                simultaneous direct 1+2Q RB without a barrier across all qubits.
            min_delay (List[int]): Optional. Define a minimum delay in each 2Q layer in units of dt. This
                delay operation will be applied in any 1Q edge of the layer during the 2Q gate layer
                in order to enforce a minimum duration of the 2Q layer. This enables some crosstalk
                testing by removing a gate from the layer without changing the layer duration. If not
                None then is a list equal in length to the number of two_qubit_layers.  Note that
                this options requires at least one 1Q edge (a qubit in physical_qubits but
                not in two_qubit_layers) to be applied. Also will not have an impact on the 2Q gates
                if layer_barrier=False.
        """
        options = super()._default_experiment_options()
        options.update_options(
            lengths=None,
            num_samples=None,
            seed=None,
            two_qubit_layers=None,
            two_qubit_gates=None,
            two_qubit_basis_gates=None,
            one_qubit_basis_gates=None,
            clifford_synthesis_method=DEFAULT_SYNTHESIS_METHOD,
            layer_barrier=True,
            min_delay=None,
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
            "Custom transpile options are not supported for LayerFidelity experiments."
        )

    def _set_backend(self, backend: Backend):
        """Set the backend V2 for RB experiments since RB experiments only support BackendV2."""
        super()._set_backend(backend)
        self.__validate_basis_gates()

    def __validate_basis_gates(self) -> None:
        if not self.backend:
            return
        opts = self.experiment_options

        target = self.backend.target

        # validate two_qubit_gates list
        if opts.two_qubit_gates:
            for twoq_gate in opts.two_qubit_gates:

                if twoq_gate.num_qubits != 2:
                    raise QiskitError(f"{twoq_gate.name} in two_qubit_gates is not a 2Q object")

                if isinstance(twoq_gate, Gate):
                    if twoq_gate.name not in target.operation_names:
                        raise QiskitError(
                            f"{twoq_gate.name} in two_qubit_gates is not in backend.target"
                        )
                    for two_q_layer in opts.two_qubit_layers:
                        for qpair in two_q_layer:
                            if not target.instruction_supported(twoq_gate.name, qpair):
                                raise QiskitError(
                                    f"{twoq_gate.name}{qpair} is not in backend.target"
                                )
                elif isinstance(twoq_gate, Instruction):
                    for circ_instr in twoq_gate.definition:
                        if not isinstance(circ_instr, CircuitInstruction):
                            raise QiskitError(
                                f"{twoq_gate.name} does not decompose into CircuitInstruction objects."
                            )

                        if circ_instr.operation.name not in target.operation_names:
                            raise QiskitError(
                                f"{circ_instr.operation.name} in two_qubit_gates is "
                                + "not in backend.target"
                            )

                        if circ_instr.operation.num_qubits == 1:
                            for q in self.physical_qubits:
                                if not target.instruction_supported(
                                    circ_instr.operation.name, (q,)
                                ):
                                    raise QiskitError(
                                        f"{circ_instr.operation.name}({q}) is not "
                                        + "in backend.target"
                                    )

                        if circ_instr.operation.num_qubits == 2:
                            for two_q_layer in opts.two_qubit_layers:
                                for qpair in two_q_layer:
                                    if not target.instruction_supported(
                                        circ_instr.operation.name, qpair
                                    ):
                                        raise QiskitError(
                                            f"{circ_instr.operation.name}{qpair} is not in "
                                            + "backend.target"
                                        )

        # validate two_qubit_basis_gates if it is set
        for gate in opts.two_qubit_basis_gates or []:
            if gate not in target.operation_names:
                raise QiskitError(f"{gate} in two_qubit_basis_gates is not in backend.target")
        for gate in opts.two_qubit_basis_gates or []:
            for two_q_layer in opts.two_qubit_layers:
                for qpair in two_q_layer:
                    if not target.instruction_supported(gate, qpair):
                        raise QiskitError(f"{gate}{qpair} is not in backend.target")

        # validate one_qubit_basis_gates if it is set
        for gate in opts.one_qubit_basis_gates or []:
            if gate not in target.operation_names:
                raise QiskitError(f"{gate} in one_qubit_basis_gates is not in backend.target")
        for gate in opts.one_qubit_basis_gates or []:
            for q in self.physical_qubits:
                if not target.instruction_supported(gate, (q,)):
                    raise QiskitError(f"{gate}({q}) is not in backend.target")

    def __residual_qubits(self, two_qubit_layer):
        qubits_in_layer = {q for qpair in two_qubit_layer for q in qpair}
        return [q for q in self.physical_qubits if q not in qubits_in_layer]

    def circuits(self) -> List[QuantumCircuit]:
        r"""Return a list of physical circuits to measure layer fidelity.

        Returns:
            A list of :class:`QuantumCircuit`\s.
        """
        return list(self.circuits_generator())

    def circuits_generator(self) -> Iterable[QuantumCircuit]:
        r"""Return a generator of physical circuits to measure layer fidelity.

        Returns:
            A generator of :class:`QuantumCircuit`\s.
        """
        opts = self.experiment_options
        residual_qubits_by_layer = [
            self.__residual_qubits(layer) for layer in opts.two_qubit_layers
        ]
        rng = default_rng(seed=opts.seed)
        # define functions and variables for speed
        _to_gate_1q = functools.partial(
            _clifford_1q_int_to_instruction,
            basis_gates=opts.one_qubit_basis_gates,
            synthesis_method=opts.clifford_synthesis_method,
        )

        # matrices for the two qubit gates
        two_q_gate_mats = []
        for two_q_gate in opts.two_qubit_gates:
            two_q_gate_mats.append(Operator(two_q_gate).to_matrix())

        # warn if min delay is not None and barrier is false
        if opts.min_delay is not None and not opts.layer_barrier:
            warnings.warn("Min delay applied when layer_barrier is False.")

        # Circuit generation
        num_qubits = max(self.physical_qubits) + 1
        for i_sample in range(opts.num_samples):
            for i_set, (two_qubit_layer, one_qubits) in enumerate(
                zip(opts.two_qubit_layers, residual_qubits_by_layer)
            ):
                num_2q_gates = len(two_qubit_layer)
                num_1q_gates = len(one_qubits)
                composite_qubits = two_qubit_layer + [(q,) for q in one_qubits]
                composite_clbits = [(2 * c, 2 * c + 1) for c in range(num_2q_gates)]
                composite_clbits.extend(
                    [(c,) for c in range(2 * num_2q_gates, 2 * num_2q_gates + num_1q_gates)]
                )

                if opts.min_delay is None:
                    min_delay = None
                else:
                    min_delay = opts.min_delay[i_set]

                # cache the 1Q cliffords
                oneq_cliff_mats = [CliffordUtils.clifford_1_qubit(i).to_matrix() for i in range(24)]

                # generate the pass manager
                target = Target.from_configuration(
                    num_qubits=2,
                    basis_gates=opts.two_qubit_basis_gates + opts.one_qubit_basis_gates,
                    coupling_map=CouplingMap(((0, 1),)),
                )
                pass_manager_2q = generate_preset_pass_manager(
                    optimization_level=1,
                    target=target,
                    backend=self.backend,
                )
                target = Target.from_configuration(
                    num_qubits=1,
                    basis_gates=opts.one_qubit_basis_gates,
                    coupling_map=None,
                )
                pass_manager_1q = generate_preset_pass_manager(
                    optimization_level=1,
                    target=target,
                    backend=self.backend,
                )

                for length in opts.lengths:
                    circ = QuantumCircuit(num_qubits, num_qubits)
                    # define the barrier instruction
                    full_barrier_inst = CircuitInstruction(Barrier(num_qubits), circ.qubits)
                    if not opts.layer_barrier:
                        # we want separate barriers for each qubit so define them individually
                        barrier_inst_gate = []
                        for two_q_gate in two_qubit_layer:
                            barrier_inst_gate.append(
                                CircuitInstruction(
                                    Barrier(2),
                                    [circ.qubits[two_q_gate[0]], circ.qubits[two_q_gate[1]]],
                                )
                            )
                        for one_q in one_qubits:
                            barrier_inst_gate.append(
                                CircuitInstruction(Barrier(1), [circ.qubits[one_q]])
                            )
                    else:
                        barrier_inst_gate = [full_barrier_inst]
                    self.__circuit_body(
                        circ,
                        length,
                        two_qubit_layer,
                        one_qubits,
                        rng,
                        _to_gate_1q,
                        opts.two_qubit_gates,
                        two_q_gate_mats,
                        barrier_inst_gate,
                        oneq_cliff_mats,
                        pass_manager_2q,
                        pass_manager_1q,
                        min_delay,
                    )
                    # add the measurements
                    circ._append(full_barrier_inst)
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
        two_qubit_gates,
        two_q_gate_mats,
        barrier_inst_lst,
        oneq_cliff_mats,
        pass_manager_2q,
        pass_manager_1q,
        min_delay=None,
    ):

        # warn if min_delay is not none and one_qubits is empty
        if min_delay is not None and len(one_qubits) == 0:
            warnings.warn("Min delay will not be applied because there are no 1Q edges.")

        # start with a set of identity matrices
        circs_2q = [np.eye(4, dtype=complex) for ii in range(len(two_qubit_layer))]
        circs_1q = [np.eye(2, dtype=complex) for ii in range(len(one_qubits))]

        for _ in range(length):
            # sample random 1q-Clifford layer
            for j, qpair in enumerate(two_qubit_layer):
                # sample product of two 1q-Cliffords as 2q interger Clifford
                samples = rng.integers(NUM_1Q_CLIFFORD, size=2)
                # multiply unitaries for the 1Q cliffords we sampled
                np.dot(
                    np.kron(oneq_cliff_mats[samples[1]], oneq_cliff_mats[samples[0]]),
                    circs_2q[j],
                    out=circs_2q[j],
                )
                for sample, q in zip(samples, qpair):
                    circ._append(_to_gate_1q(sample), (circ.qubits[q],), ())
            for k, q in enumerate(one_qubits):
                sample = rng.integers(NUM_1Q_CLIFFORD)
                circ._append(_to_gate_1q(sample), (circ.qubits[q],), ())
                np.dot(oneq_cliff_mats[sample], circs_1q[k], out=circs_1q[k])
            for barrier_inst in barrier_inst_lst:
                circ._append(barrier_inst)
            # add two qubit gates
            for j, qpair in enumerate(two_qubit_layer):
                sample = rng.integers(len(two_qubit_gates))
                circ._append(two_qubit_gates[sample], tuple(circ.qubits[q] for q in qpair), ())
                np.dot(two_q_gate_mats[sample], circs_2q[j], out=circs_2q[j])
                # TODO: add dd if necessary
            for k, q in enumerate(one_qubits):
                # TODO: add dd if necessary
                # if there is a min_delay, just need
                # to add to one of the qubits
                if min_delay is not None and k == 0:
                    circ.delay(min_delay, q)
            for barrier_inst in barrier_inst_lst:
                circ._append(barrier_inst)

        # add the last inverse
        # invert the unitary matrix and transpile to a proper
        # circuit
        for j, qpair in enumerate(two_qubit_layer):

            qc_tmp = QuantumCircuit(2)
            qc_tmp.unitary(circs_2q[j].conjugate().transpose(), [0, 1], label="test")
            qc_tmp = pass_manager_2q.run(qc_tmp)
            circ._append(qc_tmp.to_instruction(), tuple(circ.qubits[q] for q in qpair), ())

        for k, q in enumerate(one_qubits):
            qc_tmp = QuantumCircuit(1)
            qc_tmp.unitary(circs_1q[k].conjugate().transpose(), [0], label="test")
            qc_tmp = pass_manager_1q.run(qc_tmp)
            circ._append(qc_tmp.to_instruction(), (circ.qubits[q],), ())

        return circ

    def _transpiled_circuits(self) -> List[QuantumCircuit]:
        """Return a list of experiment circuits, transpiled."""
        transpiled = [_decompose_clifford_ops(circ, True) for circ in self.circuits()]

        return transpiled

    def _metadata(self):
        metadata = super()._metadata()
        metadata["two_qubit_layers"] = self.experiment_options.two_qubit_layers
        return metadata

    @classmethod
    def from_config(cls, config: Union[ExperimentConfig, Dict]) -> "LayerFidelity":
        """Initialize an experiment from experiment config"""
        if isinstance(config, dict):
            config = ExperimentConfig(**config)
        ret = cls(*config.args, **config.kwargs)
        if config.run_options:
            ret.set_run_options(**config.run_options)
        return ret
