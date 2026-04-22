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
"""
Build composite experiments for entire backends
"""

from copy import deepcopy
from typing import List, Sequence
from qiskit.circuit import QuantumCircuit, Qubit, Clbit
from .composite_experiment import BaseExperiment
from .batch_experiment import BatchExperiment
from .parallel_experiment import ParallelExperiment


class BasicExperiment(BaseExperiment):
    """
    This is an internal helper class used by :class:`.TiledExperiment` to create
    experiment instances with remapped qubits from a transpiled template.
    """

    def __init__(self, qubits: Sequence[int], template_circs: List[QuantumCircuit], analysis):
        """
        Initialize a BasicExperiment.

        Args:
            qubits: Physical qubit indices for this experiment instance
            template_circs: Pre-transpiled circuits to remap
            analysis: Analysis instance
        """
        super().__init__(qubits)
        self._template_circs = template_circs
        self.analysis = analysis

    def circuits(self):
        """Required by BaseExperiment but not used since we override _transpiled_circuits."""
        return self._template_circs

    def _transpiled_circuits(self):
        """
        Remap the template circuits to the physical qubits for this experiment.

        Returns:
            List of circuits with remapped qubits
        """
        res_circs = []
        for circ in self._template_circs:
            qubit_indices = {bit: idx for idx, bit in enumerate(circ.qubits)}
            clbit_indices = {bit: idx for idx, bit in enumerate(circ.clbits)}
            new_circ = QuantumCircuit(1 + max(self.physical_qubits), circ.num_clbits)

            for instruction in circ.data:
                new_qargs = []
                new_cargs = []

                for qubit in instruction.qubits:
                    original_qubit = qubit_indices[qubit]
                    if original_qubit < len(self.physical_qubits):
                        new_qargs.append(
                            Qubit(new_circ.qregs[0], self.physical_qubits[original_qubit])
                        )

                for clbit in instruction.clbits:
                    original_clbit = clbit_indices[clbit]
                    new_cargs.append(Clbit(new_circ.cregs[0], original_clbit))

                if len(instruction.qubits) == len(new_qargs):
                    new_circ.append(instruction.operation, new_qargs, new_cargs)

            new_circ.metadata = circ.metadata
            res_circs.append(new_circ)

        return res_circs


class TiledExperiment(BatchExperiment):
    """
    Composite experiment that duplicates a given experiment across the device.

    This class creates a batch experiment that runs copies of a template experiment
    on different groups of qubits across a device. The template experiment is transpiled
    once, and then the transpiled circuits are remapped to different physical qubits
    without re-transpiling.

    .. warning::
        **Caveat Emptor**: This approach prioritizes speed over correctness. The transpilation
        is done once and circuits are remapped by changing qubit indices. This may not correctly
        handle:

        - Qubit connectivity constraints
        - Gate directionality requirements
        - Gates only supported on a subset of qubits
        - Parameters (like delays) added based on gate durations that could vary by qubit

        Additionally, the template circuit is transpiled with qubits 0-N, so the mapped groups
        must match the connectivity of the initial qubits (e.g., if qubits 0 and 1 are not
        connected, 2-qubit experiments like RB will not work correctly).

        Use with caution and verify results, especially when using backend-specific features.

    Example:

        .. jupyter-input::

            from qiskit_experiments.library import T1
            from qiskit_experiments.framework.composite import TiledExperiment
            from qiskit_experiments.framework.backend_partition import partition_qubits

            # Create a template T1 experiment for a single qubit
            template_exp = T1([0], delays=list(range(1, 40, 3)))
            template_exp.set_transpile_options(optimization_level=3)

            # Partition the backend qubits with minimum distance of 3
            groups = partition_qubits(backend, distance=3)

            # Create tiled experiment
            tiled_exp = TiledExperiment(template_exp, groups)
            tiled_exp.run(backend)
    """

    def __init__(self, template_experiment: BaseExperiment, groups: List[List[Sequence[int]]]):
        """
        Initialize a TiledExperiment.

        Args:
            template_experiment: The experiment to tile across the device.
                This experiment will be transpiled once, and the transpiled
                circuits will be remapped to different qubit groups.
            groups: A list of groups, where each group is a list of qubit
                tuples/lists. Each group will be run in parallel, and groups
                are run in batch (sequentially).

        Example groups structure:
            [
                [[0, 1], [3, 4], [6, 7]],  # First parallel group
                [[2, 3], [5, 6], [8, 9]],  # Second parallel group
            ]
        """
        circs = template_experiment._transpiled_circuits()

        parexps = []
        for group in groups:
            exps = []
            for qubits in group:
                analysis = deepcopy(template_experiment.analysis)
                exps.append(
                    BasicExperiment(
                        qubits,
                        circs,
                        analysis,
                    )
                )
            parexps.append(ParallelExperiment(exps, flatten_results=True))

        super().__init__(parexps, backend=template_experiment.backend, flatten_results=True)
