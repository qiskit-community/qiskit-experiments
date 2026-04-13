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

.. warning::
    **Caveat Emptor**: This module transpiles circuits once and then remaps qubits
    without re-transpiling for tiled copies. This approach prioritizes speed over
    correctness in edge cases. Users should be aware that:

    - Pulse calibrations may not be correctly remapped
    - Backend-specific optimizations may not apply to all qubit mappings
    - Connectivity constraints are assumed to be satisfied after remapping

    This is primarily intended for use cases where transpilation overhead is significant
    and the experiment structure is simple enough that qubit remapping is safe.
"""

from copy import deepcopy
from qiskit.circuit import QuantumCircuit, Qubit, Clbit
from .composite_experiment import BaseExperiment
from .batch_experiment import BatchExperiment
from .parallel_experiment import ParallelExperiment


class BasicExperiment(BaseExperiment):
    """
    Basic atomic experiment that mimics the template experiment,
    but uses a pre-prepared transpiled circuit.

    This is an internal helper class used by TiledExperiment to create
    experiment instances with remapped qubits from a transpiled template.
    """

    def __init__(self, qubits, template_circs, analysis):
        """
        Initialize a BasicExperiment.

        Args:
            qubits: Physical qubit indices for this experiment instance
            template_circs: Pre-transpiled circuits to remap
            analysis: Analysis instance (will be deep copied)
        """
        super().__init__(qubits)
        self._template_circs = template_circs
        self.analysis = analysis

    def circuits(self):
        """Required by BaseExperiment but not used since we override _transpiled_circuits."""
        pass

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

            for inst, qargs, cargs in circ.data:
                new_qargs = []
                new_cargs = []

                for qubit in qargs:
                    original_qubit = qubit_indices[qubit]
                    if original_qubit < len(self.physical_qubits):
                        new_qargs.append(
                            Qubit(new_circ.qregs[0], self.physical_qubits[original_qubit])
                        )

                for clbit in cargs:
                    original_clbit = clbit_indices[clbit]
                    new_cargs.append(Clbit(new_circ.cregs[0], original_clbit))

                if len(qargs) == len(new_qargs):
                    new_circ.append(inst, new_qargs, new_cargs)

            new_circ.metadata = circ.metadata
            res_circs.append(new_circ)

        return res_circs


class TiledExperiment(BatchExperiment):
    """
    Duplicate a given experiment across the device.

    This class creates a batch experiment that runs copies of a template experiment
    on different groups of qubits across a device. The template experiment is transpiled
    once, and then the transpiled circuits are remapped to different physical qubits
    without re-transpiling.

    .. warning::
        This approach prioritizes speed over correctness. The transpilation is done once
        and circuits are remapped by changing qubit indices. This may not correctly handle:

        - Pulse-level calibrations
        - Backend-specific gate decompositions
        - Connectivity-dependent optimizations

        Use with caution and verify results, especially when using pulse gates or
        backend-specific features.

    Example:
        >>> from qiskit_experiments.library import StandardRB
        >>> from qiskit_experiments.framework.composite import TiledExperiment
        >>> from qiskit_experiments.framework.composite.tiled_experiment_utils import (
        ...     partition_qubits
        ... )
        >>>
        >>> # Create a template RB experiment for 2 qubits
        >>> template_exp = StandardRB([0, 1], lengths=[10, 20, 30])
        >>> template_exp.set_transpile_options(optimization_level=3)
        >>>
        >>> # Partition the backend qubits with minimum distance of 3
        >>> groups = partition_qubits(backend, distance=3)
        >>>
        >>> # Create tiled experiment
        >>> tiled_exp = TiledExperiment(template_exp, groups)
        >>> tiled_exp.run(backend)
    """

    def __init__(self, template_experiment, groups):
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
