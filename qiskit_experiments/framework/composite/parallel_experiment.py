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
Parallel Experiment class.
"""

import itertools
from collections import defaultdict
from typing import List

from qiskit import QuantumCircuit, ClassicalRegister

from .composite_experiment import CompositeExperiment


class ParallelExperiment(CompositeExperiment):
    """Parallel Experiment class.

    This experiment takes multiple experiment instances and generates
    a list of merged circuit to execute.
    The experimental circuits are executed in parallel on the target backend.

    This experiment is often used to simultaneously execute the same experiment on
    different qubit sets. For example, given we write a simple experiment
    ``BitFlipExperiment`` that flips the input quantum state and measure it.
    Then we feed multiple experiments of this kind into ``ParallelExperiment``.

    .. code-block:: python3

        exp1 = BitFlipExperiment(qubits=[0])
        exp2 = BitFlipExperiment(qubits=[1])

        parallel_exp = BatchExperiment([exp1, exp2])

    The direct transpiled output of ``exp1`` and ``exp2`` may look like

    .. parsed-literal::

        # exp1.run_transpile(backend)

             ┌───┐┌─┐
        q_0: ┤ X ├┤M├
             └───┘└╥┘
        q_1: ──────╫─
                   ║
        c: 1/══════╩═
                   0

        # exp2.run_transpile(backend)

        q_0: ────────
             ┌───┐┌─┐
        q_1: ┤ X ├┤M├
             └───┘└╥┘
        c: 1/══════╩═
                   0

    The ``ParallelExperiment`` merges these circuits into a single circuit like below.

    .. parsed-literal::

        # parallel_exp.run_transpile(backend)

              ┌───┐┌─┐
         q_0: ┤ X ├┤M├───
              ├───┤└╥┘┌─┐
         q_1: ┤ X ├─╫─┤M├
              └───┘ ║ └╥┘
        c0: 1/══════╩══╬═
                    0  ║
                       ║
        c1: 1/═════════╩═
                       0

    This parallel execution may save us from waiting a job queueing time for each experiment,
    at the expense of some extra crosstalk error possibility.

    The experimental result of ``i``-th experiment can be accessed by

    .. code-block:: python3

        parallel_result = parallel_exp.run(backend)

        exp1_res = parallel_result.component_experiment_data(0)  # data of exp1
        exp2_res = parallel_result.component_experiment_data(1)  # data of exp2

    Note that we can also combine different types of experiments, even if the length of
    generated circuits are different. If two experiment instances provide different
    length of circuits, for example, ``exp1`` and ``exp2`` create 10 and 20 circuits,
    respectively, the merged circuit will have the length of 20, and circuits with index
    from 10 to 19 will contain only experimental circuits of ``exp2``.

    Note:
        The transpile and analysis configurations of each experiment will be retained,
        however, the run configurations of nested experiments will be discarded.
        For example, if a backend provides a run option ``meas_level`` controlling a
        qubit state discrimination, we cannot set ``meas_level=1`` for ``exp1`` and
        ``meas_level=2`` for ``exp2``. Parallelized experiment will always be
        executed under the consistent run configurations.
    """

    def __init__(self, experiments):
        """Initialize the analysis object.

        Args:
            experiments (List[BaseExperiment]): a list of experiments.
        """
        qubits = []
        for exp in experiments:
            qubits += exp.physical_qubits
        super().__init__(experiments, qubits)

    def _flatten_circuits(
        self,
        circuits: List[List[QuantumCircuit]],
        num_qubits: int,
    ) -> List[QuantumCircuit]:
        """Flatten circuits.

        Note:
            This experiment merges sub experiment circuits into a single circuit
            by the circuit ``append`` method.
            Quantum and classical register indices are retained.
        """
        joint_circs = []
        for circ_idx, sub_circs in enumerate(itertools.zip_longest(*circuits)):
            joint_circ = QuantumCircuit(num_qubits, name=f"parallel_exp_{circ_idx}")
            joint_metadata = defaultdict(list)
            for expr_idx, sub_circ in enumerate(sub_circs):
                if not sub_circ:
                    # This experiment provides small number of circuits than others.
                    # No circuit available for combining with others.
                    # Skip merging process.
                    continue
                # Add sub circuits to joint circuit
                clbits = ClassicalRegister(sub_circ.num_clbits)
                joint_circ.add_register(clbits)
                joint_circ.compose(
                    sub_circ,
                    qubits=range(sub_circ.num_qubits),
                    clbits=list(clbits),
                    inplace=True,
                )
                joint_metadata["composite_index"].append(expr_idx)
                joint_metadata["composite_metadata"].append(sub_circ.metadata)
                joint_metadata["composite_qubits"].append(
                    self.component_experiment(expr_idx).physical_qubits
                )
                joint_metadata["composite_clbits"].append(
                    list(joint_circ.clbits.index(cb) for cb in clbits)
                )
                joint_circ.metadata = {"experiment_type": self._type, **joint_metadata}
            joint_circs.append(joint_circ)

        return joint_circs
