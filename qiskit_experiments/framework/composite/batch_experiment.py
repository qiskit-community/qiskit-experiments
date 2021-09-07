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
Batch Experiment class.
"""

from typing import List

from qiskit import QuantumCircuit


from .composite_experiment import CompositeExperiment


class BatchExperiment(CompositeExperiment):
    """Batch experiment class.

    This experiment takes multiple experiment instances and generates
    a list of flattened circuit to execute.
    The experimental circuits are executed ony be one on the target backend as a single job.

    If an experiment analysis needs results of different types of experiments,
    ``BatchExperiment`` may be convenient to describe the flow of the entire experiment.

    The experimental result of ``i``-th experiment can be accessed by

    .. code-block:: python3

        batch_exp = BatchExperiment([exp1, exp2, exp3])
        batch_result = batch_exp.run(backend)

        exp1_res = batch_result.component_experiment_data(0)  # data of exp1
        exp2_res = batch_result.component_experiment_data(1)  # data of exp2
        exp3_res = batch_result.component_experiment_data(2)  # data of exp3

    One can also create a custom analysis class that estimates some parameters by
    combining above analysis results. Here the ``exp*_res`` is a single
    :py:class:`~qiskit_experiments.framework.experiment_data.ExperimentData` class of
    a standard experiment, and the associated analysis will be performed once the batch job
    is completed. Thus analyzed parameter value of each experiment can be obtained as usual.

    .. code-block:: python3

        param_x = exp1_res.analysis_results("target_parameter_x")
        param_y = exp2_res.analysis_results("target_parameter_y")
        param_z = exp3_res.analysis_results("target_parameter_z")

        param_xyz = param_x + param_y + param_z  # do some computation

    The final parameter ``param_xyz`` can be returned as an outcome of this batch experiment.
    """

    def __init__(self, experiments):
        """Initialize a batch experiment.

        Args:
            experiments (List[BaseExperiment]): a list of experiments.
        """
        qubits = sorted(set(sum([list(expr.physical_qubits) for expr in experiments], [])))
        super().__init__(experiments, qubits)

    def _flatten_circuits(
            self,
            circuits: List[List[QuantumCircuit]],
            num_qubits: int,
    ) -> List[QuantumCircuit]:
        """Flatten circuits.

        Note:
            This experiment concatenates sub experiment circuits.
        """
        batch_circuits = []

        for expr_idx, sub_circs in enumerate(circuits):
            for sub_circ in sub_circs:
                sub_circ.metadata = {
                    "experiment_type": self._type,
                    "composite_index": [expr_idx],
                    "composite_metadata": sub_circ.metadata,
                }
                batch_circuits.append(sub_circ)

        return batch_circuits
