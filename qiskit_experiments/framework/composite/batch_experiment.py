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

from typing import List, Optional, Dict
from collections import OrderedDict, defaultdict

from qiskit import QuantumCircuit
from qiskit.providers import Job, Backend, Options
from qiskit_ibm_runtime import SamplerV2 as Sampler

from .composite_experiment import CompositeExperiment, BaseExperiment
from .composite_analysis import CompositeAnalysis


class BatchExperiment(CompositeExperiment):
    """Combine multiple experiments into a batch experiment.

    Batch experiments combine individual experiments on any subset of qubits
    into a single composite experiment which appends all the circuits from
    each component experiment into a single batch of circuits to be executed
    as one experiment job.

    Analysis of batch experiments is performed using the
    :class:`~qiskit_experiments.framework.CompositeAnalysis` class which handles
    sorting the composite experiment circuit data into individual child
    :class:`ExperimentData` containers for each component experiment which are
    then analyzed using the corresponding analysis class for that component
    experiment.

    See :class:`~qiskit_experiments.framework.CompositeAnalysis`
    documentation for additional information.
    """

    def __init__(
        self,
        experiments: List[BaseExperiment],
        backend: Optional[Backend] = None,
        flatten_results: bool = True,
        analysis: Optional[CompositeAnalysis] = None,
        experiment_type: Optional[str] = None,
    ):
        """Initialize a batch experiment.

        Args:
            experiments: a list of experiments.
            backend: Optional, the backend to run the experiment on.
            flatten_results: If True flatten all component experiment results
                             into a single ExperimentData container, including
                             nested composite experiments. If False save each
                             component experiment results as a separate child
                             ExperimentData container. This kwarg is ignored
                             if the analysis kwarg is used.
            analysis: Optional, the composite analysis class to use. If not
                      provided this will be initialized automatically from the
                      supplied experiments.
        """
        # Generate qubit map
        self._qubit_map = OrderedDict()
        logical_qubit = 0
        for expr in experiments:
            for physical_qubit in expr.physical_qubits:
                if physical_qubit not in self._qubit_map:
                    self._qubit_map[physical_qubit] = logical_qubit
                    logical_qubit += 1
        qubits = tuple(self._qubit_map.keys())
        super().__init__(
            experiments,
            qubits,
            backend=backend,
            analysis=analysis,
            flatten_results=flatten_results,
            experiment_type=experiment_type,
        )

    def circuits(self):
        return self._batch_circuits(to_transpile=False)

    def _transpiled_circuits(self):
        return self._batch_circuits(to_transpile=True)

    def _batch_circuits(self, to_transpile=False):
        batch_circuits = []

        # Generate data for combination
        for index, expr in enumerate(self._experiments):
            if self.physical_qubits == expr.physical_qubits or to_transpile:
                qubit_mapping = None
            else:
                qubit_mapping = [self._qubit_map[qubit] for qubit in expr.physical_qubits]

            if isinstance(expr, BatchExperiment):
                # Batch experiments don't contain their own native circuits.
                # If to_transpile is True then the circuits will be transpiled at the non-batch
                # experiments.
                # Fetch the circuits from the sub-experiments.
                expr_circuits = expr._batch_circuits(to_transpile)
            elif to_transpile:
                expr_circuits = expr._transpiled_circuits()
            else:
                expr_circuits = expr.circuits()

            for circuit in expr_circuits:
                # Update metadata
                circuit.metadata = {
                    "experiment_type": self._type,
                    "composite_metadata": [circuit.metadata],
                    "composite_index": [index],
                }
                # Remap qubits if required
                if qubit_mapping:
                    circuit = self._remap_qubits(circuit, qubit_mapping)
                batch_circuits.append(circuit)

        return batch_circuits

    def _remap_qubits(self, circuit, qubit_mapping):
        """Remap qubits if physical qubit layout is different to batch layout"""
        num_qubits = self.num_qubits
        num_clbits = circuit.num_clbits
        new_circuit = QuantumCircuit(num_qubits, num_clbits, name="batch_" + circuit.name)
        new_circuit.metadata = circuit.metadata
        new_circuit.append(circuit, qubit_mapping, list(range(num_clbits)))
        return new_circuit

    def _run_jobs_recursive(
        self,
        circuits: List[QuantumCircuit],
        truncated_metadata: List[Dict],
        sampler: Sampler = None,
        **run_options,
    ) -> List[Job]:
        # The truncated metadata is a truncation of the original composite metadata.
        # During the recursion, the current experiment (self) will be at the head of the truncated
        # metadata.

        if self.experiment_options.separate_jobs:
            # A dictionary that maps sub-experiments to their circuits
            circs_by_subexps = defaultdict(list)
            for circ_iter, (circ, tmd) in enumerate(zip(circuits, truncated_metadata)):
                # For batch experiments the composite index is always a list of length 1,
                # because unlike parallel experiment, each circuit originates from a single
                # sub-experiment.
                circ_index = tmd["composite_index"][0]
                circs_by_subexps[circ_index].append(circ)

                # For batch experiments the composite metadata is always a list of length 1,
                # because unlike parallel experiment, each circuit originates from a single
                # sub-experiment.
                truncated_metadata[circ_iter] = tmd["composite_metadata"][0]

            jobs = []
            for index, exp in enumerate(self.component_experiment()):
                # Currently all the sub-experiments must use the same set of run options,
                # even if they run in different jobs
                if isinstance(exp, BatchExperiment):
                    new_jobs = exp._run_jobs_recursive(
                        circs_by_subexps[index], truncated_metadata, sampler, **run_options
                    )
                else:
                    new_jobs = exp._run_jobs(circs_by_subexps[index], sampler, **run_options)
                jobs.extend(new_jobs)
        else:
            jobs = super()._run_jobs(circuits, sampler, **run_options)

        return jobs

    def _run_jobs(
        self, circuits: List[QuantumCircuit], sampler: Sampler = None, **run_options
    ) -> List[Job]:
        truncated_metadata = [circ.metadata for circ in circuits]
        jobs = self._run_jobs_recursive(circuits, truncated_metadata, sampler, **run_options)
        return jobs

    @classmethod
    def _default_experiment_options(cls) -> Options:
        """Default experiment options.

        Experiment Options:
            separate_jobs (Boolean): Whether to route different sub-experiments to different jobs.
        """
        options = super()._default_experiment_options()
        options.separate_jobs = False
        return options
