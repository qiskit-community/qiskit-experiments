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
from typing import List, Optional

from qiskit import QuantumCircuit, ClassicalRegister
from qiskit.providers.backend import Backend
from .composite_experiment import CompositeExperiment, BaseExperiment


class ParallelExperiment(CompositeExperiment):
    """Combine multiple experiments into a parallel experiment.

    Parallel experiments combine individual experiments on disjoint subsets
    of qubits into a single composite experiment on the union of those qubits.
    The component experiment circuits are combined to run in parallel on the
    respective qubits.

    Analysis of parallel experiments is performed using the
    :class:`~qiskit_experiments.framework.CompositeAnalysis` class which handles
    marginalizing the composite experiment circuit data into individual child
    :class:`ExperimentData` containers for each component experiment which are
    then analyzed using the corresponding analysis class for that component
    experiment.

    See :class:`~qiskit_experiments.framework.CompositeAnalysis`
    documentation for additional information.
    """

    def __init__(self, experiments: List[BaseExperiment], backend: Optional[Backend] = None):
        """Initialize the analysis object.

        Args:
            experiments: a list of experiments.
            backend: Optional, the backend to run the experiment on.
        """
        qubits = []
        for exp in experiments:
            qubits += exp.physical_qubits
        super().__init__(experiments, qubits, backend=backend)

    def circuits(self):

        sub_circuits = []
        sub_qubits = []
        sub_size = []
        num_qubits = 0

        # Generate data for combination
        for expr in self._experiments:
            # Add subcircuits
            circs = expr.circuits()
            sub_circuits.append(circs)
            sub_size.append(len(circs))

            # Add sub qubits
            qubits = list(range(num_qubits, num_qubits + expr.num_qubits))
            sub_qubits.append(qubits)
            num_qubits += expr.num_qubits

        # Generate empty joint circuits
        num_circuits = max(sub_size)
        joint_circuits = []
        for circ_idx in range(num_circuits):
            # Create joint circuit
            circuit = QuantumCircuit(self.num_qubits, name=f"parallel_exp_{circ_idx}")
            circuit.metadata = {
                "experiment_type": self._type,
                "composite_index": [],
                "composite_metadata": [],
                "composite_qubits": [],
                "composite_clbits": [],
            }
            for exp_idx in range(self._num_experiments):
                if circ_idx < sub_size[exp_idx]:
                    # Add subcircuits to joint circuit
                    sub_circ = sub_circuits[exp_idx][circ_idx]
                    num_clbits = circuit.num_clbits
                    qubits = sub_qubits[exp_idx]
                    clbits = list(range(num_clbits, num_clbits + sub_circ.num_clbits))
                    circuit.add_register(ClassicalRegister(sub_circ.num_clbits))
                    circuit.append(sub_circ, qubits, clbits)
                    # Add subcircuit metadata
                    circuit.metadata["composite_index"].append(exp_idx)
                    circuit.metadata["composite_metadata"].append(sub_circ.metadata)
                    circuit.metadata["composite_qubits"].append(qubits)
                    circuit.metadata["composite_clbits"].append(clbits)

                    # Add the calibrations
                    for gate, cals in sub_circ.calibrations.items():
                        for key, sched in cals.items():
                            circuit.add_calibration(
                                gate, qubits=key[0], schedule=sched, params=key[1]
                            )

            # Add joint circuit to returned list
            joint_circuits.append(circuit.decompose())

        return joint_circuits
