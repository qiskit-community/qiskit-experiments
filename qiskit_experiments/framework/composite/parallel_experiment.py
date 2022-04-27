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
from qiskit_experiments.exceptions import QiskitError
from .composite_experiment import CompositeExperiment, BaseExperiment
from .composite_analysis import CompositeAnalysis


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

    def __init__(
        self,
        experiments: List[BaseExperiment],
        backend: Optional[Backend] = None,
        flatten_results: bool = False,
        analysis: Optional[CompositeAnalysis] = None,
    ):
        """Initialize the analysis object.

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
        qubits = []
        for exp in experiments:
            qubits += exp.physical_qubits
        super().__init__(
            experiments, qubits, backend=backend, analysis=analysis, flatten_results=flatten_results
        )

    def circuits(self):

        sub_circuits = []
        sub_qubits = []
        sub_maps = []
        sub_size = []
        num_qubits = 0

        # Generate data for combination
        for sub_exp in self._experiments:

            # Generate transpiled subcircuits
            circuits = sub_exp._transpiled_circuits()

            # Add subcircuits
            sub_circuits.append(circuits)
            sub_size.append(len(circuits))

            # Sub experiment logical qubits in the combined circuits full qubits
            qubits = list(range(num_qubits, num_qubits + sub_exp.num_qubits))
            sub_qubits.append(qubits)
            # Construct mapping for the sub-experiments logical qubits to physical qubits
            # in the full combined circuits
            sub_maps.append({q: qubits[i] for i, q in enumerate(sub_exp.physical_qubits)})
            num_qubits += sub_exp.num_qubits

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
                    qargs_map = sub_maps[exp_idx]
                    clbits = list(range(num_clbits, num_clbits + sub_circ.num_clbits))
                    circuit.add_register(ClassicalRegister(sub_circ.num_clbits))

                    # Apply transpiled subcircuit
                    # Note that this assumes the circuit was not expanded to use
                    # any qubits outside the specified physical qubits
                    for inst, qargs, cargs in sub_circ.data:
                        try:
                            mapped_qargs = [
                                circuit.qubits[qargs_map[sub_circ.find_bit(i).index]] for i in qargs
                            ]
                        except KeyError as ex:
                            # Instruction is outside physical qubits for the component
                            # experiment.
                            # This could legitimately happen if the subcircuit was
                            # explicitly scheduled during transpilation which would
                            # insert delays on all auxillary device qubits.
                            # We skip delay instructions to allow for this.
                            if inst.name == "delay":
                                continue
                            raise QiskitError(
                                "Component experiment has been transpiled outside of the "
                                "allowed physical qubits for that component. Check the "
                                "experiment is valid on the backends coupling map."
                            ) from ex
                        mapped_cargs = [
                            circuit.clbits[clbits[sub_circ.find_bit(i).index]] for i in cargs
                        ]
                        circuit._append(inst, mapped_qargs, mapped_cargs)

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
            joint_circuits.append(circuit)

        return joint_circuits
