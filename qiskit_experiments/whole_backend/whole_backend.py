"""
Functions to build composite experiments for entire backends
"""

from copy import deepcopy
from qiskit.circuit import QuantumCircuit, Qubit, Clbit
from qiskit_experiments.framework import (
    BatchExperiment,
    ParallelExperiment,
    BaseExperiment,
)


class BasicExperiment(BaseExperiment):
    """
    Basic atmoic experiment that mimics the template experiment,
    but uses a pre-prepared transpiled circuit
    """

    def __init__(self, qubits, template_circs, analysis):
        super().__init__(qubits)
        self._template_circs = template_circs
        self.analysis = analysis

    def circuits(self):
        pass

    def _transpiled_circuits(self):
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


def build_whole_backend_experiment(template_experiment, groups):
    """
    Return an experiment that covers all the groups of qubits/edges
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
        parexps.append(ParallelExperiment(exps))

    return BatchExperiment(parexps, backend=template_experiment.backend, flatten_results=True)
