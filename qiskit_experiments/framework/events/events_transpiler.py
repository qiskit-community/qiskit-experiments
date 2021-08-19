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
Transpile events.
"""

import copy

from qiskit import transpile
from qiskit.test.mock import FakeBackend

from qiskit_experiments.framework import ParallelExperiment


def count_transpiled_ops(experiment, circuits, **kwargs):

    def get_metadata(circuit):
        if circuit.metadata["experiment_type"] == experiment.experiment_type:
            return circuit.metadata
        if circuit.metadata["experiment_type"] == ParallelExperiment.__name__:
            for meta in circuit.metadata["composite_metadata"]:
                if meta["physical_qubits"] == experiment.physical_qubits:
                    return meta
        return dict()

    for circ in circuits:
        meta = get_metadata(circ)
        if meta:
            qubits = range(len(circ.qubits))
            c_count_ops = {}
            for instr, qargs, _ in circ:
                instr_qubits = []
                skip_instr = False
                for qubit in qargs:
                    qubit_index = circ.qubits.index(qubit)
                    if qubit_index not in qubits:
                        skip_instr = True
                    instr_qubits.append(qubit_index)
                if not skip_instr:
                    instr_qubits = tuple(instr_qubits)
                    c_count_ops[(instr_qubits, instr.name)] = (
                            c_count_ops.get((instr_qubits, instr.name), 0) + 1
                    )
            circuit_length = meta["xval"]
            count_ops = [(key, (value, circuit_length)) for key, value in c_count_ops.items()]
            meta.update({"count_ops": count_ops})


def set_scheduling_contraints(expeirment, backend, **kwargs):
    if not backend.configuration().simulator and not isinstance(backend, FakeBackend):
        timing_constraints = getattr(
            expeirment.transpile_options.__dict__,
            "timing_constraints",
            {}
        )
        timing_constraints["acquire_alignment"] = getattr(
            timing_constraints, "acquire_alignment", 16
        )
        scheduling_method = getattr(
            expeirment.transpile_options.__dict__, "scheduling_method", "alap"
        )
        expeirment.set_transpile_options(
            timing_constraints=timing_constraints, scheduling_method=scheduling_method
        )


def transpile_circuits(experiment, backend, **kwargs):
    options = copy.copy(experiment.transpile_options.__dict__)
    options["initial_layout"] = list(experiment.physical_qubits)
    circuits = transpile(experiment.circuits(backend), backend, **options)

    return {"circuits": circuits}
