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
Pre processing events.
"""
import copy

from qiskit_experiments.framework import ExperimentData
from qiskit.exceptions import QiskitError


def initialize_experiment_data(experiment, backend, expeirment_data, **kwargs):
    if expeirment_data is not None:
        init_data = experiment.__experiment_data__(experiment=experiment, backend=backend)
    else:
        # Validate experiment is compatible with existing data
        if not isinstance(expeirment_data, ExperimentData):
            raise QiskitError("Input `experiment_data` is not a valid ExperimentData.")
        if expeirment_data.experiment_type != experiment.experiment_type:
            raise QiskitError("Existing ExperimentData contains data from a different experiment.")
        if expeirment_data.metadata.get("physical_qubits") != list(experiment.physical_qubits):
            raise QiskitError(
                "Existing ExperimentData contains data for a different set of physical qubits."
            )
        init_data = expeirment_data._copy_metadata()

    return {"expeirment_data": init_data}


def update_run_options(experiment, run_options, **kwargs):
    options = copy.copy(experiment.run_options).__dict__
    options.update(**run_options)

    return {"run_options": options}
