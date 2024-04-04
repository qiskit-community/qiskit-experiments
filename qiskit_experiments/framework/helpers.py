# This code is part of Qiskit.
#
# (C) Copyright IBM 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
"""
Helper functions.
"""

from typing import Tuple, Optional

from qiskit_ibm_experiment import IBMExperimentService
from qiskit.providers import Provider
from qiskit_experiments.framework import BaseExperiment, ExperimentData


def load_all(
    experiment_id: str,
    service: Optional[IBMExperimentService] = None,
    provider: Optional[Provider] = None,
    run_analysis: bool = False,
) -> Tuple["BaseExperiment", "ExperimentData"]:
    """Load a saved experiment and its experiment data from a database service.

    Args:
        experiment_id: The experiment ID.
        service: the database service.
        provider: an IBMProvider required for loading the experiment data and
            can be used to initialize the service. When using
            :external+qiskit_ibm_runtime:doc:`qiskit-ibm-runtime <index>`,
            this is the :class:`~qiskit_ibm_runtime.QiskitRuntimeService` and should
            not be confused with the experiment database service
            :meth:`qiskit_ibm_experiment.IBMExperimentService`.
        run_analysis: Whether to run the analysis associated with the experiment and return a new
            experiment data container. Defaults to ``False``.

    Returns:
        A tuple consisting of the reconstructed experiment and experiment data.

    """
    load_exp = BaseExperiment.load(experiment_id, provider=provider, service=service)
    load_expdata = ExperimentData.load(experiment_id, provider=provider, service=service)
    if not run_analysis:
        return (load_exp, load_expdata)
    return (load_exp, load_exp.analysis.run(load_expdata))
