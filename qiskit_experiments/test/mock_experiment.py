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
Standard RB analysis class.
"""

from qiskit_experiments.base_experiment import BaseExperiment
from qiskit_experiments.experiment_data import ExperimentData
from qiskit_experiments.analysis import CurveAnalysis, SeriesDef
from typing import Tuple, Optional
import numpy as np


class FakeExperiment(BaseExperiment):
    """A fake experiment class."""

    def __init__(self, qubits=(0,)):
        super().__init__(qubits=qubits, experiment_type="fake_experiment")

    def circuits(self, backend=None, **circuit_options):
        return []


def curve_model_based_level2_probability_experiment(
        experiment: BaseExperiment,
        xvals: np.ndarray,
        target_params: np.ndarray,
        outcome_labels: Tuple[str, str],
        shots: Optional[int] = 1024
) -> ExperimentData:
    """Simulate fake experiment data with curve analysis fit model.

    Args:
        experiment: Target experiment class.
        xvals: Values to scan.
        target_params: Parameters used to generate sampling data with the fit function.
        outcome_labels: Two strings tuple of out come label in count dictionary.
            The first and second label should be assigned to success and failure, respectively.
        shots: Number of shot to generate count data.

    Returns:
        A fake experiment data.
    """
    analysis = experiment.__analysis_class__

    if isinstance(analysis, CurveAnalysis):
        raise Exception("The attached analysis is not CurveAnalysis subclass. "
                        "This function cannot simulate output data.")

    x_key = analysis.__x_key__
    series = analysis.__series__
    fit_funcs = analysis.__fit_funcs__
    param_names = analysis.__param_names__

    if series is None:
        series = [
            SeriesDef(
                name='default',
                param_names=[f"p{idx}" for idx in range(len(target_params))],
                fit_func_index=0,
                filter_kwargs=dict()
            )
        ]

    data = []
    for curve_properties in series:
        fit_func = fit_funcs[curve_properties.fit_func_index]
        params = []
        for param_name in curve_properties.param_names:
            param_index = param_names.index(param_name)
            params.append(param_index)
        y_values = fit_func(xvals, *params)
        counts = np.asarray(y_values * shots, dtype=int)

        for xi, count in zip(xvals, counts):
            metadata = {
                x_key: xi,
                "qubits": experiment._physical_qubits,
                "experiment_type": experiment._type
            }
            metadata.update(**series.filter_kwargs)

            data.append(
                {
                    "counts": {outcome_labels[0]: shots - count, outcome_labels[1]: count},
                    "metadata": metadata
                }
            )

    expdata = ExperimentData(experiment=experiment)
    for datum in data:
        expdata.add_data(datum)

    return expdata
