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
Utility for checking equality of data of Qiskit Experiments class which doesn't
officially implement the equality dunder method.
"""

import dataclasses
from typing import Any, List, Union

import numpy as np
import uncertainties
from lmfit import Model
from multimethod import multimethod
from qiskit_experiments.curve_analysis.curve_data import CurveFitResult
from qiskit_experiments.data_processing import DataAction, DataProcessor
from qiskit_experiments.database_service.utils import (
    ThreadSafeList,
    ThreadSafeOrderedDict,
)
from qiskit_experiments.framework import (
    ExperimentData,
    BaseExperiment,
    BaseAnalysis,
    AnalysisResult,
)
from qiskit_experiments.visualization import BaseDrawer


@multimethod
def is_equivalent(
    data1: object,
    data2: object,
):
    """Equality check finally falls into this function."""
    if data1 is None and data2 is None:
        return True
    if dataclasses.is_dataclass(data1) and dataclasses.is_dataclass(data2):
        # Avoid dataclass asdict. This copies dictionary but experiment config dataclass
        # may include type information which cannot be copied.
        return is_equivalent(
            data1.__dict__,
            data2.__dict__,
        )
    evaluated = data1 == data2
    if not isinstance(evaluated, bool):
        # When either one of input is numpy array type, it may broadcast equality check
        # and return ndarray of dtype=bool. e.g. np.array([]) == 123
        # The input values should not be equal in this case.
        return False

    # Return the outcome of native equivalence check.
    return evaluated


@is_equivalent.register
def _check_dicts(
    data1: Union[dict, ThreadSafeOrderedDict],
    data2: Union[dict, ThreadSafeOrderedDict],
):
    """Check equality of dictionary which may involve Qiskit Experiments classes."""
    if set(data1) != set(data2):
        return False
    return all(is_equivalent(data1[k], data2[k]) for k in data1.keys())


@is_equivalent.register
def _check_floats(
    data1: Union[float, np.floating],
    data2: Union[float, np.floating],
):
    """Check equality of float.

    Both python built-in float and numpy floating subtypes can be compared.
    This function also supports comparison of float("nan").
    """
    if np.isnan(data1) and np.isnan(data2):
        # Special case
        return True
    return float(data1) == float(data2)


@is_equivalent.register
def _check_integer(
    data1: Union[int, np.integer],
    data2: Union[int, np.integer],
):
    """Check equality of integer.

    Both python built-in integer and numpy integer subtypes can be compared.
    """
    return int(data1) == int(data2)


@is_equivalent.register
def _check_sequences(
    data1: Union[list, tuple, np.ndarray, ThreadSafeList],
    data2: Union[list, tuple, np.ndarray, ThreadSafeList],
):
    """Check equality of sequence."""
    if len(data1) != len(data2):
        return False
    return all(is_equivalent(e1, e2) for e1, e2 in zip(data1, data2))


@is_equivalent.register
def _check_unordered_sequences(
    data1: set,
    data2: set,
):
    """Check equality of sequence after sorting."""
    if len(data1) != len(data2):
        return False
    return all(is_equivalent(e1, e2) for e1, e2 in zip(sorted(data1), sorted(data2)))


@is_equivalent.register
def _check_ufloats(
    data1: uncertainties.UFloat,
    data2: uncertainties.UFloat,
):
    """Check equality of UFloat instance. Correlations are ignored."""
    return data1.n == data2.n and data1.s == data2.s


@is_equivalent.register
def _check_lmfit_models(
    data1: Model,
    data2: Model,
):
    """Check equality of LMFIT model."""
    return is_equivalent(data1.dumps(), data2.dumps())


@is_equivalent.register
def _check_dataprocessing_instances(
    data1: Union[DataAction, DataProcessor],
    data2: Union[DataAction, DataProcessor],
):
    """Check equality of classes in the data_processing module."""
    return repr(data1) == repr(data2)


@is_equivalent.register
def _check_curvefit_results(
    data1: CurveFitResult,
    data2: CurveFitResult,
):
    """Check equality of curve fit result."""
    return _check_all_attributes(
        attrs=[
            "method",
            "model_repr",
            "success",
            "nfev",
            "message",
            "dof",
            "init_params",
            "chisq",
            "reduced_chisq",
            "aic",
            "bic",
            "params",
            "var_names",
            "x_data",
            "y_data",
            "covar",
        ],
        data1=data1,
        data2=data2,
    )


@is_equivalent.register
def _check_service_analysis_results(
    data1: AnalysisResult,
    data2: AnalysisResult,
):
    """Check equality of AnalysisResult class which is payload for experiment service."""
    return _check_all_attributes(
        attrs=[
            "name",
            "value",
            "extra",
            "device_components",
            "result_id",
            "experiment_id",
            "chisq",
            "quality",
            "verified",
            "tags",
            "auto_save",
            "source",
        ],
        data1=data1,
        data2=data2,
    )


@is_equivalent.register
def _check_configurable_classes(
    data1: Union[BaseExperiment, BaseAnalysis, BaseDrawer],
    data2: Union[BaseExperiment, BaseAnalysis, BaseDrawer],
):
    """Check equality of Qiskit Experiments class with config method."""
    return is_equivalent(data1.config(), data2.config())


@is_equivalent.register
def _check_experiment_data(
    data1: ExperimentData,
    data2: ExperimentData,
):
    """Check equality of ExperimentData."""
    attributes_equiv = _check_all_attributes(
        attrs=[
            "experiment_id",
            "experiment_type",
            "parent_id",
            "tags",
            "job_ids",
            "figure_names",
            "share_level",
            "metadata",
        ],
        data1=data1,
        data2=data2,
    )
    data_equiv = is_equivalent(
        data1.data(),
        data2.data(),
    )
    analysis_results_equiv = is_equivalent(
        data1._analysis_results,
        data2._analysis_results,
    )
    child_equiv = is_equivalent(
        data1.child_data(),
        data2.child_data(),
    )
    return all([attributes_equiv, data_equiv, analysis_results_equiv, child_equiv])


def _check_all_attributes(
    attrs: List[str],
    data1: Any,
    data2: Any,
):
    """Helper function to check all attributes."""
    return all(is_equivalent(getattr(data1, att), getattr(data2, att)) for att in attrs)
