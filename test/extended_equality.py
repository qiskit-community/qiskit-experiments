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

# pylint: disable=unused-argument

"""
Utility for checking equality of data of Qiskit Experiments class which doesn't
officially implement the equality dunder method.
"""

import dataclasses
from typing import Any, List, Union

import numpy as np
import pandas as pd
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
    AnalysisResultTable,
    ArtifactData,
)
from qiskit_experiments.visualization import BaseDrawer


def is_equivalent(
    data1: Any,
    data2: Any,
    *,
    strict_type: bool = True,
    numerical_precision: float = 1e-8,
) -> bool:
    """Check if two input data are equivalent.

    This function is used for custom equivalence evaluation only for unittest purpose.
    Some third party class may not preserve equivalence after JSON round-trip with
    Qiskit Experiments JSON Encoder/Decoder, or some Qiskit Experiments class doesn't
    define the equality dunder method intentionally.

    Args:
        data1: First data to compare.
        data2: Second data to compare.
        strict_type: Set True to enforce type check before comparison. Note that serialization
            and deserialization round-trip may not preserve data type.
            If the data type doesn't matter and only behavioral equivalence is considered,
            e.g. iterator with the same element; tuple vs list,
            you can turn off this flag to relax the constraint for data type.
        numerical_precision: Tolerance of difference between two real numbers.

    Returns:
        True when two objects are equivalent.
    """
    if strict_type and type(data1) is not type(data2):
        return False
    evaluated = _is_equivalent_dispatcher(
        data1,
        data2,
        strict_type=strict_type,
        numerical_precision=numerical_precision,
    )
    if not isinstance(evaluated, (bool, np.bool_)):
        # When either one of input is numpy array type, it may broadcast equality check
        # and return ndarray of dtype=bool. e.g. np.array([]) == 123
        # The input values should not be equal in this case.
        return False
    return evaluated


@multimethod
def _is_equivalent_dispatcher(
    data1: object,
    data2: object,
    **kwargs,
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
            **kwargs,
        )
    return data1 == data2


@_is_equivalent_dispatcher.register
def _check_dicts(
    data1: Union[dict, ThreadSafeOrderedDict],
    data2: Union[dict, ThreadSafeOrderedDict],
    **kwargs,
):
    """Check equality of dictionary which may involve Qiskit Experiments classes."""
    if set(data1) != set(data2):
        return False
    return all(is_equivalent(data1[k], data2[k], **kwargs) for k in data1.keys())


@_is_equivalent_dispatcher.register
def _check_floats(
    data1: Union[float, np.floating],
    data2: Union[float, np.floating],
    **kwargs,
):
    """Check equality of float.

    Both python built-in float and numpy floating subtypes can be compared.
    This function also supports comparison of float("nan").
    """
    if np.isnan(data1) and np.isnan(data2):
        # Special case
        return True

    precision = kwargs.get("numerical_precision", 0.0)
    if precision == 0.0:
        return float(data1) == float(data2)
    return np.isclose(np.abs(data1 - data2), 0.0, atol=precision)


@_is_equivalent_dispatcher.register
def _check_integer(
    data1: Union[int, np.integer],
    data2: Union[int, np.integer],
    **kwargs,
):
    """Check equality of integer.

    Both python built-in integer and numpy integer subtypes can be compared.
    """
    return int(data1) == int(data2)


@_is_equivalent_dispatcher.register
def _check_sequences(
    data1: Union[list, tuple, np.ndarray, ThreadSafeList],
    data2: Union[list, tuple, np.ndarray, ThreadSafeList],
    **kwargs,
):
    """Check equality of sequence."""
    if len(data1) != len(data2):
        return False
    return all(is_equivalent(e1, e2, **kwargs) for e1, e2 in zip(data1, data2))


@_is_equivalent_dispatcher.register
def _check_unordered_sequences(
    data1: set,
    data2: set,
    **kwargs,
):
    """Check equality of sequence after sorting."""
    if len(data1) != len(data2):
        return False
    return all(is_equivalent(e1, e2, **kwargs) for e1, e2 in zip(sorted(data1), sorted(data2)))


@_is_equivalent_dispatcher.register
def _check_ufloats(
    data1: uncertainties.UFloat,
    data2: uncertainties.UFloat,
    **kwargs,
):
    """Check equality of UFloat instance. Correlations are ignored."""
    return is_equivalent(data1.n, data2.n, **kwargs) and is_equivalent(data1.s, data2.s, **kwargs)


@_is_equivalent_dispatcher.register
def _check_lmfit_models(
    data1: Model,
    data2: Model,
    **kwargs,
):
    """Check equality of LMFIT model."""
    return is_equivalent(data1.dumps(), data2.dumps(), **kwargs)


@_is_equivalent_dispatcher.register
def _check_dataprocessing_instances(
    data1: Union[DataAction, DataProcessor],
    data2: Union[DataAction, DataProcessor],
    **kwargs,
):
    """Check equality of classes in the data_processing module."""
    return repr(data1) == repr(data2)


@_is_equivalent_dispatcher.register
def _check_curvefit_results(
    data1: CurveFitResult,
    data2: CurveFitResult,
    **kwargs,
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
        **kwargs,
    )


@_is_equivalent_dispatcher.register
def _check_service_analysis_results(
    data1: AnalysisResult,
    data2: AnalysisResult,
    **kwargs,
):
    """Check equality of AnalysisResult class which is payload for experiment service."""
    return _check_all_attributes(
        attrs=[
            "name",
            "value",
            "extra",
            "device_components",
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
        **kwargs,
    )


@_is_equivalent_dispatcher.register
def _check_artifact_data(
    data1: ArtifactData,
    data2: ArtifactData,
    **kwargs,
):
    """Check equality of the ArtifactData class."""
    return _check_all_attributes(
        attrs=[
            "name",
            "data",
            "device_components",
            "experiment_id",
            "experiment",
        ],
        data1=data1,
        data2=data2,
        **kwargs,
    )


@_is_equivalent_dispatcher.register
def _check_configurable_classes(
    data1: Union[BaseExperiment, BaseAnalysis, BaseDrawer],
    data2: Union[BaseExperiment, BaseAnalysis, BaseDrawer],
    **kwargs,
):
    """Check equality of Qiskit Experiments class with config method."""
    return is_equivalent(data1.config(), data2.config(), **kwargs)


@_is_equivalent_dispatcher.register
def _check_dataframes(
    data1: pd.DataFrame,
    data2: pd.DataFrame,
    **kwargs,
):
    """Check equality of data frame which may involve Qiskit Experiments class value."""
    return is_equivalent(
        data1.to_dict(orient="index"),
        data2.to_dict(orient="index"),
        **kwargs,
    )


@_is_equivalent_dispatcher.register
def _check_result_table(
    data1: AnalysisResultTable,
    data2: AnalysisResultTable,
    **kwargs,
):
    """Check equality of data frame which may involve Qiskit Experiments class value."""
    table1 = data1.dataframe.to_dict(orient="index")
    table2 = data2.dataframe.to_dict(orient="index")
    for table in (table1, table2):
        for result in table.values():
            result.pop("created_time")
            # Must ignore result ids because they are internally generated with
            # random values by the ExperimentData wrapping object.
            result.pop("result_id")
    # Keys of the dict are based on the result ids so they must be ignored
    # as well. Try to sort entries so equivalent entries will be in the same
    # order.
    table1 = sorted(
        table1.values(),
        key=lambda x: (
            x["name"],
            () if x["components"] is None else tuple(repr(d) for d in x["components"]),
            x["value"],
        ),
    )
    table2 = sorted(
        table2.values(),
        key=lambda x: (
            x["name"],
            () if x["components"] is None else tuple(repr(d) for d in x["components"]),
            x["value"],
        ),
    )
    return is_equivalent(
        table1,
        table2,
        **kwargs,
    )


@_is_equivalent_dispatcher.register
def _check_experiment_data(
    data1: ExperimentData,
    data2: ExperimentData,
    **kwargs,
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
        **kwargs,
    )
    data_equiv = is_equivalent(
        data1.data(),
        data2.data(),
        **kwargs,
    )
    analysis_results_equiv = is_equivalent(
        data1._analysis_results,
        data2._analysis_results,
        **kwargs,
    )
    child_equiv = is_equivalent(
        data1.child_data(),
        data2.child_data(),
        **kwargs,
    )
    artifact_equiv = is_equivalent(
        data1.artifacts(),
        data2.artifacts(),
        **kwargs,
    )

    return all([attributes_equiv, data_equiv, analysis_results_equiv, child_equiv, artifact_equiv])


def _check_all_attributes(
    attrs: List[str],
    data1: Any,
    data2: Any,
    **kwargs,
):
    """Helper function to check all attributes."""
    test = {}
    for att in attrs:
        test[att] = is_equivalent(getattr(data1, att), getattr(data2, att), **kwargs)

    return all(is_equivalent(getattr(data1, att), getattr(data2, att), **kwargs) for att in attrs)
