# This code is part of Qiskit.
#
# (C) Copyright IBM 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
"""Table representation of the x, y data for curve fitting."""

import logging
from typing import List, Sequence, Dict, Any, Union

import numpy as np
import pandas as pd

from qiskit.utils import deprecate_func

from qiskit_experiments.framework.table_mixin import DefaultColumnsMixIn


LOG = logging.getLogger(__name__)


class ScatterTable(pd.DataFrame, DefaultColumnsMixIn):
    """A table to store x and y data with metadata associated with the data point.

    This class is implemented upon the pandas dataframe.
    See `pandas dataframe documentation <https://pandas.pydata.org/docs/index.html>`_
    for the base class API documentation.

    A single ``ScatterTable`` object can contain different kind of intermediate data
    generated through the curve fitting, which are categorized by the fit model.
    When an experiment has sub-data for ``model_abc``, the formatted x, y, and y-error
    array data may be obtained from the original table object as follows:

    .. code-block::python

        formatted = table.filter(like="formatted", axis="index")
        abc_data = formatted[formatted.model_name == "model_abc"]
        x, y, e = abc_data.xval.to_numpy(), abc_data.yval.to_numpy(), abc_data.yerr.to_numpy()

    """

    @classmethod
    def _default_columns(cls) -> List[str]:
        return [
            "xval",
            "yval",
            "yerr",
            "model_name",
            "model_id",
            "shots",
        ]

    @deprecate_func(
        since="0.6",
        additional_msg="Curve data uses dataframe representation. Use dataframe filtering method.",
        pending=True,
        package_name="qiskit-experiments",
    )
    def get_subset_of(self, index: Union[str, int]) -> "ScatterTable":
        """Filter data by series name or index.

        Args:
            index: Series index of name.

        Returns:
            A subset of data corresponding to a particular series.
        """
        if isinstance(index, int):
            index = self.labels[index]
        return self[self.model_name == index]

    @property
    @deprecate_func(
        since="0.6",
        additional_msg="Curve data uses dataframe representation. Call .xval.to_numpy() instead.",
        pending=True,
        package_name="qiskit-experiments",
        is_property=True,
    )
    def x(self) -> np.ndarray:
        """X values."""
        return self.xval.to_numpy()

    @property
    @deprecate_func(
        since="0.6",
        additional_msg="Curve data uses dataframe representation. Call .yval.to_numpy() instead.",
        pending=True,
        package_name="qiskit-experiments",
        is_property=True,
    )
    def y(self) -> np.ndarray:
        """Y values."""
        return self.yval.to_numpy()

    @property
    @deprecate_func(
        since="0.6",
        additional_msg="Curve data uses dataframe representation. Call .yerr.to_numpy() instead.",
        pending=True,
        package_name="qiskit-experiments",
        is_property=True,
    )
    def y_err(self) -> np.ndarray:
        """Standard deviation of y values."""
        return self.yerr.to_numpy()

    @property
    @deprecate_func(
        since="0.6",
        additional_msg="Curve data uses dataframe representation. Call .shots.to_numpy() instead.",
        pending=True,
        package_name="qiskit-experiments",
        is_property=True,
    )
    def shots(self):
        """Shot number of data points."""
        return self.shots.to_numpy()

    @property
    @deprecate_func(
        since="0.6",
        additional_msg="Curve data uses dataframe representation. Call .model_id.to_numpy() instead.",
        pending=True,
        package_name="qiskit-experiments",
        is_property=True,
    )
    def data_allocation(self) -> np.ndarray:
        """Index of corresponding fit model."""
        # pylint: disable=no-member
        return self.model_id.to_numpy()

    @property
    @deprecate_func(
        since="0.6",
        additional_msg="Curve data uses dataframe representation. Labels are a part of table.",
        pending=True,
        package_name="qiskit-experiments",
        is_property=True,
    )
    def labels(self) -> List[str]:
        """List of model names."""
        # Order sensitive
        name_id_tups = self.groupby(["model_name", "model_id"]).groups.keys()
        return [k[0] for k in sorted(name_id_tups, key=lambda k: k[1])]

    def append_list_values(
        self,
        other: Sequence,
        prefix: str,
    ) -> "ScatterTable":
        """Add another list of dataframe values to this dataframe.

        Args:
            other: List of dataframe values to be added.
            prefix: Prefix of row labels of the added values.

        Returns:
            New scatter table instance including both self and added data.
        """
        other_index = [f"{prefix}-{i:04d}" for i in range(len(other))]
        return ScatterTable(
            data=[*self.values, *other],
            columns=self.columns,
            index=[*self.index, *other_index],
        )

    def __json_encode__(self) -> Dict[str, Any]:
        return {
            "class": "ScatterTable",
            "data": self.to_dict(orient="index"),
        }

    @classmethod
    def __json_decode__(cls, value: Dict[str, Any]) -> "ScatterTable":
        if not value.get("class", None) == "ScatterTable":
            raise ValueError("JSON decoded value for ScatterTable is not valid class type.")

        instance = cls.from_dict(
            data=value.get("data", {}),
            orient="index",
        ).replace({np.nan: None})
        return instance

    @property
    def _constructor(self):
        # https://pandas.pydata.org/pandas-docs/stable/development/extending.html
        return ScatterTable
