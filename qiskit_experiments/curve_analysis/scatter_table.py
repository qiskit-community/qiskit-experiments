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
from __future__ import annotations

import logging
from collections.abc import Iterator
from typing import Any
from itertools import groupby
from operator import itemgetter

import numpy as np
import pandas as pd

from qiskit.utils import deprecate_func


LOG = logging.getLogger(__name__)


class ScatterTable:
    """A table-like dataset for curve fitting intermediate data.

    Default table columns are defined in the class attribute :attr:`.DEFAULT_COLUMNS`.
    This table cannot be expanded with user-provided column names.
    See attribute documentation for what columns represent.

    This dataset is not thread safe. Do not use the same instance in multiple threads.
    """

    DEFAULT_COLUMNS = [
        "xval",
        "yval",
        "yerr",
        "name",
        "class_id",
        "category",
        "shots",
        "analysis",
    ]

    DEFAULT_DTYPES = [
        "Float64",
        "Float64",
        "Float64",
        "string",
        "Int64",
        "string",
        "Int64",
        "string",
    ]

    def __init__(self):
        """Create new dataset."""
        super().__init__()
        self._lazy_add_rows = []
        self._dump = pd.DataFrame(columns=self.DEFAULT_COLUMNS)

    @property
    def _data(self) -> pd.DataFrame:
        if self._lazy_add_rows:
            # Add data when table element is called.
            # Adding rows in loop is extremely slow in pandas.
            tmp_df = pd.DataFrame(self._lazy_add_rows, columns=self.DEFAULT_COLUMNS)
            tmp_df = self._format_table(tmp_df)
            if len(self._dump) == 0:
                self._dump = tmp_df
            else:
                self._dump = pd.concat([self._dump, tmp_df], ignore_index=True)
            self._lazy_add_rows.clear()
        return self._dump

    @classmethod
    def from_dataframe(cls, data: pd.DataFrame) -> "ScatterTable":
        """Create new dataset with existing dataframe.

        Args:
            data: Data dataframe object.

        Returns:
            A new ScatterTable instance.
        """
        if list(data.columns) != cls.DEFAULT_COLUMNS:
            raise ValueError("Input dataframe columns don't match with the ScatterTable spec.")
        instance = object.__new__(ScatterTable)
        instance._lazy_add_rows = []
        instance._dump = cls._format_table(data)
        return instance

    @property
    def dataframe(self):
        """Dataframe object of data points."""
        return self._data

    @property
    def x(self) -> np.ndarray:
        """X values."""
        return self._data.xval.to_numpy(dtype=float, na_value=np.nan)

    @x.setter
    def x(self, new_values):
        self._data.loc[:, "xval"] = new_values

    @property
    def y(self) -> np.ndarray:
        """Y values."""
        return self._data.yval.to_numpy(dtype=float, na_value=np.nan)

    @y.setter
    def y(self, new_values: np.ndarray):
        self._data.loc[:, "yval"] = new_values

    @property
    def y_err(self) -> np.ndarray:
        """Standard deviation of y values."""
        return self._data.yerr.to_numpy(dtype=float, na_value=np.nan)

    @y_err.setter
    def y_err(self, new_values: np.ndarray):
        self._data.loc[:, "yerr"] = new_values

    @property
    def name(self) -> np.ndarray:
        """Corresponding data name."""
        return self._data.name.to_numpy(dtype=object, na_value=None)

    @name.setter
    def name(self, new_values: np.ndarray):
        self._data.loc[:, "name"] = new_values

    @property
    def class_id(self) -> np.ndarray:
        """Corresponding data UID."""
        return self._data.class_id.to_numpy(dtype=object, na_value=None)

    @class_id.setter
    def class_id(self, new_values: np.ndarray):
        self._data.loc[:, "class_id"] = new_values

    @property
    def category(self) -> np.ndarray:
        """Category of data points."""
        return self._data.category.to_numpy(dtype=object, na_value=None)

    @category.setter
    def category(self, new_values: np.ndarray):
        self._data.loc[:, "category"] = new_values

    @property
    def shots(self) -> np.ndarray:
        """Shot number used to acquire data points."""
        return self._data.shots.to_numpy(dtype=object, na_value=None)

    @shots.setter
    def shots(self, new_values: np.ndarray):
        self._data.loc[:, "shots"] = new_values

    @property
    def analysis(self) -> np.ndarray:
        """Corresponding analysis name."""
        return self._data.analysis.to_numpy(dtype=object, na_value=None)

    @analysis.setter
    def analysis(self, new_values: np.ndarray):
        self._data.loc[:, "analysis"] = new_values

    def filter(
        self,
        kind: int | str | None = None,
        category: str | None = None,
        analysis: str | None = None,
    ) -> ScatterTable:
        """Filter data by class, category, and/or analysis name.

        Args:
            kind: Identifier of the data, either data UID or name.
            category: Name of data category.
            analysis: Name of analysis.

        Returns:
            New ScatterTable object with filtered data.
        """
        filt_data = self._data

        if kind is not None:
            if isinstance(kind, int):
                index = self._data.class_id == kind
            elif isinstance(kind, str):
                index = self._data.name == kind
            else:
                raise ValueError(f"Invalid kind type {type(kind)}. This must be integer or string.")
            filt_data = filt_data.loc[index, :]
        if category is not None:
            index = self._data.category == category
            filt_data = filt_data.loc[index, :]
        if analysis is not None:
            index = self._data.analysis == analysis
            filt_data = filt_data.loc[index, :]
        return ScatterTable.from_dataframe(filt_data)

    def iter_by_class(self) -> Iterator[tuple[int, "ScatterTable"]]:
        """Iterate over subset of data sorted by the data UID.

        Yields:
            Tuple of data UID and subset of ScatterTable.
        """
        ids = self._data.class_id.dropna().sort_values().unique()
        for mid in ids:
            index = self._data.class_id == mid
            yield mid, ScatterTable.from_dataframe(self._data.loc[index, :])

    def iter_groups(
        self,
        *group_by: str,
    ) -> Iterator[tuple[tuple[Any, ...], "ScatterTable"]]:
        """Iterate over the subset sorted by multiple column values.

        Args:
            group_by: Name of column to group by.

        Yields:
            Tuple of keys and subset of ScatterTable.
        """
        try:
            sort_by = itemgetter(*[self.DEFAULT_COLUMNS.index(c) for c in group_by])
        except ValueError as ex:
            raise ValueError(
                f"Specified columns don't exist: {group_by} are not subset of {self.DEFAULT_COLUMNS}."
            ) from ex

        # Use python native groupby method on dataframe ndarray when sorting by multiple columns.
        # This is more performant than pandas groupby implementation.
        for vals, sub_data in groupby(sorted(self._data.values, key=sort_by), key=sort_by):
            tmp_df = pd.DataFrame(list(sub_data), columns=self.DEFAULT_COLUMNS)
            yield vals, ScatterTable.from_dataframe(tmp_df)

    def add_row(
        self,
        name: str | pd.NA = pd.NA,
        class_id: int | pd.NA = pd.NA,
        category: str | pd.NA = pd.NA,
        x: float | pd.NA = pd.NA,
        y: float | pd.NA = pd.NA,
        y_err: float | pd.NA = pd.NA,
        shots: float | pd.NA = pd.NA,
        analysis: str | pd.NA = pd.NA,
    ):
        """Add new data group to the table.

        Data must be the same length.

        Args:
            x: X value.
            y: Y value.
            y_err: Standard deviation of y value.
            shots: Shot number used to acquire this data point.
            name: Name of this data if available.
            class_id: Data UID of if available.
            category: Data category if available.
            analysis: Analysis name if available.
        """
        self._lazy_add_rows.append([x, y, y_err, name, class_id, category, shots, analysis])

    @classmethod
    def _format_table(cls, data: pd.DataFrame) -> pd.DataFrame:
        return (
            data.replace(np.nan, pd.NA)
            .astype(dict(zip(cls.DEFAULT_COLUMNS, cls.DEFAULT_DTYPES)))
            .reset_index(drop=True)
        )

    @property
    @deprecate_func(
        since="0.6",
        additional_msg="Curve data uses dataframe representation. Call .model_id instead.",
        pending=True,
        package_name="qiskit-experiments",
        is_property=True,
    )
    def data_allocation(self) -> np.ndarray:
        """Index of corresponding fit model."""
        return self.class_id

    @property
    @deprecate_func(
        since="0.6",
        additional_msg="No alternative is provided. Use .name with set operation.",
        pending=True,
        package_name="qiskit-experiments",
        is_property=True,
    )
    def labels(self) -> list[str]:
        """List of model names."""
        # Order sensitive
        name_id_tups = self._data.groupby(["name", "class_id"]).groups.keys()
        return [k[0] for k in sorted(name_id_tups, key=lambda k: k[1])]

    @deprecate_func(
        since="0.6",
        additional_msg="Use filter method instead.",
        pending=True,
        package_name="qiskit-experiments",
    )
    def get_subset_of(self, index: str | int) -> "ScatterTable":
        """Filter data by series name or index.

        Args:
            index: Series index of name.

        Returns:
            A subset of data corresponding to a particular series.
        """
        return self.filter(kind=index)

    def __len__(self):
        return len(self._data)

    def __eq__(self, other):
        return self.dataframe.equals(other.dataframe)

    def __json_encode__(self) -> dict[str, Any]:
        return {
            "class": "ScatterTable",
            "data": self._data.to_dict(orient="index"),
        }

    @classmethod
    def __json_decode__(cls, value: dict[str, Any]) -> "ScatterTable":
        if not value.get("class", None) == "ScatterTable":
            raise ValueError("JSON decoded value for ScatterTable is not valid class type.")
        tmp_df = pd.DataFrame.from_dict(value.get("data", {}), orient="index")
        return ScatterTable.from_dataframe(tmp_df)
