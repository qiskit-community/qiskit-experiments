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
import warnings
from collections.abc import Iterator
from typing import Any
from functools import reduce
from itertools import product

import numpy as np
import pandas as pd


LOG = logging.getLogger(__name__)


class ScatterTable:
    """A table-like dataset for the intermediate data used for curve fitting.

    Default table columns are defined in the class attribute :attr:`.COLUMNS`.
    This table cannot be expanded with user-provided column names.

    In a standard :class:`.CurveAnalysis` subclass, a ScatterTable instance may be
    stored in the :class:`.ExperimentData` as an artifact.
    Users can retrieve the table data at a later time to rerun a fitting with a homemade program
    or with different fit options, or to visualize the curves in a preferred format.
    This table dataset is designed to seamlessly provide such information
    that an experimentalist may want to reuse for a custom workflow.

    .. note::

        This dataset is not thread safe. Do not use the same instance in multiple threads.

    See the tutorial of :ref:`data_management_with_scatter_table` for the
    role of each table column and how values are typically provided.

    """

    COLUMNS = [
        "xval",
        "yval",
        "yerr",
        "series_name",
        "series_id",
        "category",
        "shots",
        "analysis",
    ]

    DTYPES = [
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
        self._lazy_add_rows = []
        self._dump = pd.DataFrame(columns=self.COLUMNS)

    @classmethod
    def from_dataframe(
        cls,
        data: pd.DataFrame,
    ) -> "ScatterTable":
        """Create new dataset with existing dataframe.

        Args:
            data: Data dataframe object.

        Returns:
            A new ScatterTable instance.
        """
        if list(data.columns) != cls.COLUMNS:
            raise ValueError("Input dataframe columns don't match with the ScatterTable spec.")
        format_data = cls._format_table(data)
        return cls._create_new_instance(format_data)

    @classmethod
    def _create_new_instance(
        cls,
        data: pd.DataFrame,
    ) -> "ScatterTable":
        # A shortcut for creating instance.
        # This bypasses data formatting and column compatibility check.
        # User who calls this method must guarantee the quality of the input data.
        instance = object.__new__(cls)
        instance._lazy_add_rows = []
        instance._dump = data
        return instance

    @property
    def dataframe(self):
        """Dataframe object of data points."""
        if self._lazy_add_rows:
            # Add data when table element is called.
            # Adding rows in loop is extremely slow in pandas.
            tmp_df = pd.DataFrame(self._lazy_add_rows, columns=self.COLUMNS)
            tmp_df = self._format_table(tmp_df)
            if len(self._dump) == 0:
                self._dump = tmp_df
            else:
                self._dump = pd.concat([self._dump, tmp_df], ignore_index=True)
            self._lazy_add_rows.clear()
        return self._dump

    @property
    def x(self) -> np.ndarray:
        """X values."""
        # For backward compatibility with CurveData.x
        return self.dataframe.xval.to_numpy(dtype=float, na_value=np.nan)

    @x.setter
    def x(self, new_values):
        self.dataframe.loc[:, "xval"] = new_values

    def xvals(
        self,
        series: int | str | None = None,
        category: str | None = None,
        analysis: str | None = None,
        check_unique: bool = True,
    ) -> np.ndarray:
        """Get subset of X values.

        A convenient shortcut for getting X data with filtering.

        Args:
            series: Identifier of the data series, either integer series index or name.
            category: Name of data category.
            analysis: Name of analysis.
            check_unique: Set True to check if multiple series are contained.
                When multiple series are contained, it raises a user warning.

        Returns:
            Numpy array of X values.
        """
        sub_table = self.filter(series, category, analysis)
        if check_unique:
            sub_table._warn_composite_data()
        return sub_table.x

    @property
    def y(self) -> np.ndarray:
        """Y values."""
        # For backward compatibility with CurveData.y
        return self.dataframe.yval.to_numpy(dtype=float, na_value=np.nan)

    @y.setter
    def y(self, new_values: np.ndarray):
        self.dataframe.loc[:, "yval"] = new_values

    def yvals(
        self,
        series: int | str | None = None,
        category: str | None = None,
        analysis: str | None = None,
        check_unique: bool = True,
    ) -> np.ndarray:
        """Get subset of Y values.

        A convenient shortcut for getting Y data with filtering.

        Args:
            series: Identifier of the data series, either integer series index or name.
            category: Name of data category.
            analysis: Name of analysis.
            check_unique: Set True to check if multiple series are contained.
                When multiple series are contained, it raises a user warning.

        Returns:
            Numpy array of Y values.
        """
        sub_table = self.filter(series, category, analysis)
        if check_unique:
            sub_table._warn_composite_data()
        return sub_table.y

    @property
    def y_err(self) -> np.ndarray:
        """Standard deviation of Y values."""
        # For backward compatibility with CurveData.y_err
        return self.dataframe.yerr.to_numpy(dtype=float, na_value=np.nan)

    @y_err.setter
    def y_err(self, new_values: np.ndarray):
        self.dataframe.loc[:, "yerr"] = new_values

    def yerrs(
        self,
        series: int | str | None = None,
        category: str | None = None,
        analysis: str | None = None,
        check_unique: bool = True,
    ) -> np.ndarray:
        """Get subset of standard deviation of Y values.

        A convenient shortcut for getting Y error data with filtering.

        Args:
            series: Identifier of the data series, either integer series index or name.
            category: Name of data category.
            analysis: Name of analysis.
            check_unique: Set True to check if multiple series are contained.
                When multiple series are contained, it raises a user warning.

        Returns:
            Numpy array of Y error values.
        """
        sub_table = self.filter(series, category, analysis)
        if check_unique:
            sub_table._warn_composite_data()
        return sub_table.y_err

    @property
    def series_name(self) -> np.ndarray:
        """Corresponding data name for each data point."""
        return self.dataframe.series_name.to_numpy(dtype=object, na_value=None)

    @series_name.setter
    def series_name(self, new_values: np.ndarray):
        self.dataframe.loc[:, "series_name"] = new_values

    @property
    def series_id(self) -> np.ndarray:
        """Corresponding data UID for each data point."""
        return self.dataframe.series_id.to_numpy(dtype=object, na_value=None)

    @series_id.setter
    def series_id(self, new_values: np.ndarray):
        self.dataframe.loc[:, "series_id"] = new_values

    @property
    def category(self) -> np.ndarray:
        """Array of categories of the data points."""
        return self.dataframe.category.to_numpy(dtype=object, na_value=None)

    @category.setter
    def category(self, new_values: np.ndarray):
        self.dataframe.loc[:, "category"] = new_values

    @property
    def shots(self) -> np.ndarray:
        """Shot number used to acquire each data point."""
        return self.dataframe.shots.to_numpy(dtype=object, na_value=np.nan)

    @shots.setter
    def shots(self, new_values: np.ndarray):
        self.dataframe.loc[:, "shots"] = new_values

    @property
    def analysis(self) -> np.ndarray:
        """Corresponding analysis name for each data point."""
        return self.dataframe.analysis.to_numpy(dtype=object, na_value=None)

    @analysis.setter
    def analysis(self, new_values: np.ndarray):
        self.dataframe.loc[:, "analysis"] = new_values

    def filter(
        self,
        series: int | str | None = None,
        category: str | None = None,
        analysis: str | None = None,
    ) -> ScatterTable:
        """Filter data by series, category, and/or analysis name.

        Args:
            series: Identifier of the data series, either integer series index or name.
            category: Name of data category.
            analysis: Name of analysis.

        Returns:
            New ScatterTable object with filtered data.
        """
        filt_data = self.dataframe

        if series is not None:
            if isinstance(series, int):
                index = filt_data.series_id == series
            elif isinstance(series, str):
                index = filt_data.series_name == series
            else:
                raise ValueError(
                    f"Invalid series identifier {series}. This must be integer or string."
                )
            filt_data = filt_data.loc[index, :]
        if category is not None:
            index = filt_data.category == category
            filt_data = filt_data.loc[index, :]
        if analysis is not None:
            index = filt_data.analysis == analysis
            filt_data = filt_data.loc[index, :]
        return ScatterTable._create_new_instance(filt_data)

    def iter_by_series_id(self) -> Iterator[tuple[int, "ScatterTable"]]:
        """Iterate over subset of data sorted by the data series index.

        Yields:
            Tuple of data series index and subset of ScatterTable.
        """
        id_values = self.dataframe.series_id
        for did in id_values.dropna().sort_values().unique():
            yield did, ScatterTable._create_new_instance(self.dataframe.loc[id_values == did, :])

    def iter_groups(
        self,
        *group_by: str,
    ) -> Iterator[tuple[tuple[Any, ...], "ScatterTable"]]:
        """Iterate over the subset sorted by multiple column values.

        Args:
            group_by: Names of columns to group by.

        Yields:
            Tuple of values for the grouped columns and the corresponding subset of the scatter table.
        """
        out = self.dataframe
        try:
            values_iter = product(*[out.get(col).unique() for col in group_by])
        except AttributeError as ex:
            raise ValueError(
                f"Specified columns don't exist: {group_by} is not a subset of {self.COLUMNS}."
            ) from ex

        for values in sorted(values_iter):
            each_matched = [out.get(c) == v for c, v in zip(group_by, values)]
            all_matched = reduce(lambda x, y: x & y, each_matched)
            if not any(all_matched):
                continue
            yield values, ScatterTable._create_new_instance(out.loc[all_matched, :])

    def add_row(
        self,
        xval: float | pd.NA = pd.NA,
        yval: float | pd.NA = pd.NA,
        yerr: float | pd.NA = pd.NA,
        series_name: str | pd.NA = pd.NA,
        series_id: int | pd.NA = pd.NA,
        category: str | pd.NA = pd.NA,
        shots: float | pd.NA = pd.NA,
        analysis: str | pd.NA = pd.NA,
    ):
        """Add new data point to the table.

        Data must be the same length.

        Args:
            xval: X value.
            yval: Y value.
            yerr: Standard deviation of y value.
            series_name: Name of this data series if available.
            series_id: Index of this data series if available.
            category: Data category if available.
            shots: Shot number used to acquire this data point.
            analysis: Analysis name if available.
        """
        self._lazy_add_rows.append(
            [xval, yval, yerr, series_name, series_id, category, shots, analysis]
        )

    @classmethod
    def _format_table(cls, data: pd.DataFrame) -> pd.DataFrame:
        return (
            data.replace(np.nan, pd.NA)
            .astype(dict(zip(cls.COLUMNS, cls.DTYPES)))
            .reset_index(drop=True)
        )

    def _warn_composite_data(self):
        if len(self.dataframe.series_name.unique()) > 1:
            warnings.warn(
                "Table data contains multiple data series. "
                "You may want to filter the data by a specific series_id or series_name.",
                UserWarning,
            )
        if len(self.dataframe.category.unique()) > 1:
            warnings.warn(
                "Table data contains multiple categories. "
                "You may want to filter the data by a specific category name.",
                UserWarning,
            )
        if len(self.dataframe.analysis.unique()) > 1:
            warnings.warn(
                "Table data contains multiple datasets from different component analyses. "
                "You may want to filter the data by a specific analysis name.",
                UserWarning,
            )

    def __len__(self):
        """Return the number of data points stored in the table."""
        return len(self.dataframe)

    def __eq__(self, other):
        return self.dataframe.equals(other.dataframe)

    def __json_encode__(self) -> dict[str, Any]:
        return {
            "class": "ScatterTable",
            "data": self.dataframe.to_dict(orient="index"),
        }

    @classmethod
    def __json_decode__(cls, value: dict[str, Any]) -> "ScatterTable":
        if not value.get("class", None) == "ScatterTable":
            raise ValueError("JSON decoded value for ScatterTable is not valid class type.")
        tmp_df = pd.DataFrame.from_dict(value.get("data", {}), orient="index")
        return ScatterTable.from_dataframe(tmp_df)
