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

"""Experiment utility functions."""

import io
import logging
import threading
import traceback
import warnings
from abc import ABC, abstractmethod
from collections import OrderedDict
from datetime import datetime, timezone
from typing import Callable, Tuple, List, Dict, Any, Union, Type, Optional
import json
import uuid

import pandas as pd
import dateutil.parser
import pkg_resources
from dateutil import tz

from qiskit.version import __version__ as terra_version

from qiskit_ibm_experiment import (
    IBMExperimentEntryExists,
    IBMExperimentEntryNotFound,
)

from .exceptions import ExperimentEntryNotFound, ExperimentEntryExists, ExperimentDataError
from ..version import __version__ as experiments_version

LOG = logging.getLogger(__name__)


def qiskit_version():
    """Return the Qiskit version."""
    try:
        return pkg_resources.get_distribution("qiskit").version
    except Exception:  # pylint: disable=broad-except
        return {"qiskit-terra": terra_version, "qiskit-experiments": experiments_version}


def parse_timestamp(utc_dt: Union[datetime, str]) -> datetime:
    """Parse a UTC ``datetime`` object or string.

    Args:
        utc_dt: Input UTC `datetime` or string.

    Returns:
        A ``datetime`` with the UTC timezone.

    Raises:
        TypeError: If the input parameter value is not valid.
    """
    if isinstance(utc_dt, str):
        utc_dt = dateutil.parser.parse(utc_dt)
    if not isinstance(utc_dt, datetime):
        raise TypeError("Input `utc_dt` is not string or datetime.")
    utc_dt = utc_dt.replace(tzinfo=timezone.utc)
    return utc_dt


def utc_to_local(utc_dt: datetime) -> datetime:
    """Convert input UTC timestamp to local timezone.

    Args:
        utc_dt: Input UTC timestamp.

    Returns:
        A ``datetime`` with the local timezone.
    """
    local_dt = utc_dt.astimezone(tz.tzlocal())
    return local_dt


def plot_to_svg_bytes(figure: "pyplot.Figure") -> bytes:
    """Convert a pyplot Figure to SVG in bytes.

    Args:
        figure: Figure to be converted

    Returns:
        Figure in bytes.
    """
    buf = io.BytesIO()
    opaque_color = list(figure.get_facecolor())
    opaque_color[3] = 1.0  # set alpha to opaque
    figure.savefig(
        buf, format="svg", facecolor=tuple(opaque_color), edgecolor="none", bbox_inches="tight"
    )
    buf.seek(0)
    figure_data = buf.read()
    buf.close()
    return figure_data


def save_data(
    is_new: bool,
    new_func: Callable,
    update_func: Callable,
    new_data: Dict,
    update_data: Dict,
    json_encoder: Optional[Type[json.JSONEncoder]] = None,
) -> Tuple[bool, Any]:
    """Save data in the database.

    Args:
        is_new: ``True`` if `new_func` should be called. Otherwise `update_func` is called.
        new_func: Function to create new entry in the database.
        update_func: Function to update an existing entry in the database.
        new_data: In addition to `update_data`, this data will be stored if creating
            a new entry.
        update_data: Data to be stored if updating an existing entry.
        json_encoder: Custom JSON encoder to use to encode the experiment.

    Returns:
        A tuple of whether the data was saved and the function return value.

    Raises:
        ExperimentDataError: If unable to determine whether the entry exists.
    """
    attempts = 0
    no_entry_exception = (ExperimentEntryNotFound, IBMExperimentEntryNotFound)
    dup_entry_exception = (ExperimentEntryExists, IBMExperimentEntryExists)

    try:
        kwargs = {}
        if json_encoder:
            kwargs["json_encoder"] = json_encoder
        # Attempt 3x for the unlikely scenario wherein is_new=False but the
        # entry doesn't actually exist. The second try might also fail if an entry
        # with the same ID somehow got created in the meantime.
        while attempts < 3:
            attempts += 1
            if is_new:
                try:
                    kwargs.update(new_data)
                    kwargs.update(update_data)
                    return True, new_func(**kwargs)
                except dup_entry_exception:
                    is_new = False
            else:
                try:
                    kwargs.update(update_data)
                    return True, update_func(**kwargs)
                except no_entry_exception:
                    is_new = True
        raise ExperimentDataError("Unable to determine the existence of the entry.")
    except Exception:  # pylint: disable=broad-except
        # Don't fail the experiment just because its data cannot be saved.
        LOG.error("Unable to save the experiment data: %s", traceback.format_exc())
        return False, None


class ThreadSafeContainer(ABC):
    """Base class for thread safe container."""

    def __init__(self, init_values=None):
        """ThreadSafeContainer constructor."""
        self._lock = threading.RLock()
        self._container = self._init_container(init_values)

    @abstractmethod
    def _init_container(self, init_values):
        """Initialize the container."""
        pass

    def __iter__(self):
        with self._lock:
            return iter(self._container)

    def __getitem__(self, key):
        with self._lock:
            return self._container[key]

    def __setitem__(self, key, value):
        with self._lock:
            self._container[key] = value

    def __delitem__(self, key):
        with self._lock:
            del self._container[key]

    def __contains__(self, item):
        with self._lock:
            return item in self._container

    def __len__(self):
        with self._lock:
            return len(self._container)

    @property
    def lock(self):
        """Return lock used for this container."""
        return self._lock

    def copy(self):
        """Returns a copy of the container."""
        with self.lock:
            return self._container.copy()

    def copy_object(self):
        """Returns a copy of this object."""
        obj = self.__class__()
        obj._container = self.copy()
        return obj

    def clear(self):
        """Remove all elements from this container."""
        with self.lock:
            self._container.clear()

    def __json_encode__(self):
        cpy = self.copy_object()
        return {"_container": cpy._container}

    @classmethod
    def __json_decode__(cls, value):
        ret = cls()
        ret._container = value["_container"]
        return ret

    def __getstate__(self):
        state = self.__dict__.copy()
        # Remove non-pickleable attribute
        del state["_lock"]
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        # Initialize non-pickleable attribute
        self._lock = threading.RLock()


class ThreadSafeOrderedDict(ThreadSafeContainer):
    """Thread safe OrderedDict."""

    def _init_container(self, init_values):
        """Initialize the container."""
        return OrderedDict.fromkeys(init_values or [])

    def get(self, key, default):
        """Return the value of the given key."""
        with self._lock:
            return self._container.get(key, default)

    def keys(self):
        """Return all key values."""
        with self._lock:
            return list(self._container.keys())

    def values(self):
        """Return all values."""
        with self._lock:
            return list(self._container.values())

    def items(self):
        """Return the key value pairs."""
        return self._container.items()


class ThreadSafeList(ThreadSafeContainer):
    """Thread safe list."""

    def _init_container(self, init_values):
        """Initialize the container."""
        return init_values or []

    def append(self, value):
        """Append to the list."""
        with self._lock:
            self._container.append(value)


class ThreadSafeDataFrame(ThreadSafeContainer):
    """Thread safe data frame.

    This class wraps pandas dataframe with predefined column labels,
    which is specified by the class method `_default_columns`.
    Subclass can override this method to provide default labels specific to its data structure.

    This object is expected to be used internally in the ExperimentData.
    """

    def __init__(self, init_values=None):
        """ThreadSafeContainer constructor."""
        self._columns = self._default_columns()
        self._extra = []
        super().__init__(init_values)

    @classmethod
    def _default_columns(cls) -> List[str]:
        return []

    def _init_container(self, init_values: Optional[Union[Dict, pd.DataFrame]] = None):
        """Initialize the container."""
        if init_values is None:
            return pd.DataFrame(columns=self.get_columns())
        if isinstance(init_values, pd.DataFrame):
            input_columns = list(init_values.columns)
            if input_columns != self.get_columns():
                raise ValueError(
                    f"Input data frame contains unexpected columns {input_columns}. "
                    f"{self.__class__.__name__} defines {self.get_columns()} as default columns."
                )
            return init_values
        if isinstance(init_values, dict):
            return pd.DataFrame.from_dict(
                data=init_values,
                orient="index",
                columns=self.get_columns(),
            )
        raise TypeError(f"Initial value of {type(init_values)} is not valid data type.")

    def get_columns(self) -> List[str]:
        """Return current column names.

        Returns:
            List of column names.
        """
        return self._columns.copy()

    def add_columns(self, *new_columns: str, default_value: Any = None):
        """Add new columns to the table.

        This operation mutates the current container.

        Args:
            new_columns: Name of columns to add.
            default_value: Default value to fill added columns.
        """
        # Order sensitive
        new_columns = [c for c in new_columns if c not in self.get_columns()]
        self._extra.extend(new_columns)

        # Update current table
        with self._lock:
            for new_column in new_columns:
                self._container.insert(len(self._container.columns), new_column, default_value)
        self._columns.extend(new_columns)

    def clear(self):
        """Remove all elements from this container."""
        with self._lock:
            self._container = self._init_container()
            self._columns = self._default_columns()
            self._extra = []

    def container(
        self,
        collapse_extra: bool = True,
    ) -> pd.DataFrame:
        """Return bare pandas dataframe.

        Args:
            collapse_extra: Set True to show only default columns.

        Returns:
            Bare pandas dataframe. This object is no longer thread safe.
        """
        with self._lock:
            container = self._container

        if collapse_extra:
            return container[self._default_columns()]
        return container

    def add_entry(
        self,
        index: str,
        **kwargs,
    ):
        """Add new entry to the dataframe.

        Args:
            index: Name of this entry. Must be unique in this table.
            kwargs: Description of new entry to register.

        Raises:
            ValueError: When index is not unique in this table.
        """
        with self.lock:
            if index in self._container.index:
                raise ValueError(f"Table index {index} already exists in the table.")

        columns = self.get_columns()
        missing = kwargs.keys() - set(columns)
        if missing:
            self.add_columns(*sorted(missing))

        template = dict.fromkeys(self.get_columns())
        template.update(kwargs)

        with self._lock:
            self._container.loc[index] = list(template.values())

    def _repr_html_(self) -> Union[str, None]:
        """Return HTML representation of this dataframe."""
        with self._lock:
            # Remove underscored columns.
            return self._container._repr_html_()

    def __getattr__(self, item):
        lock = object.__getattribute__(self, "_lock")

        with lock:
            # Lock when access to container's member.
            container = object.__getattribute__(self, "_container")
            if hasattr(container, item):
                return getattr(container, item)
        raise AttributeError(f"'ThreadSafeDataFrame' object has no attribute '{item}'")

    def __json_encode__(self) -> Dict[str, Any]:
        return {
            "class": "ThreadSafeDataFrame",
            "data": self._container.to_dict(orient="index"),
            "columns": self._columns,
            "extra": self._extra,
        }

    @classmethod
    def __json_decode__(cls, value: Dict[str, Any]) -> "ThreadSafeDataFrame":
        if not value.get("class", None) == "ThreadSafeDataFrame":
            raise ValueError("JSON decoded value for ThreadSafeDataFrame is not valid class type.")

        instance = object.__new__(AnalysisResultTable)
        # Need to update self._columns first to set extra columns in the dataframe container.
        instance._columns = value.get("columns", cls._default_columns())
        instance._extra = value.get("extra", [])
        instance._lock = threading.RLock()
        instance._container = instance._init_container(init_values=value.get("data", {}))
        return instance


class AnalysisResultTable(ThreadSafeDataFrame):
    """Thread safe dataframe to store the analysis results."""

    @classmethod
    def _default_columns(cls) -> List[str]:
        return [
            "name",
            "experiment",
            "components",
            "value",
            "quality",
            "experiment_id",
            "result_id",
            "tags",
            "backend",
            "run_time",
            "created_time",
        ]

    def filter_columns(self, columns: Union[str, List[str]]) -> List[str]:
        """Filter columns names available in this table.

        Args:
            columns: Specifying a set of columns to return. You can pass a list of each
                column name to return, otherwise builtin column groups are available.

                    * "all": Return all columns, including metadata to communicate
                        with experiment service, such as entry IDs.
                    * "default": Return columns including analysis result with supplementary
                        information about experiment.
                    * "minimal": Return only analysis subroutine returns.

        Raises:
            ValueError: When column is given in string which doesn't match with any builtin group.
        """
        if columns == "all":
            return self._columns
        if columns == "default":
            return [
                "name",
                "experiment",
                "components",
                "value",
                "quality",
                "backend",
                "run_time",
            ] + self._extra
        if columns == "minimal":
            return [
                "name",
                "components",
                "value",
                "quality",
            ] + self._extra
        if not isinstance(columns, str):
            out = []
            for column in columns:
                if column in self._columns:
                    out.append(column)
                else:
                    warnings.warn(
                        f"Specified column name {column} does not exist in this table.", UserWarning
                    )
            return out
        raise ValueError(
            f"Column group {columns} is not valid name. Use either 'all', 'default', 'minimal'."
        )

    # pylint: disable=arguments-renamed
    def add_entry(
        self,
        result_id: Optional[str] = None,
        **kwargs,
    ):
        """Add new entry to the table.

        Args:
            result_id: Result ID. Automatically generated when not provided.
                This must be valid hexadecimal UUID string.
            kwargs: Description of new entry to register.
        """
        if result_id:
            try:
                result_id = uuid.UUID(result_id, version=4).hex
            except ValueError as ex:
                raise ValueError(f"{result_id} is not a valid hexadecimal UUID string.") from ex
        else:
            result_id = self._unique_table_index()

        # Short unique index is generated from full UUID.
        # Showing full UUID unnecessary occupies horizontal space of the html table.
        short_index = result_id[:8]

        super().add_entry(
            index=short_index,
            result_id=result_id,
            **kwargs,
        )

    def _unique_table_index(self):
        """Generate unique UUID which is unique in the table with first 8 characters."""
        with self.lock:
            n = 0
            while n < 1000:
                tmp_id = uuid.uuid4().hex
                if tmp_id[:8] not in self._container.index:
                    return tmp_id
        raise RuntimeError(
            "Unique result_id string cannot be prepared for this table within 1000 trials. "
            "Reduce number of entries, or manually provide a unique result_id."
        )
