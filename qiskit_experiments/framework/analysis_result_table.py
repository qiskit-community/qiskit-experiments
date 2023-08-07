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

"""Table representation of analysis results."""

import logging
import re
import uuid
import warnings
from typing import List, Union, Optional

import pandas as pd

from qiskit_experiments.database_service.utils import ThreadSafeDataFrame

LOG = logging.getLogger(__name__)


class AnalysisResultTable(ThreadSafeDataFrame):
    """Table form container of analysis results.

    This table is a dataframe wrapper with the thread-safe mechanism with predefined columns.
    This object is attached to the :class:`.ExperimentData` container to store
    analysis results. Each table row contains series of metadata in addition to the
    result value itself.

    User can rely on the dataframe filtering mechanism to analyze large scale experiment
    results, e.g. massive parallel experiment and batch experiment outcomes, efficiently.
    See `pandas dataframe documentation <https://pandas.pydata.org/docs/index.html>`_
    for more details.
    """

    VALID_ID_REGEX = re.compile(r"\A(?P<short_id>\w{8})-\w{4}-\w{4}-\w{4}-\w{12}\Z")

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

    def result_ids(self) -> List[str]:
        """Return all result IDs in this table."""
        with self._lock:
            return self._container["result_id"].to_list()

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
        with self._lock:
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
                            f"Specified column name {column} does not exist in this table.",
                            UserWarning,
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
    ) -> pd.Series:
        """Add new entry to the table.

        Args:
            result_id: Result ID. Automatically generated when not provided.
                This must be valid hexadecimal UUID string.
            kwargs: Description of new entry to register.

        Returns:
            Pandas Series of added entry. This doesn't mutate the table.
        """
        if not result_id:
            result_id = self._unique_table_index()

        matched = self.VALID_ID_REGEX.match(result_id)
        if matched is None:
            raise ValueError(f"The result ID {result_id} is not a valid result ID string.")

        # Short unique index is generated from result id.
        # Showing full result id unnecessary occupies horizontal space of the html table.
        # This mechanism is similar with the github commit hash.
        short_id = matched.group("short_id")

        with self._lock:
            if short_id in self._container.index:
                raise ValueError(
                    f"The short ID of the result_id '{short_id}' already exists in the "
                    "experiment data. Please use another ID to avoid index collision."
                )

        return super().add_entry(
            index=short_id,
            result_id=result_id,
            **kwargs,
        )

    def _unique_table_index(self):
        """Generate unique UUID which is unique in the table with first 8 characters."""
        with self._lock:
            n = 0
            while n < 1000:
                tmp_id = str(uuid.uuid4())
                if tmp_id[:8] not in self._container.index:
                    return tmp_id
        raise RuntimeError(
            "Unique result_id string cannot be prepared for this table within 1000 trials. "
            "Reduce number of entries, or manually provide a unique result_id."
        )
