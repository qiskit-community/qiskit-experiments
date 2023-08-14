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

# pylint: disable=no-member

"""MinIn class for pandas dataframe."""
from typing import List, Callable
from functools import wraps

import pandas as pd


class DefaultColumnsMixIn:
    """A mixin that sets a default data columns to a dataframe subclass.

    Subclass must define _default_columns class method to provide default column names.
    This name list is order sensitive and the first element will show up at the
    most left column of the dataframe table.
    """

    _default_columns: Callable

    def __init_subclass__(cls, **kwargs):
        # To make sure the mixin constructor is called.
        super().__init_subclass__(**kwargs)

        @wraps(cls.__init__, assigned=("__annotations__",))
        def _call_init_and_set_columns(self, *init_args, **init_kwargs):
            super(cls, self).__init__(*init_args, **init_kwargs)
            if len(self.columns) == 0:
                self.add_columns(*cls._default_columns())

        # Monkey patch the mixed class constructor to make sure default columns are added
        cls.__init__ = _call_init_and_set_columns

    def add_columns(
        self: pd.DataFrame,
        *new_columns: str,
    ):
        """Add new columns to the table.

        This operation mutates the current container.

        Args:
            new_columns: Name of columns to add.
        """
        # Order sensitive
        new_columns = [c for c in new_columns if c not in self.columns]
        if len(new_columns) == 0:
            return

        # Update columns
        for new_column in new_columns:
            loc = len(self.columns)
            self.insert(loc, new_column, value=None)

    def add_entry(
        self: pd.DataFrame,
        index: str,
        **kwargs,
    ):
        """Add new entry to the dataframe.

        Args:
            index: Name of this entry. Must be unique in this table.
            kwargs: Description of new entry to register.

        Returns:
            Pandas Series of added entry. This doesn't mutate the table.
        """
        if not isinstance(index, str):
            index = str(index)
        if kwargs.keys() - set(self.columns):
            self.add_columns(*kwargs.keys())

        # A hack to avoid unwanted dtype update. Appending new row with .loc indexer
        # performs enlargement and implicitly changes dtype. This often induces a confusion of
        # NaN (numeric container) and None (object container) for missing values.
        # Filling a row with None values before assigning actual values can keep column dtype,
        # but this behavior might change in future pandas version.
        # https://github.com/pandas-dev/pandas/issues/6485
        # Also see test.framework.test_data_table.TestBaseTable.test_type_*
        self.loc[index] = [None] * len(self.columns)

        template = dict.fromkeys(self.columns, None)
        template.update(kwargs)
        self.loc[index] = pd.array(list(template.values()), dtype=object)

    def extra_columns(
        self: pd.DataFrame,
    ) -> List[str]:
        """Return a list of columns added by a user."""
        return [c for c in self.columns if c not in self._default_columns()]
