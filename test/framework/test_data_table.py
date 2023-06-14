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

"""Test case for data table."""

from test.base import QiskitExperimentsTestCase

import numpy as np
import pandas as pd
from qiskit.tools import parallel_map

from qiskit_experiments.database_service.utils import ThreadSafeDataFrame, AnalysisResultTable


class TestBaseTable(QiskitExperimentsTestCase):
    """Test case for data frame base class."""

    class TestTable(ThreadSafeDataFrame):
        """A table class under test with test columns."""

        @classmethod
        def _default_columns(cls):
            return ["value1", "value2", "value3"]

    def test_initializing_with_dict(self):
        """Test initializing table with dictionary. Columns are filled with default."""
        table = TestBaseTable.TestTable(
            {
                "x": [1.0, 2.0, 3.0],
                "y": [4.0, 5.0, 6.0],
            }
        )
        self.assertListEqual(table.get_columns(), ["value1", "value2", "value3"])

    def test_raises_initializing_with_wrong_table(self):
        """Test table cannot be initialized with non-default columns."""
        wrong_table = pd.DataFrame.from_dict(
            data={"x": [1.0, 2.0], "y": [3.0, 4.0], "z": [5.0, 6.0]},
            orient="index",
            columns=["wrong", "columns"],
        )
        with self.assertRaises(ValueError):
            # columns doesn't match with default_columns
            TestBaseTable.TestTable(wrong_table)

    def test_add_entry(self):
        """Test adding data with default keys to table."""
        table = TestBaseTable.TestTable()
        table.add_entry(index="x", value1=0.0, value2=1.0, value3=2.0)

        self.assertListEqual(table.loc["x"].to_list(), [0.0, 1.0, 2.0])

    def test_add_entry_with_missing_key(self):
        """Test adding entry with partly specified keys."""
        table = TestBaseTable.TestTable()
        table.add_entry(index="x", value1=0.0, value3=2.0)

        # NaN value cannot be compared with assert
        np.testing.assert_equal(table.loc["x"].to_list(), [0.0, float("nan"), 2.0])

    def test_add_entry_with_new_key(self):
        """Test adding data with new keys to table."""
        table = TestBaseTable.TestTable()
        table.add_entry(index="x", value1=0.0, value2=1.0, value3=2.0, extra=3.0)

        self.assertListEqual(table.get_columns(), ["value1", "value2", "value3", "extra"])
        self.assertListEqual(table.loc["x"].to_list(), [0.0, 1.0, 2.0, 3.0])

    def test_add_entry_with_new_key_with_existing_entry(self):
        """Test adding new key will expand existing entry."""
        table = TestBaseTable.TestTable()
        table.add_entry(index="x", value1=0.0, value2=1.0, value3=2.0)
        table.add_entry(index="y", value1=0.0, value2=1.0, value3=2.0, extra=3.0)

        self.assertListEqual(table.get_columns(), ["value1", "value2", "value3", "extra"])
        self.assertListEqual(table.loc["y"].to_list(), [0.0, 1.0, 2.0, 3.0])

        # NaN value cannot be compared with assert
        np.testing.assert_equal(table.loc["x"].to_list(), [0.0, 1.0, 2.0, float("nan")])

    def test_return_only_default_columns(self):
        """Test extra entry is correctly recognized."""
        table = TestBaseTable.TestTable()
        table.add_entry(index="x", value1=0.0, value2=1.0, value3=2.0, extra=3.0)

        default_table = table.container(collapse_extra=True)
        self.assertListEqual(default_table.loc["x"].to_list(), [0.0, 1.0, 2.0])

    def test_raises_adding_duplicated_index(self):
        """Test adding duplicated index should raise."""
        table = TestBaseTable.TestTable()
        table.add_entry(index="x", value1=0.0, value2=1.0, value3=2.0)

        with self.assertRaises(ValueError):
            # index x is already used
            table.add_entry(index="x", value1=3.0, value2=4.0, value3=5.0)

    def test_clear_container(self):
        """Test reset table."""
        table = TestBaseTable.TestTable()
        table.add_entry(index="x", value1=0.0, value2=1.0, value3=2.0)
        self.assertEqual(len(table), 1)

        table.clear()
        self.assertEqual(len(table), 0)

    def test_multi_thread_add_entry(self):
        """Test add entry with parallel thread access."""
        table = TestBaseTable.TestTable()

        def _test_thread_local_add_entry(args, thread_table):
            index, kwargs = args
            thread_table.add_entry(index, **kwargs)

        # Mutate thread safe table in parallel thread. No race should occur.
        parallel_map(
            _test_thread_local_add_entry,
            [
                ["x", {"value1": 0.0, "value2": 1.0, "value3": 2.0}],
                ["y", {"value1": 3.0, "value2": 4.0, "value3": 5.0}],
                ["z", {"value1": 6.0, "value2": 7.0, "value3": 8.0}],
            ],
            task_kwargs={"thread_table": table},
        )
        self.assertEqual(len(table), 3)

        self.assertListEqual(table.loc["x"].to_list(), [0.0, 1.0, 2.0])
        self.assertListEqual(table.loc["y"].to_list(), [3.0, 4.0, 5.0])
        self.assertListEqual(table.loc["z"].to_list(), [6.0, 7.0, 8.0])

    def test_round_trip(self):
        """Test JSON roundtrip serialization with the experiment encoder."""
        table = TestBaseTable.TestTable()
        table.add_entry(index="x", value1=0.0, value2=1.0, value3=2.0)
        table.add_entry(index="y", value1=1.0, extra=2.0)

        self.assertRoundTripSerializable(table)


class TestAnalysisTable(QiskitExperimentsTestCase):
    """Test case for extra functionality of analysis table."""

    def test_add_entry_with_result_id(self):
        """Test adding entry with result_id. Index is created by truncating long string."""
        table = AnalysisResultTable()
        table.add_entry(result_id="9a0bdec8c0104ef7bb7db84939717a6b", value=0.123)
        self.assertEqual(table.loc["9a0bdec8"].value, 0.123)

    def test_raises_adding_entry_with_invalid_result_id(self):
        """Test adding entry with non-hexadecimal UUID result id."""
        table = AnalysisResultTable()
        with self.assertRaises(ValueError):
            table.add_entry(result_id="12345678910")

    def test_extra_column_name_is_always_returned(self):
        """Test extra column names are always returned in filtered column names."""
        table = AnalysisResultTable()
        table.add_entry(extra=0.123)

        minimal_columns = table.filter_columns("minimal")
        self.assertTrue("extra" in minimal_columns)

        default_columns = table.filter_columns("default")
        self.assertTrue("extra" in default_columns)

        all_columns = table.filter_columns("all")
        self.assertTrue("extra" in all_columns)

    def test_no_overlap_result_id(self):
        """Test automatically prepare unique result IDs for sufficient number of entries."""
        table = AnalysisResultTable()

        for i in range(100):
            table.add_entry(value=i)

        self.assertEqual(len(table), 100)
