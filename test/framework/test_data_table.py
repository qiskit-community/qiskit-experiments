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

"""Test case for data table."""

from test.base import QiskitExperimentsTestCase

import uuid
import pandas as pd

from qiskit_experiments.framework.analysis_result_table import AnalysisResultTable
from qiskit_experiments.framework.table_mixin import DefaultColumnsMixIn


def _callable_thread_local_add_entry(args, thread_table):
    """A test callable that is called from multi-thread."""
    index, kwargs = args
    thread_table.add_entry(index, **kwargs)


class TestBaseTable(QiskitExperimentsTestCase):
    """Test case for default columns mix-in."""

    class TestTable(pd.DataFrame, DefaultColumnsMixIn):
        """A table class under test with test columns."""

        @classmethod
        def _default_columns(cls):
            return ["value1", "value2", "value3"]

    def test_initializing_with_dict(self):
        """Test initializing table with dictionary."""
        table = TestBaseTable.TestTable.from_dict(
            {
                "x": {"value1": 1.0, "value2": 2.0, "value3": 3.0},
                "y": {"value1": 4.0, "value2": 5.0, "value3": 6.0},
            },
            orient="index",
        )
        self.assertListEqual(list(table.columns), ["value1", "value2", "value3"])

    def test_add_entry(self):
        """Test adding data with default keys to table."""
        table = TestBaseTable.TestTable()
        table.add_entry(index="x", value1=0.0, value2=1.0, value3=2.0)

        self.assertListEqual(table.loc["x"].to_list(), [0.0, 1.0, 2.0])

    def test_add_entry_with_missing_key(self):
        """Test adding entry with partly specified keys."""
        table = TestBaseTable.TestTable()
        table.add_entry(index="x", value1=0.0, value3=2.0)
        self.assertListEqual(table.loc["x"].to_list(), [0.0, None, 2.0])

    def test_add_entry_with_new_key(self):
        """Test adding data with new keys to table."""
        table = TestBaseTable.TestTable()
        table.add_entry(index="x", value1=0.0, value2=1.0, value3=2.0, extra=3.0)

        self.assertListEqual(list(table.columns), ["value1", "value2", "value3", "extra"])
        self.assertListEqual(table.loc["x"].to_list(), [0.0, 1.0, 2.0, 3.0])

    def test_add_entry_with_multiple_new_keys(self):
        """Test new keys are added to column and the key order is preserved."""
        table = TestBaseTable.TestTable()
        table.add_entry(index="x", phi=0.1, lamb=0.2, theta=0.3)

        self.assertListEqual(
            list(table.columns), ["value1", "value2", "value3", "phi", "lamb", "theta"]
        )

    def test_dtype_missing_value_is_none(self):
        """Test if missing value is always None.

        Deta frame implicitly convert None into NaN for numeric container.
        This should not happen.
        """
        table = TestBaseTable.TestTable()
        table.add_entry(index="x", value1=1.0)
        table.add_entry(index="y", value2=1.0)

        self.assertEqual(table.loc["x", "value2"], None)
        self.assertEqual(table.loc["y", "value1"], None)

    def test_dtype_adding_extra_later(self):
        """Test adding new row later with a numeric value doesn't change None to NaN."""
        table = TestBaseTable.TestTable()
        table.add_entry(index="x")
        table.add_entry(index="y", extra=1.0)

        self.assertListEqual(table.loc["x"].to_list(), [None, None, None, None])

    def test_dtype_adding_null_row(self):
        """Test adding new row with empty value doesn't change dtype of the columns."""
        table = TestBaseTable.TestTable()
        table.add_entry(index="x", extra1=1, extra2=1.0, extra3=True, extra4="abc")
        table.add_entry(index="y")

        self.assertIsInstance(table.loc["x", "extra1"], int)
        self.assertIsInstance(table.loc["x", "extra2"], float)
        self.assertIsInstance(table.loc["x", "extra3"], bool)
        self.assertIsInstance(table.loc["x", "extra4"], str)

    def test_filter_columns(self):
        """Test filtering table with columns."""
        table = TestBaseTable.TestTable()
        table.add_entry(index="x", value1=0.0, value2=1.0, value3=2.0)

        filt_table = table[["value1", "value3"]]
        self.assertListEqual(filt_table.loc["x"].to_list(), [0.0, 2.0])


class TestAnalysisTable(QiskitExperimentsTestCase):
    """Test case for extra functionality of analysis table."""

    def test_add_get_entry_with_result_id(self):
        """Test adding entry with result_id. Index is created by truncating long string."""
        table = AnalysisResultTable()
        table.add_entry(result_id="9a0bdec8-c010-4ef7-bb7d-b84939717a6b", value=0.123)
        self.assertEqual(table.get_entry("9a0bdec8").value, 0.123)

    def test_drop_entry(self):
        """Test drop entry from the table."""
        table = AnalysisResultTable()
        table.add_entry(result_id="9a0bdec8-c010-4ef7-bb7d-b84939717a6b", value=0.123)
        table.drop_entry("9a0bdec8")

        self.assertEqual(len(table), 0)

    def test_drop_non_existing_entry(self):
        """Test dropping non-existing entry raises ValueError."""
        table = AnalysisResultTable()
        with self.assertRaises(ValueError):
            table.drop_entry("9a0bdec8")

    def test_raises_adding_duplicated_index(self):
        """Test adding duplicated index should raise."""
        table = AnalysisResultTable()
        table.add_entry(result_id="9a0bdec8-c010-4ef7-bb7d-b84939717a6b", value=0.0)

        with self.assertRaises(ValueError):
            # index 9a0bdec8 is already used
            table.add_entry(result_id="9a0bdec8-c010-4ef7-bb7d-b84939717a6b", value=1.0)

    def test_clear_container(self):
        """Test reset table."""
        table = AnalysisResultTable()
        table.add_entry(result_id="9a0bdec8-c010-4ef7-bb7d-b84939717a6b", value=0.0, extra=123)
        self.assertEqual(len(table), 1)

        table.clear()
        self.assertEqual(len(table), 0)
        self.assertListEqual(table.copy().extra_columns(), [])

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

    def test_listing_result_id(self):
        """Test returning result IDs of all stored entries."""
        table = AnalysisResultTable()

        ref_ids = [str(uuid.uuid4()) for _ in range(10)]
        for ref_id in ref_ids:
            table.add_entry(result_id=ref_id, value=0)

        self.assertListEqual(table.result_ids(), ref_ids)

    def test_no_overlap_result_id(self):
        """Test automatically prepare unique result IDs for sufficient number of entries."""
        table = AnalysisResultTable()

        for i in range(100):
            table.add_entry(value=i)

        self.assertEqual(len(table), 100)

    def test_round_trip(self):
        """Test JSON roundtrip serialization with the experiment encoder."""
        table = AnalysisResultTable()
        table.add_entry(result_id="30d5d05c-c074-4d3c-9530-07a83d48883a", name="x", value=0.0)
        table.add_entry(result_id="7c305972-858d-42a0-9b5e-57162efe20a1", name="y", value=1.0)
        table.add_entry(result_id="61d8d351-c0cf-4a0a-ae57-fde0f3baa00d", name="z", value=2.0)

        self.assertRoundTripSerializable(table)

    def test_round_trip_with_extra(self):
        """Test JSON roundtrip serialization with extra columns containing missing value."""
        table = AnalysisResultTable()
        table.add_entry(
            result_id="30d5d05c-c074-4d3c-9530-07a83d48883a",
            name="x",
            value=0.0,
            extra1=2,
        )
        table.add_entry(
            result_id="7c305972-858d-42a0-9b5e-57162efe20a1",
            name="y",
            value=1.0,
            extra2=0.123,
        )
        self.assertRoundTripSerializable(table)
