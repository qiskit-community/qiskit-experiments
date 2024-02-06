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
from qiskit_experiments.framework.analysis_result_table import AnalysisResultTable
from qiskit_experiments.database_service.exceptions import ExperimentEntryNotFound


class TestAnalysisTable(QiskitExperimentsTestCase):
    """Test case for extra functionality of analysis table."""

    def test_add_get_entry_with_result_id(self):
        """Test adding entry with result_id. Index is created by truncating long string."""
        table = AnalysisResultTable()
        table.add_data(result_id="9a0bdec8-c010-4ef7-bb7d-b84939717a6b", value=0.123)
        self.assertEqual(table.get_data("9a0bdec8").iloc[0].value, 0.123)

    def test_drop_entry(self):
        """Test drop entry from the table."""
        table = AnalysisResultTable()
        table.add_data(result_id="9a0bdec8-c010-4ef7-bb7d-b84939717a6b", value=0.123)
        table.del_data("9a0bdec8")

        self.assertEqual(len(table), 0)

    def test_drop_non_existing_entry(self):
        """Test dropping non-existing entry raises ValueError."""
        table = AnalysisResultTable()
        with self.assertRaises(ExperimentEntryNotFound):
            table.del_data("9a0bdec8")

    def test_raises_adding_duplicated_index(self):
        """Test adding duplicated index should raise."""
        table = AnalysisResultTable()
        table.add_data(result_id="9a0bdec8-c010-4ef7-bb7d-b84939717a6b", value=0.0)

        with self.assertRaises(ValueError):
            # index 9a0bdec8 is already used
            table.add_data(result_id="9a0bdec8-c010-4ef7-bb7d-b84939717a6b", value=1.0)

    def test_clear_container(self):
        """Test reset table."""
        table = AnalysisResultTable()
        table.add_data(result_id="9a0bdec8-c010-4ef7-bb7d-b84939717a6b", value=0.0, extra=123)
        self.assertEqual(len(table), 1)

        table.clear()
        self.assertEqual(len(table), 0)
        self.assertListEqual(table.columns, AnalysisResultTable.DEFAULT_COLUMNS)

    def test_extra_column_name_is_always_returned(self):
        """Test extra column names are always returned in filtered column names."""
        table = AnalysisResultTable()
        table.add_data(extra=0.123)

        minimal_columns = table.get_data(0, "minimal")
        self.assertTrue("extra" in minimal_columns.columns)

        default_columns = table.get_data(0, "default")
        self.assertTrue("extra" in default_columns.columns)

        all_columns = table.get_data(0, "all")
        self.assertTrue("extra" in all_columns.columns)

    def test_get_custom_columns(self):
        """Test getting entry with user-specified columns."""
        table = AnalysisResultTable()
        table.add_data(name="test", value=0)

        cols = ["name", "value"]
        custom_columns = table.get_data(0, cols)
        self.assertListEqual(list(custom_columns.columns), cols)

    def test_warning_non_existing_columns(self):
        """Test raise user warning when attempt to get non-existing column."""
        table = AnalysisResultTable()
        table.add_data(name="test", value=0)

        with self.assertWarns(UserWarning):
            table.get_data(0, ["not_existing_column"])

    def test_listing_result_id(self):
        """Test returning result IDs of all stored entries."""
        table = AnalysisResultTable()

        ref_ids = [str(uuid.uuid4()) for _ in range(10)]
        for ref_id in ref_ids:
            table.add_data(result_id=ref_id, value=0)

        self.assertListEqual(table.result_ids, ref_ids)

    def test_no_overlap_result_id(self):
        """Test automatically prepare unique result IDs for sufficient number of entries."""
        table = AnalysisResultTable()

        for i in range(100):
            table.add_data(value=i)

        self.assertEqual(len(table), 100)

    def test_round_trip(self):
        """Test JSON roundtrip serialization with the experiment encoder."""
        table = AnalysisResultTable()
        table.add_data(result_id="30d5d05c-c074-4d3c-9530-07a83d48883a", name="x", value=0.0)
        table.add_data(result_id="7c305972-858d-42a0-9b5e-57162efe20a1", name="y", value=1.0)
        table.add_data(result_id="61d8d351-c0cf-4a0a-ae57-fde0f3baa00d", name="z", value=2.0)

        self.assertRoundTripSerializable(table)

    def test_round_trip_with_extra(self):
        """Test JSON roundtrip serialization with extra columns containing missing value."""
        table = AnalysisResultTable()
        table.add_data(
            result_id="30d5d05c-c074-4d3c-9530-07a83d48883a",
            name="x",
            value=0.0,
            extra1=2,
        )
        table.add_data(
            result_id="7c305972-858d-42a0-9b5e-57162efe20a1",
            name="y",
            value=1.0,
            extra2=0.123,
        )
        self.assertRoundTripSerializable(table)
