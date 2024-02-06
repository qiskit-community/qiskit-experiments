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

"""Test scatter table."""

from test.base import QiskitExperimentsTestCase
import pandas as pd
import numpy as np

from qiskit_experiments.curve_analysis.scatter_table import ScatterTable


class TestScatterTable(QiskitExperimentsTestCase):
    """Test cases for curve analysis ScatterTable."""

    def setUp(self):
        super().setUp()

        source = {
            "xval": [0.100, 0.100, 0.200, 0.200, 0.100, 0.200, 0.100, 0.200, 0.100, 0.200],
            "yval": [0.192, 0.784, 0.854, 0.672, 0.567, 0.488, 0.379, 0.671, 0.784, 0.672],
            "yerr": [0.002, 0.091, 0.090, 0.027, 0.033, 0.038, 0.016, 0.048, 0.091, 0.027],
            "series_name": [
                "model1",
                "model2",
                "model1",
                "model2",
                "model1",
                "model1",
                "model1",
                "model1",
                "model2",
                "model2",
            ],
            "series_id": [0, 1, 0, 1, 0, 0, 0, 0, 1, 1],
            "category": [
                "raw",
                "raw",
                "raw",
                "raw",
                "raw",
                "raw",
                "formatted",
                "formatted",
                "formatted",
                "formatted",
            ],
            "shots": [
                1000,
                1000,
                1000,
                1000,
                1000,
                1000,
                2000,
                2000,
                1000,
                1000,
            ],
            "analysis": [
                "Fit1",
                "Fit1",
                "Fit1",
                "Fit1",
                "Fit2",
                "Fit2",
                "Fit1",
                "Fit1",
                "Fit1",
                "Fit1",
            ],
        }
        self.reference = pd.DataFrame.from_dict(source)

    def test_create_table_from_dataframe(self):
        """Test creating table from dataframe and output dataframe."""
        # ScatterTable automatically converts dtype.
        # For pure dataframe equality check pre-format the source.
        formatted_ref = ScatterTable._format_table(self.reference)

        obj = ScatterTable.from_dataframe(formatted_ref)
        self.assertTrue(obj.dataframe.equals(formatted_ref))

    def test_factory_method_check_all_members(self):
        """Test to check the factory method populates all instance members."""
        to_test = ScatterTable.from_dataframe(pd.DataFrame(columns=ScatterTable.COLUMNS))
        ref = ScatterTable()
        self.assertEqual(to_test.__dict__.keys(), ref.__dict__.keys())

    def test_two_construction_method_identical(self):
        """Check if two tables constructed differently from the same source are identical."""
        new_table = ScatterTable()
        for _, row_data in self.reference.iterrows():
            new_table.add_row(**row_data)

        ref_table = ScatterTable.from_dataframe(self.reference)
        self.assertEqual(new_table, ref_table)

    def test_add_row(self):
        """Test adding single row to the table without and with missing data."""
        obj = ScatterTable()
        obj.add_row(
            xval=0.1,
            yval=2.3,
            yerr=0.4,
            series_name="model1",
            series_id=0,
            category="raw",
            shots=1000,
            analysis="Test",
        )
        obj.add_row(
            category="raw",
            xval=0.2,
            yval=3.4,
        )
        self.assertEqual(len(obj), 2)
        np.testing.assert_array_equal(obj.x, np.array([0.1, 0.2]))
        np.testing.assert_array_equal(obj.y, np.array([2.3, 3.4]))
        np.testing.assert_array_equal(obj.y_err, np.array([0.4, np.nan]))
        np.testing.assert_array_equal(obj.series_name, np.array(["model1", None]))
        np.testing.assert_array_equal(obj.series_id, np.array([0, None]))
        np.testing.assert_array_equal(obj.category, np.array(["raw", "raw"]))
        np.testing.assert_array_equal(
            # Numpy tries to handle nan strictly, but isnan only works for float dtype.
            # Original data is object type, because we want to keep shot number integer,
            # and there is no Numpy nullable integer.
            obj.shots.astype(float),
            np.array([1000, np.nan], dtype=float),
        )
        np.testing.assert_array_equal(obj.analysis, np.array(["Test", None]))

    def test_set_values(self):
        """Test setting new column values through setter."""
        obj = ScatterTable()
        # add three empty rows
        obj.add_row()
        obj.add_row()
        obj.add_row()

        # Set sequence
        obj.x = [0.1, 0.2, 0.3]
        obj.y = [1.3, 1.4, 1.5]
        obj.y_err = [0.3, 0.5, 0.7]

        # Broadcast single value
        obj.series_id = 0
        obj.series_name = "model0"

        np.testing.assert_array_equal(obj.x, np.array([0.1, 0.2, 0.3]))
        np.testing.assert_array_equal(obj.y, np.array([1.3, 1.4, 1.5]))
        np.testing.assert_array_equal(obj.y_err, np.array([0.3, 0.5, 0.7]))
        np.testing.assert_array_equal(obj.series_id, np.array([0, 0, 0]))
        np.testing.assert_array_equal(obj.series_name, np.array(["model0", "model0", "model0"]))

    def test_get_subset_numbers(self):
        """Test end-user shortcut for getting the subset of x, y, y_err data."""
        obj = ScatterTable.from_dataframe(self.reference)

        np.testing.assert_array_equal(obj.xvals("model1", "raw", "Fit1"), np.array([0.100, 0.200]))
        np.testing.assert_array_equal(obj.yvals("model1", "raw", "Fit1"), np.array([0.192, 0.854]))
        np.testing.assert_array_equal(obj.yerrs("model1", "raw", "Fit1"), np.array([0.002, 0.090]))

    def test_warn_composite_values(self):
        """Test raise warning when returned x, y, y_err data contains multiple data series."""
        obj = ScatterTable.from_dataframe(self.reference)

        with self.assertWarns(UserWarning):
            obj.xvals()
        with self.assertWarns(UserWarning):
            obj.yvals()
        with self.assertWarns(UserWarning):
            obj.yerrs()

    def test_filter_data_by_series_id(self):
        """Test filter table data with series index."""
        obj = ScatterTable.from_dataframe(self.reference)

        filtered = obj.filter(series=0)
        self.assertEqual(len(filtered), 6)
        np.testing.assert_array_equal(filtered.x, np.array([0.1, 0.2, 0.1, 0.2, 0.1, 0.2]))
        np.testing.assert_array_equal(filtered.series_id, np.array([0, 0, 0, 0, 0, 0]))

    def test_filter_data_by_series_name(self):
        """Test filter table data with series name."""
        obj = ScatterTable.from_dataframe(self.reference)

        filtered = obj.filter(series="model1")
        self.assertEqual(len(filtered), 6)
        np.testing.assert_array_equal(filtered.x, np.array([0.1, 0.2, 0.1, 0.2, 0.1, 0.2]))
        np.testing.assert_array_equal(
            filtered.series_name,
            np.array(["model1", "model1", "model1", "model1", "model1", "model1"]),
        )

    def test_filter_data_by_category(self):
        """Test filter table data with data category."""
        obj = ScatterTable.from_dataframe(self.reference)

        filtered = obj.filter(category="formatted")
        self.assertEqual(len(filtered), 4)
        np.testing.assert_array_equal(filtered.x, np.array([0.1, 0.2, 0.1, 0.2]))
        np.testing.assert_array_equal(
            filtered.category, np.array(["formatted", "formatted", "formatted", "formatted"])
        )

    def test_filter_data_by_analysis(self):
        """Test filter table data with associated analysis class."""
        obj = ScatterTable.from_dataframe(self.reference)

        filtered = obj.filter(analysis="Fit2")
        self.assertEqual(len(filtered), 2)
        np.testing.assert_array_equal(filtered.x, np.array([0.1, 0.2]))
        np.testing.assert_array_equal(filtered.analysis, np.array(["Fit2", "Fit2"]))

    def test_filter_multiple(self):
        """Test filter table data with multiple attributes."""
        obj = ScatterTable.from_dataframe(self.reference)

        filtered = obj.filter(series=0, category="raw", analysis="Fit1")
        self.assertEqual(len(filtered), 2)
        np.testing.assert_array_equal(filtered.x, np.array([0.1, 0.2]))
        np.testing.assert_array_equal(filtered.series_id, np.array([0, 0]))
        np.testing.assert_array_equal(filtered.category, np.array(["raw", "raw"]))
        np.testing.assert_array_equal(filtered.analysis, np.array(["Fit1", "Fit1"]))

    def test_iter_class(self):
        """Test iterating over mini tables associated with different series indices."""
        obj = ScatterTable.from_dataframe(self.reference).filter(category="raw")

        class_iter = obj.iter_by_series_id()

        series_id, table0 = next(class_iter)
        ref_table_cls0 = obj.filter(series=0)
        self.assertEqual(series_id, 0)
        self.assertEqual(table0, ref_table_cls0)

        series_id, table1 = next(class_iter)
        ref_table_cls1 = obj.filter(series=1)
        self.assertEqual(series_id, 1)
        self.assertEqual(table1, ref_table_cls1)

    def test_iter_groups(self):
        """Test iterating over mini tables associated with multiple attributes."""
        obj = ScatterTable.from_dataframe(self.reference).filter(category="raw")

        class_iter = obj.iter_groups("series_id", "xval")

        (series_id, xval), table0 = next(class_iter)
        self.assertEqual(series_id, 0)
        self.assertEqual(xval, 0.1)
        self.assertEqual(len(table0), 2)
        np.testing.assert_array_equal(table0.y, [0.192, 0.567])

        (series_id, xval), table1 = next(class_iter)
        self.assertEqual(series_id, 0)
        self.assertEqual(xval, 0.2)
        self.assertEqual(len(table1), 2)
        np.testing.assert_array_equal(table1.y, [0.854, 0.488])

        (series_id, xval), table2 = next(class_iter)
        self.assertEqual(series_id, 1)
        self.assertEqual(xval, 0.1)
        self.assertEqual(len(table2), 1)
        np.testing.assert_array_equal(table2.y, [0.784])

        (series_id, xval), table3 = next(class_iter)
        self.assertEqual(series_id, 1)
        self.assertEqual(xval, 0.2)
        self.assertEqual(len(table3), 1)
        np.testing.assert_array_equal(table3.y, [0.672])

    def test_roundtrip_table(self):
        """Test ScatterTable is JSON serializable."""
        obj = ScatterTable.from_dataframe(self.reference)
        self.assertRoundTripSerializable(obj)
