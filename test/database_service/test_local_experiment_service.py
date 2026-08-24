# This code is part of Qiskit.
#
# (C) Copyright IBM 2021-2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Local experiment client tests"""
import unittest
import json
from dataclasses import asdict
from datetime import datetime, timezone
from tempfile import TemporaryDirectory
from typing import Any

from test.base import QiskitExperimentsTestCase

import yaml

from qiskit_experiments.database_service import (
    DbAnalysisResultData,
    DbExperimentData,
    ExperimentEntryNotFound,
    LocalExperimentService,
    ResultQuality,
)


class TestExperimentLocalClient(QiskitExperimentsTestCase):
    """Test experiment modules."""

    def setUp(self):
        """Initial class level setup."""
        super().setUp()
        self.service = LocalExperimentService()

    def test_create_or_update_experiment(self):
        """Tests creating an experiment"""
        data = DbExperimentData(
            experiment_type="test_experiment",
            backend="ibmq_qasm_simulator",
            metadata={"float_data": 3.14, "string_data": "foo"},
        )
        exp_id = self.service.create_or_update_experiment(data).experiment_id
        self.assertIsNotNone(exp_id)

        exp = self.service.experiment(experiment_id=exp_id)
        self.assertEqual(exp.experiment_type, "test_experiment")
        self.assertEqual(exp.backend, "ibmq_qasm_simulator")
        self.assertEqual(exp.metadata["float_data"], 3.14)
        self.assertEqual(exp.metadata["string_data"], "foo")

    def test_update_experiment(self):
        """Tests updating an experiment"""
        data = DbExperimentData(
            experiment_type="test_experiment",
            backend="ibmq_qasm_simulator",
            metadata={"float_data": 3.14, "string_data": "foo"},
        )
        exp_id = self.service.create_or_update_experiment(data).experiment_id
        data = self.service.experiment(exp_id)
        data.metadata["float_data"] = 2.71
        data.experiment_type = "foo_type"
        data.notes = ["foo_note"]
        self.service.create_or_update_experiment(data)
        result = self.service.experiment(exp_id)
        self.assertEqual(result.metadata["float_data"], 2.71)
        self.assertEqual(result.experiment_type, "foo_type")
        self.assertEqual(result.notes[0], "foo_note")

    def test_delete_experiment(self):
        """Tests deleting an experiment"""
        data = DbExperimentData(
            experiment_type="test_experiment",
            backend="ibmq_qasm_simulator",
        )
        exp_id = self.service.create_or_update_experiment(data).experiment_id
        # Check the experiment exists
        self.service.experiment(experiment_id=exp_id)
        self.service.delete_experiment(exp_id)
        with self.assertRaises(ExperimentEntryNotFound):
            self.service.experiment(experiment_id=exp_id)

    def test_create_or_update_analysis_result(self):
        """Tests creating an analysis result"""
        exp_id = self.service.create_or_update_experiment(
            DbExperimentData(experiment_type="test_experiment", backend="ibmq_qasm_simulator")
        ).experiment_id
        analysis_result_value = {"str": "foo", "float": 3.14}
        analysis_data = DbAnalysisResultData(
            experiment_id=exp_id,
            result_data=analysis_result_value,
            result_type="qiskit_test",
        )
        analysis_id = self.service.create_or_update_analysis_result(analysis_data)
        result = self.service.analysis_result(result_id=analysis_id)
        self.assertEqual(result.result_type, "qiskit_test")
        self.assertEqual(result.result_data["str"], analysis_result_value["str"])
        self.assertEqual(result.result_data["float"], analysis_result_value["float"])

    def test_get_analysis_results(self):
        """Tests getting an analysis result"""
        exp_id = self.service.create_or_update_experiment(
            DbExperimentData(experiment_type="test_experiment", backend="ibmq_qasm_simulator")
        ).experiment_id
        result_ids = ["00", "01", "10", "11"]
        for result_id in result_ids:
            analysis_result_value = {
                "str": f"foo_{result_id}",
                "float": 3.14 + int(result_id),
            }
            analysis_data = DbAnalysisResultData(
                experiment_id=exp_id,
                result_id=result_id,
                result_data=analysis_result_value,
                result_type=f"test_get_analysis_results_{result_id[0]}",
            )
            self.service.create_or_update_analysis_result(analysis_data)
        results = self.service.analysis_results(
            result_type="test_get_analysis_results_0",
        )
        self.assertEqual(len(results), 2)
        results = self.service.analysis_results(result_type="test_get_analysis_results_1")
        self.assertEqual(len(results), 2)
        self.assertSetEqual({r.result_data["float"] for r in results}, {3.14 + 10, 3.14 + 11})

    def test_delete_analysis_result(self):
        """Tests deleting an analysis result"""
        exp_id = self.service.create_or_update_experiment(
            DbExperimentData(experiment_type="test_experiment", backend="ibmq_qasm_simulator")
        ).experiment_id
        analysis_data = DbAnalysisResultData(
            experiment_id=exp_id,
            result_data={"foo": "delete_bar"},
            result_type="test_result",
        )
        result_id = self.service.create_or_update_analysis_result(analysis_data)
        result = self.service.analysis_result(result_id)
        self.assertEqual(result.result_data["foo"], "delete_bar")
        self.service.delete_analysis_result(result_id)
        with self.assertRaises(ExperimentEntryNotFound):
            result = self.service.analysis_result(result_id)

    def test_update_analysis_result(self):
        """Test updating an analysis result."""
        result_id = self._create_analysis_result()
        fit = {"value": 41.456, "variance": 4.051}
        chisq = 1.3253

        self.service.create_or_update_analysis_result(
            DbAnalysisResultData(
                result_id=result_id,
                result_data=fit,
                tags=["qiskit_test"],
                quality=ResultQuality.GOOD,
                verified=True,
                chisq=chisq,
            ),
            create=False,
        )

        rresult = self.service.analysis_result(result_id)
        self.assertEqual(result_id, rresult.result_id)
        self.assertEqual(fit, rresult.result_data)
        self.assertEqual(["qiskit_test"], rresult.tags)
        self.assertEqual(ResultQuality.GOOD, rresult.quality)
        self.assertTrue(rresult.verified)
        self.assertEqual(chisq, rresult.chisq)

    def test_figure(self):
        """Test getting a figure."""
        exp_id = self.service.create_or_update_experiment(
            DbExperimentData(experiment_type="test_experiment", backend="ibmq_qasm_simulator")
        ).experiment_id
        hello_bytes = str.encode("hello world")
        figure_name = "hello.svg"
        self.service.create_or_update_figure(
            experiment_id=exp_id, figure=hello_bytes, figure_name=figure_name
        )
        fig = self.service.figure(exp_id, figure_name)
        self.assertEqual(fig, hello_bytes)
        hello_bytes = str.encode("hello world version 2")
        self.service.create_or_update_figure(
            experiment_id=exp_id,
            figure=hello_bytes,
            figure_name=figure_name,
            create=False,
        )
        fig = self.service.figure(exp_id, figure_name)
        self.assertEqual(fig, hello_bytes)

    def test_files(self):
        """Test upload and download of files"""
        exp_id = self.service.create_or_update_experiment(
            DbExperimentData(experiment_type="test_experiment", backend="ibmq_qasm_simulator")
        ).experiment_id
        hello_data = {"hello": "world", "foo": "bar"}
        filename = "test_file.json"
        self.service.file_upload(exp_id, filename, hello_data)
        rfile_data = self.service.file_download(exp_id, filename)
        self.assertEqual(hello_data, rfile_data)
        self.assertTrue(self.service.experiment_has_file(exp_id, filename))
        file_list = self.service.files(exp_id)["files"]
        self.assertEqual(len(file_list), 1)
        self.assertEqual(file_list[0]["Key"], filename)

        exp_id2 = self.service.create_or_update_experiment(
            DbExperimentData(experiment_type="test_experiment", backend="ibmq_qasm_simulator")
        ).experiment_id
        file_list = self.service.files(exp_id2)["files"]
        self.assertEqual(len(file_list), 0)

    def test_server_setting_start_time(self):
        """Tests that start time is initialized by the server unless already present"""
        ref_start_dt = datetime.now(timezone.utc)
        exp_id = self.service.create_or_update_experiment(
            DbExperimentData(
                experiment_type="qiskit_time_test",
                backend="ibmq_qasm_simulator",
            )
        ).experiment_id
        experiments = self.service.experiments()
        found = False
        for exp_id in experiments:
            exp = self.service.experiment(exp_id)
            if exp.experiment_id == exp_id:
                found = True
        self.assertTrue(found)
        self.assertGreaterEqual(exp.start_datetime, ref_start_dt)

    def test_file_upload_formats(self):
        """Test file upload/download for JSON and YAML formats"""
        exp_id = self._create_experiment()
        data = {"string": "b-string", "int": 10, "float": 0.333}
        yaml_data = yaml.dump(data)
        json_data = json.dumps(data)
        yaml_filename = "data.yaml"
        json_filename = "data.json"

        self.service.file_upload(exp_id, json_filename, json_data)
        rjson_data = self.service.file_download(exp_id, json_filename)
        self.assertEqual(data, rjson_data)

        self.service.file_upload(exp_id, yaml_filename, yaml_data)
        ryaml_data = self.service.file_download(exp_id, yaml_filename)
        self.assertEqual(data, ryaml_data)
        file_list = self.service.files(exp_id)["files"]
        self.assertEqual(len(file_list), 2)

    def test_save_to_disk(self):
        """Test round trip of data to disk"""
        # Make an experiemnt, add result, add figure, add json, add artifact (zip)
        # Read it all back
        data = DbExperimentData(
            experiment_type="test_experiment",
            backend="ibmq_qasm_simulator",
            metadata={"float_data": 3.14, "string_data": "foo"},
        )

        result_value = {"str": "foo", "float": 3.14}
        result_data = DbAnalysisResultData(
            experiment_id=data.experiment_id,
            result_data=result_value,
            result_type="qiskit_test",
        )

        figure_data = b"figure_data"
        figure_name = "figure.svg"

        file_data = {"string": "b-string", "int": 10, "float": 0.333}
        yaml_filename = "data.yaml"
        json_filename = "data.json"

        zip_data = b"zip_data"
        zip_name = "data.zip"

        with TemporaryDirectory() as tmpdirname:
            save_service = LocalExperimentService(db_dir=tmpdirname)
            save_service.create_or_update_experiment(data)
            save_service.create_or_update_analysis_result(result_data)
            save_service.create_or_update_figure(data.experiment_id, figure_data, figure_name)
            save_service.file_upload(data.experiment_id, json_filename, file_data)
            save_service.file_upload(data.experiment_id, yaml_filename, file_data)
            save_service.file_upload(data.experiment_id, zip_name, zip_data)

            load_service = LocalExperimentService(db_dir=tmpdirname)
            load_data = load_service.experiment(data.experiment_id)
            load_result = load_service.analysis_result(result_data.result_id)
            load_figure = load_service.figure(data.experiment_id, figure_name)
            load_json = load_service.file_download(data.experiment_id, json_filename)
            load_yaml = load_service.file_download(data.experiment_id, yaml_filename)
            load_zip = load_service.file_download(data.experiment_id, zip_name)

        # Filter values because service can insert dates and change empty values' types
        data_dict = {k: v for k, v in asdict(data).items() if v}
        load_data_dict = {k: v for k, v in asdict(load_data).items() if k in data_dict}
        self.assertDictEqual(data_dict, load_data_dict)

        result_data_dict = {k: v for k, v in asdict(result_data).items() if v}
        load_result_data_dict = {
            k: v for k, v in asdict(load_result).items() if k in result_data_dict
        }
        self.assertEqual(result_data_dict, load_result_data_dict)

        self.assertEqual(figure_data, load_figure)
        self.assertEqual(file_data, load_json)
        self.assertEqual(file_data, load_yaml)
        self.assertEqual(zip_data, load_zip)

    def _create_experiment(
        self,
        experiment_type: str | None = None,
        json_encoder: json.JSONEncoder | None = None,
        **kwargs,
    ) -> str:
        """Create a new experiment."""
        experiment_type = experiment_type or "qiskit_test"
        exp_id = self.service.create_or_update_experiment(
            DbExperimentData(
                experiment_type=experiment_type,
                **kwargs,
            ),
            json_encoder=json_encoder,
        ).experiment_id
        return exp_id

    def _create_analysis_result(
        self,
        exp_id: str | None = None,
        result_type: str | None = None,
        result_data: dict | None = None,
        json_encoder: json.JSONEncoder | None = None,
        **kwargs: Any,
    ):
        """Create a simple analysis result."""
        experiment_id = exp_id or self._create_experiment()
        result_type = result_type or "qiskit_test"
        result_data = result_data or {}
        aresult_id = self.service.create_or_update_analysis_result(
            DbAnalysisResultData(
                experiment_id=experiment_id,
                result_data=result_data,
                result_type=result_type,
                **kwargs,
            ),
            json_encoder=json_encoder,
        )
        return aresult_id


if __name__ == "__main__":
    unittest.main()
