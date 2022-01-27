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
"""
A test for RB analysis. Using pre-Generated data from rb_generate_data.py.
"""
from test.base import QiskitExperimentsTestCase
import os
import json
import numpy as np
from qiskit.quantum_info.operators.predicates import matrix_equal

from qiskit.circuit.library import (
    XGate,
    CXGate,
)
from qiskit_experiments.framework import ExperimentData
from qiskit_experiments.library import StandardRB, InterleavedRB
from qiskit_experiments.framework import ExperimentDecoder, FitVal

ATOL_DEFAULT = 1e-2
RTOL_DEFAULT = 1e-5


class TestRBAnalysis(QiskitExperimentsTestCase):
    """
    A base class for the tests of analysis of the RB experiments
    """

    def _load_json_data(self, rb_exp_data_file_name: str):
        """
        loader for the experiment data and configuration setup.
        Args:
            rb_exp_data_file_name(str): The file name that contain the experiment data.
        Returns:
            list: containing dict of the experiment setup configuration and list of dictionaries
                containing the experiment results.
            ExperimentData:  ExperimentData object that was creates by the analysis function.
        """
        expdata1 = ExperimentData()
        self.assertTrue(
            os.path.isfile(rb_exp_data_file_name),
            "The file containing the experiment data doesn't exist."
            " Please run the data generator.",
        )
        with open(rb_exp_data_file_name, "r") as json_file:
            data = json.load(json_file)
            # The experiment attributes added
            exp_attributes = data[0]
            # pylint: disable=protected-access, invalid-name
            expdata1._metadata = data[0]
            # The experiment data located in index [1] as it is a list of dicts
            expdata1.add_data(data[1])

        return data, exp_attributes, expdata1

    def _analysis_load(self, analysis_file_path: str):
        """
        Loads the expected data of the analysis and changing the the values type
        to match the originals.
        Args:
            analysis_file_path(str): The full path of the json containing
            the expected analysis results.
        Returns:
            list(dict): A list of dicts which contains the analysis results.
        """
        self.assertTrue(
            os.path.isfile(analysis_file_path),
            "The file containing the experiment analysis data doesn't exist."
            " Please run the data generator.",
        )
        with open(analysis_file_path, "r") as expected_results_file:
            analysis_data = json.load(expected_results_file, cls=ExperimentDecoder)
        return analysis_data

    def _validate_counts(self, analysis_results_data: list, exp_data: list):
        """
        Function to check that the count statistics that is stored in the ExpirimentData object
        matches the data in the json file.
        Args:
            analysis_results_data(list): The data that is stored in the analysis object.
            exp_data(list): The setup data for the experiment.
        Returns(bool):
            return if the validation result.
        """
        for idx, exp_result in enumerate(analysis_results_data):
            # making a dict with all the shared keys with the same value
            shared_items = {
                k: exp_result["counts"][k]
                for k in exp_result
                if k in exp_data[idx]["counts"]
                and exp_result["counts"][k] == exp_data[idx]["counts"][k]
            }
            # check if all the keys and values are identical by length
            self.assertTrue(
                len(shared_items) != len(exp_data[idx]["counts"]),
                "The counts statistics doesn't match the data from the json.",
            )
            self.assertTrue(
                len(shared_items) != len(exp_result["counts"]),
                "The counts statistics doesn't match the data from the analytics.",
            )

    def _validate_metadata(self, analysis_results_data: list, exp_setup: list):
        """
        Function to check that the metadata that is stored in the ExpirimentData matches the
        metadata in the json file.
        Args:
            analysis_results_data(list): The data that is stored in the analysis object.
            exp_setup(list): The setup data for the experiment.
        Returns(bool):
            return if the validation result.
        """
        for exp_result in analysis_results_data:
            self.assertTrue(
                exp_result["metadata"]["xval"] in exp_setup["lengths"],
                "the gate sequence length isn't in the setup length list.",
            )

    def _validate_fitting_parameters(self, analysis_results: list, expected_analysis: list):
        """
        The function checking that the results of the analysis matches to the expected one.
        Args:
            analysis_results: list of experiment analysis results.
            expected_analysis: list of reference results as dicts.
        """
        for result, reference in zip(analysis_results, expected_analysis):
            # Check names match
            self.assertEqual(result.name, reference["name"])

            # Check values match
            value = result.value
            target = reference["value"]
            if isinstance(value, FitVal):
                value = value.value
                target = target.value
            if isinstance(value, float):
                self.assertAlmostEqual(value, target)
            elif isinstance(value, np.ndarray):
                self.assertTrue(matrix_equal(value, target))

            # Check extra match
            for key, value in result.extra.items():
                target = reference["extra"][key]
                if isinstance(value, FitVal):
                    value = value.value
                    target = target.value
                if isinstance(value, float):
                    self.assertAlmostEqual(value, target)
                elif isinstance(value, np.ndarray):
                    self.assertTrue(matrix_equal(value, target))

    def _run_tests(self, data_filenames, analysis_filenames):
        """
        A function to validate the data that is stored and the jsons and
        check that the analysis is correct.
        """
        dir_name = os.path.dirname(os.path.abspath(__file__))
        for rb_exp_data_file_name, rb_analysis_file_name in zip(data_filenames, analysis_filenames):
            json_data, analysis_obj = self._load_rb_data(
                os.path.join(dir_name, rb_exp_data_file_name)
            )
            # experiment_setup is the attributes passed to the experiment while
            # experiment_data is the data of the experiment that was simulated
            experiment_setup, experiment_data = json_data[0], json_data[1]
            self._validate_metadata(analysis_obj.data(), experiment_setup)
            self._validate_counts(analysis_obj.data(), experiment_data)
            analysis_results_expected = self._analysis_load(
                os.path.join(dir_name, rb_analysis_file_name)
            )
            self._validate_fitting_parameters(
                analysis_obj.analysis_results(), analysis_results_expected
            )

    def _load_rb_data(self, rb_exp_data_file_name: str):
        """
        loader for the experiment data and configuration setup.
        Args:
            rb_exp_data_file_name(str): The file name that contain the experiment data.
        Returns:
            list: containing dict of the experiment setup configuration and list of dictionaries
                containing the experiment results.
            ExperimentData:  ExperimentData object that was creates by the analysis function.
        """
        data, exp_attributes, expdata1 = self._load_json_data(rb_exp_data_file_name)
        rb_exp = StandardRB(
            exp_attributes["physical_qubits"],
            exp_attributes["lengths"],
            num_samples=exp_attributes["num_samples"],
            seed=exp_attributes["seed"],
        )
        gate_error_ratio = {
            ((0,), "id"): 1,
            ((0,), "rz"): 0,
            ((0,), "sx"): 1,
            ((0,), "x"): 1,
            ((0, 1), "cx"): 1,
        }
        rb_exp.analysis.set_options(gate_error_ratio=gate_error_ratio)
        analysis_results = rb_exp.analysis.run(expdata1)
        self.assertExperimentDone(analysis_results)
        return data, analysis_results


class TestStandardRBAnalysis(TestRBAnalysis):
    """
    Tests for the analysis of the standard RB experiment
    """

    def test_standard_rb_analysis_test(self):
        """Runs the standard RB analysis tests"""

        rb_exp_data_file_names = [
            "rb_standard_1qubit_output_data.json",
            "rb_standard_2qubits_output_data.json",
        ]
        rb_exp_analysis_file_names = [
            "rb_standard_1qubit_output_analysis.json",
            "rb_standard_2qubits_output_analysis.json",
        ]
        self._run_tests(rb_exp_data_file_names, rb_exp_analysis_file_names)


class TestInterleavedRBAnalysis(TestRBAnalysis):
    """
    A test for the analysis of the standard RB experiment
    """

    def _load_rb_data(self, rb_exp_data_file_name: str):
        """
        loader for the experiment data and configuration setup.
        Args:
            rb_exp_data_file_name(str): The file name that contain the experiment data.
        Returns:
            list: containing dict of the experiment setup configuration and list of dictionaries
                containing the experiment results.
            ExperimentData:  ExperimentData object that was creates by the analysis function.
        """
        interleaved_gates = {"x": XGate(), "cx": CXGate()}
        data, exp_attributes, expdata1 = self._load_json_data(rb_exp_data_file_name)
        rb_exp = InterleavedRB(
            interleaved_gates[exp_attributes["interleaved_element"]],
            exp_attributes["physical_qubits"],
            exp_attributes["lengths"],
            num_samples=exp_attributes["num_samples"],
            seed=exp_attributes["seed"],
        )
        gate_error_ratio = {
            ((0,), "id"): 1,
            ((0,), "rz"): 0,
            ((0,), "sx"): 1,
            ((0,), "x"): 1,
            ((0, 1), "cx"): 1,
        }
        rb_exp.analysis.set_options(gate_error_ratio=gate_error_ratio)
        analysis_results = rb_exp.analysis.run(expdata1)
        self.assertExperimentDone(analysis_results)
        return data, analysis_results

    def test_interleaved_rb_analysis_test(self):
        """Runs the standard RB analysis tests"""

        rb_exp_data_file_names = [
            "rb_interleaved_1qubit_output_data.json",
            "rb_interleaved_2qubits_output_data.json",
        ]
        rb_exp_analysis_file_names = [
            "rb_interleaved_1qubit_output_analysis.json",
            "rb_interleaved_2qubits_output_analysis.json",
        ]
        self._run_tests(rb_exp_data_file_names, rb_exp_analysis_file_names)
