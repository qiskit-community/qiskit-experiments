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
import os
import json
import numpy as np
from qiskit.quantum_info.operators.predicates import matrix_equal
from qiskit.test import QiskitTestCase
import qiskit_experiments as qe


class TestStandardRBAnalysis(QiskitTestCase):
    """
    A test for the analysis of the RB experiment
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
        expdata1 = qe.ExperimentData()
        self.assertTrue(
            os.path.isfile(rb_exp_data_file_name),
            "The file containing the experiment data doesn't exist."
            " Please run the data generator.",
        )
        gate_error_ratio = {
            ((0,), "id"): 1,
            ((0,), "rz"): 0,
            ((0,), "sx"): 1,
            ((0,), "x"): 1,
            ((0, 1), "cx"): 1,
        }
        with open(rb_exp_data_file_name, "r") as json_file:
            data = json.load(json_file)
            # The experiment attributes added
            exp_attributes = data[0]
            # pylint: disable=protected-access, invalid-name
            expdata1._metadata = data[0]
            # The experiment data located in index [1] as it is a list of dicts
            expdata1.add_data(data[1])
        rb_class = qe.randomized_benchmarking
        rb_exp = rb_class.StandardRB(
            exp_attributes["physical_qubits"],
            exp_attributes["lengths"],
            num_samples=exp_attributes["num_samples"],
            seed=exp_attributes["seed"],
        )
        rb_exp.set_analysis_options(gate_error_ratio=gate_error_ratio)
        analysis_results = rb_exp.run_analysis(expdata1)
        return data, analysis_results

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
        samples_analysis_list = []
        with open(analysis_file_path, "r") as expected_results_file:
            analysis_data_experiments = json.load(expected_results_file)
        for analysis_data_experiment in analysis_data_experiments:
            analysis_data_experiment["popt"] = np.array(analysis_data_experiment["popt"])
            analysis_data_experiment["popt_err"] = np.array(analysis_data_experiment["popt_err"])
            for idx, item in enumerate(analysis_data_experiment["pcov"]):
                analysis_data_experiment["pcov"][idx] = np.array(item)
            analysis_data_experiment["pcov"] = np.array(analysis_data_experiment["pcov"])
            samples_analysis_list.append(analysis_data_experiment)
        return samples_analysis_list

    def _validate_fitting_parameters(
        self, calculated_analysis_samples_data: list, expected_analysis_samples_data: list
    ):
        """
        The function checking that the results of the analysis matches to the expected one.
        Args:
            calculated_analysis_samples_data(list): list of dictionary containing the
            analysis result.
            expected_analysis_samples_data(list): list of dictionary containing the analysis
                expected result.
        """
        keys_for_array_data = ["popt", "popt_err", "pcov", "xrange"]
        keys_for_string_data = ["popt_keys", "analysis_type"]
        for idx, calculated_analysis_sample_data in enumerate(calculated_analysis_samples_data):
            for key in calculated_analysis_sample_data:
                if key in keys_for_array_data:
                    self.assertTrue(
                        matrix_equal(
                            calculated_analysis_sample_data[key],
                            expected_analysis_samples_data[idx][key],
                        ),
                        "The calculated value for the key '"
                        + key
                        + "', doesn't match the expected value.",
                    )
                else:
                    if key in keys_for_string_data:
                        self.assertTrue(
                            calculated_analysis_sample_data[key]
                            == expected_analysis_samples_data[idx][key],
                            "The analysis_type doesn't match to the one expected.",
                        )
                    else:
                        if key == "EPG":
                            self._validate_epg(
                                calculated_analysis_sample_data[key],
                                expected_analysis_samples_data[idx][key],
                            )
                        else:
                            self.assertAlmostEqual(
                                np.float64(calculated_analysis_sample_data[key]),
                                np.float64(expected_analysis_samples_data[idx][key]),
                                msg="The calculated value for key '"
                                + key
                                + "', doesn't match the expected value.",
                            )

    def _validate_epg(self, calculated_epg_dict: dict, expected_epg_dict: dict):
        """
        Confirm that the EPG that is calculated is the same as the expected one.
        The attributes are dictionaries of the form (qubits, gate) -> value where value
        is the epg for the given gate on the specified qubits
        Args:
            calculated_epg_dict: Dictionary of the calculated EPG
            expected_epg_dict: Dictionary of the expected EPG
        """
        for physical_qubit in calculated_epg_dict.keys():
            for epg_key, epg_value in calculated_epg_dict[physical_qubit].items():
                self.assertAlmostEqual(
                    np.float64(epg_value),
                    expected_epg_dict[str(physical_qubit)][epg_key],
                    msg="The calculated value for EPG for qubit '"
                    + str(physical_qubit)
                    + "' and key '"
                    + str(epg_key)
                    + "', doesn't match the expected value.",
                )

    def test_standard_rb_analysis_test(self):
        """
        A function to validate the data that is stored and the jsons and
        check that the analysis is correct.
        """
        dir_name = os.path.dirname(os.path.abspath(__file__))
        rb_exp_data_file_names = [
            "rb_standard_1qubit_output_data.json",
            "rb_standard_2qubits_output_data.json",
        ]
        rb_exp_analysis_file_names = [
            "rb_standard_1qubit_output_analysis.json",
            "rb_standard_2qubits_output_analysis.json",
        ]
        for idx, rb_exp_data_file_name in enumerate(rb_exp_data_file_names):
            json_data, analysis_obj = self._load_rb_data(
                os.path.join(dir_name, rb_exp_data_file_name)
            )
            # experiment_setup is the attributes passed to the experiment while
            # experiment_data is the data of the experiment that was simulated
            experiment_setup, experiment_data = json_data[0], json_data[1]
            self._validate_metadata(analysis_obj.data(), experiment_setup)
            self._validate_counts(analysis_obj.data(), experiment_data)
            analysis_data_expected = self._analysis_load(
                os.path.join(dir_name, rb_exp_analysis_file_names[idx])
            )
            self._validate_fitting_parameters(
                analysis_obj.analysis_result(None), analysis_data_expected
            )
