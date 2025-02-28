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

"""A mock IQ backend for testing."""
import datetime
from typing import Sequence, List, Tuple, Dict, Union, Any

import numpy as np

from qiskit import QuantumCircuit
from qiskit.circuit import Gate
from qiskit.result import Result
from qiskit.providers import BackendV2, Provider, convert_to_target
from qiskit.providers.fake_provider import FakeOpenPulse2Q
from qiskit.qobj.utils import MeasLevel

from qiskit_experiments.exceptions import QiskitError
from qiskit_experiments.framework import Options
from qiskit_experiments.test.utils import FakeJob
from qiskit_experiments.test.mock_iq_helpers import (
    MockIQExperimentHelper,
    MockIQParallelExperimentHelper,
    IQPoint,
)


class FakeOpenPulse2QV2(BackendV2):
    """BackendV2 conversion of qiskit.providers.fake_provider.FakeOpenPulse2Q"""

    def __init__(
        self,
        provider: Provider = None,
        name: str = None,
        description: str = None,
        online_date: datetime.datetime = None,
        backend_version: str = None,
        **fields,
    ):
        super().__init__(provider, name, description, online_date, backend_version, **fields)

        backend_v1 = FakeOpenPulse2Q()
        # convert_to_target requires the description attribute
        backend_v1._configuration.description = "A fake test backend with pulse defaults"

        self._target = convert_to_target(
            backend_v1.configuration(),
            backend_v1.properties(),
            backend_v1.defaults(),
            add_delay=True,
        )
        # See commented out defaults() method below
        self._defaults = backend_v1._defaults

    # This method is not defined in the base class as we would like to avoid
    # relying on it as much as necessary. Individual tests should add it when
    # necessary.
    # def defaults(self):
    #     """Pulse defaults"""
    #     return self._defaults

    @property
    def max_circuits(self):
        return 300

    @property
    def target(self):
        return self._target


class MockIQBackend(FakeOpenPulse2QV2):
    """A mock backend for testing with IQ data."""

    def __init__(
        self,
        experiment_helper: MockIQExperimentHelper = None,
        rng_seed: int = 0,
    ):
        """
        Initialize the backend.

        Args:
            experiment_helper(MockIQExperimentHelper): Experiment helper class that contains
                :meth:`~MockIQExperimentHelper.compute_probabilities` and
                :meth:`~MockIQExperimentHelper.iq_phase` methods for the backend to execute.
            rng_seed(int): The random seed value.
        """

        self._experiment_helper = experiment_helper
        self._rng = np.random.default_rng(rng_seed)
        self.simulator = True

        super().__init__()

    @classmethod
    def _default_options(cls):
        """Default options of the test backend."""
        return Options(
            shots=1024,
            meas_level=MeasLevel.KERNELED,
            meas_return="single",
        )

    @property
    def experiment_helper(self):
        """return the 'experiment_helper' attribute"""
        return self._experiment_helper

    @experiment_helper.setter
    def experiment_helper(self, value):
        """
        Setter for the experiment helper.
        Args:
            value(MockIQExperimentHelper): The helper for the backend to use for generating IQ shots.

        Raises:
            ValueError: Raised if the value to set is not of type `MockIQExperimentHelper`
        """
        cls = MockIQExperimentHelper
        if not isinstance(value, cls):
            raise ValueError(
                f"The input type is {str(type(value))} while the type expected type is "
                f"<{str(type(cls()))}>."
            )
        self._experiment_helper = value

    @staticmethod
    def _verify_parameters(output_length: int, prob_dict: Dict[str, float]):
        if output_length < 1:
            raise ValueError(f"The output length {output_length} is smaller than 1.")

        if not np.allclose(1, sum(prob_dict.values())):
            raise ValueError("The probabilities given don't sum up to 1.")
        for key in prob_dict.keys():
            if output_length is not len(key):
                raise ValueError(
                    "The output lengths of the circuit and the output lengths in the dictionary"
                    " don't match."
                )

    def _get_normal_samples_for_shot(
        self,
        qubits: Sequence[int],
    ) -> np.ndarray:
        """
        Produce a list in the size of num_qubits. Each entry value is produced from normal distribution
        with expected value of '0' and standard deviation of 1. The intention is that these samples are
        scaled by :func:`_scale_samples_for_widths` for various circuits, experiments, and their IQ
        widths; removing the need to query a RNG for each new width list.

        Example:
            .. code-block::
                # Generate template data
                template_iq_data = [np.nan] * shots
                for i_shot in range(n_shots):
                    real_data = self._get_normal_samples_for_shot(qubits)
                    imag_data = self._get_normal_samples_for_shot(qubits)
                    template_iq_data[i_shot] = np.array([real_data, imag_data], dtype="float").T

                # Scale template data to separate widths
                iq_data_1 = self._scale_samples_for_widths(template_iq_data, widths_1)
                iq_data_2 = self._scale_samples_for_widths(template_iq_data, widths_2)

                # IQ data should then be indexed randomly so that repeated usage does not give the same
                # order of samples.
                iq_data_circuit_1 = iq_data_1[random_indices_1]
                iq_data_circuit_2a = iq_data_2[random_indices_2a]
                iq_data_circuit_2b = iq_data_2[random_indices_2b]

        Args:
            num_qubits: The number of qubits in the circuit.

        Returns:
            Ndarray: A numpy array with values that were produced from normal distribution.
        """
        samples = [self._rng.normal(0, 1, size=1) for qubit in qubits]
        # we squeeze the second dimension because samples is List[qubit_number][0][0\1] = I\Q
        # and we want to change it to be List[qubit_number][0\1]
        return np.squeeze(np.array(samples), axis=1)

    def _scale_samples_for_widths(
        self, samples: List[np.ndarray], widths: List[float]
    ) -> List[np.ndarray]:
        """Scales `samples` by `widths` so that the data has the necessary std-dev.

        `samples` contains `n_shots` elements, each being :math:`n\times{}2` float values, representing
        the I and Q values for :math:`n` qubits. `widths` is a list of :math:`n` standard-deviations for
        each qubit. The IQ values for each list element in `samples` is scaled by the values in `widths`,
        for their respective qubits. It is assumed that the standard deviation of `samples` is :math:`1`.

        Args:
            samples: List of np.ndarrays containing random IQ samples for n qubits.
            widths: List of widths/standard-deviations to scale the data by.

        Returns:
            List: A list of samples with standard-deviations matching `widths`.
        """
        return [circ_samples * np.tile(widths, (2, 1)).T for circ_samples in samples]

    def _probability_dict_to_probability_array(
        self, prob_dict: Dict[str, float], num_qubits: int
    ) -> List[float]:
        prob_list = [0] * (2**num_qubits)
        for output_str, probability in prob_dict.items():
            index = int(output_str, 2)
            prob_list[index] = probability
        return prob_list

    def _draw_iq_shots(
        self,
        prob: List[float],
        shots: int,
        circ_qubits: Sequence[int],
        iq_cluster_centers: List[Tuple[IQPoint, IQPoint]],
        iq_cluster_width: List[float],
        phase: float = 0.0,
    ) -> List[List[List[Union[float, complex]]]]:
        """
        Produce an IQ shot.

        Args:
            prob: A list of probabilities for each output.
            shots: The number of times the circuit will run.
            circ_qubits: The qubits of the circuit.
            iq_cluster_centers: A list of tuples containing the clusters' centers in the IQ plane. There
            are different centers for different logical values of the qubit.
            iq_cluster_width: A list of standard deviation values for the sampling of each qubit.
            phase: The added phase needed to apply to the shot data.
        Returns:
            List[List[Tuple[float, float]]]: A list of shots. Each shot consists of a list of qubits.
            The qubits are tuples with two values [I,Q].
            The output structure is  - List[shot index][qubit index] = [I,Q]
        """
        # Randomize samples (width=1)
        qubits_iq_template_rand = [np.nan] * shots
        for shot in range(shots):
            rand_i = self._get_normal_samples_for_shot(circ_qubits)
            rand_q = self._get_normal_samples_for_shot(circ_qubits)
            qubits_iq_template_rand[shot] = np.array([rand_i, rand_q], dtype="float").T

        # Scale samples to use iq_cluster_width.
        exp_widths = [iq_cluster_width[i_qubit] for i_qubit in circ_qubits]
        qubits_iq_rand = self._scale_samples_for_widths(qubits_iq_template_rand, exp_widths)

        memory = []
        shot_num = 0

        for output_number, number_of_occurrences in enumerate(
            self._rng.multinomial(shots, prob, size=1)[0]
        ):
            state_str = str(format(output_number, "b").zfill(len(circ_qubits)))
            for _ in range(number_of_occurrences):
                shot_memory = []
                # the iteration on the string variable state_str starts from the MSB. For readability,
                # we will reverse the string so the loop will run from the LSB to MSB.
                for iq_center, qubit_iq_rand_sample, char_qubit in zip(
                    iq_cluster_centers, qubits_iq_rand[shot_num], state_str[::-1]
                ):
                    # The structure of iq_cluster_centers is [qubit_number][logic_result][I/Q].
                    i_center = iq_center[int(char_qubit)][0]
                    q_center = iq_center[int(char_qubit)][1]

                    point_i = i_center + qubit_iq_rand_sample[0]
                    point_q = q_center + qubit_iq_rand_sample[1]

                    # Adding phase if not 0.0
                    if not np.allclose(phase, 0.0):
                        complex_iq = (point_i + 1.0j * point_q) * np.exp(1.0j * phase)
                        point_i, point_q = np.real(complex_iq), np.imag(complex_iq)

                    shot_memory.append([point_i, point_q])
                # We proceed to the next occurrence - meaning it's a new shot.
                memory.append(shot_memory)
                shot_num += 1

        return memory

    def _generate_data(
        self, prob_dict: Dict[str, float], circuit: QuantumCircuit
    ) -> Dict[str, Any]:
        """
        Generate data for the circuit.

        Args:
            prob_dict: A dictionary whose keys are strings representing the output vectors and
            their values are the probability to get the output in this circuit.
            circuit: The circuit that needs to be simulated.

        Returns:
            A dictionary that's filled with the simulated data. The output format is different between
            measurement level 1 and measurement level 2.
        """
        # The output is proportional to the number of classical bit.
        output_length = int(np.sum([creg.size for creg in circuit.cregs]))
        self._verify_parameters(output_length, prob_dict)
        prob_arr = self._probability_dict_to_probability_array(prob_dict, output_length)
        shots = self.options.get("shots")
        meas_level = self.options.get("meas_level")
        meas_return = self.options.get("meas_return")
        run_result = {}

        if meas_level == MeasLevel.CLASSIFIED:
            counts = {}
            results = self._rng.multinomial(shots, prob_arr, size=1)[0]
            for result, num_occurrences in enumerate(results):
                result_in_str = str(format(result, "b").zfill(output_length))
                counts[result_in_str] = num_occurrences
            run_result["counts"] = counts
            if meas_return == "single" or self.options.get("memory"):
                run_result["memory"] = [
                    format(result, "x") for result, num in enumerate(results) for _ in range(num)
                ]
        else:
            # Phase has meaning only for IQ shot, so we calculate it here
            phase = self.experiment_helper.iq_phase([circuit])[0]
            iq_cluster_centers, iq_cluster_width = self.experiment_helper.iq_clusters([circuit])[0]

            # 'circ_qubits' get a list of all the qubits
            memory = self._draw_iq_shots(
                prob_arr,
                shots,
                list(range(output_length)),
                iq_cluster_centers,
                iq_cluster_width,
                phase,
            )
            if meas_return == "avg":
                memory = np.average(np.array(memory), axis=0).tolist()

            run_result["memory"] = memory
        return run_result

    def run(self, run_input: List[QuantumCircuit], **run_options) -> FakeJob:
        """
        Run the IQ backend.

        Args:
            run_input: A list of QuantumCircuit for which the backend will generate
                data.
            **run_options: Experiment running options. The options that are supported
                in this backend are `meas_level`, `meas_return` and `shots`:

                * meas_level: To generate data in the IQ plane, `meas_level` should be
                  assigned 1 or ``MeasLevel.KERNELED``. If `meas_level` is 2 or
                  ``MeasLevel.CLASSIFIED``, the generated data will be in the form
                  of `counts`.
                * meas_return: This option will only take effect if `meas_level` =
                  ``MeasLevel.CLASSIFIED``. It can get either
                  ``MeasReturnType.AVERAGE`` or ``MeasReturnType.SINGLE``. For
                  ``MeasReturnType.SINGLE``, the data of each shot will be stored in
                  the result. For ``MeasReturnType.AVERAGE``, an average of all the
                  shots will be calculated and stored in the result.
                * shots: The number of times the circuit will run.

        Returns:
            FakeJob: A job that contains the simulated data.

        Raises:
            QiskitError: Raised if the user try to run the experiment without setting a helper.
        """

        if not self.experiment_helper:
            raise QiskitError("The backend `experiment_helper` attribute cannot be 'None'.")

        self.options.update_options(**run_options)
        shots = self.options.get("shots")
        meas_level = self.options.get("meas_level")

        result = {
            "backend_name": f"{self.__class__.__name__}",
            "backend_version": "0",
            "qobj_id": "0",
            "job_id": "0",
            "success": True,
            "results": [],
        }
        prob_list = self.experiment_helper.compute_probabilities(run_input)
        for prob, circ in zip(prob_list, run_input):
            run_result = {
                "shots": shots,
                "success": True,
                "header": {"metadata": circ.metadata},
                "meas_level": meas_level,
            }

            run_result["data"] = self._generate_data(prob, circ)
            result["results"].append(run_result)

        return FakeJob(self, Result.from_dict(result))


class MockIQParallelBackend(MockIQBackend):
    """A mock backend for testing parallel experiments with IQ data."""

    def __init__(
        self,
        experiment_helper: MockIQParallelExperimentHelper = None,
        rng_seed: int = 0,
    ):
        """
        Initialize the backend.

        Args:
            experiment_helper: Parallel experiment helper class that contains
                helper classes for each experiment.
            rng_seed: The random seed value.
        """
        super().__init__(experiment_helper, rng_seed)

    @property
    def experiment_helper(self):
        """return the 'experiment_helper' attribute"""
        return self._experiment_helper

    @experiment_helper.setter
    def experiment_helper(self, value):
        """
        Setter for the experiment helper.

        Args:
            value(MockIQParallelExperimentHelper): The helper for the backend to use for generating IQ
             shots.

        Raises:
            ValueError: Raised if the value to set is not of type `MockIQExperimentHelper`
        """
        cls = MockIQParallelExperimentHelper
        if not isinstance(value, cls):
            raise ValueError(
                f"The input type is {str(type(value))} while the type expected type is <{str(cls)}>."
            )
        self._experiment_helper = value

    def _parallel_draw_iq_shots(
        self,
        list_exp_dict: List[Dict[str, Union[List, int]]],
        shots: int,
        circ_qubits: List[int],
        circ_idx: int,
    ) -> List[List[List[Union[float, complex]]]]:
        """
        Produce an IQ shot.
        Args:
            list_exp_dict: A list of dictionaries for each experiment. It is determined by the
                ``MockIQParallelExperimentHelper`` object provided to the backend.
            shots: The number of times the circuit will run.
            circ_qubits: List of qubits that are used in this circuit.
            circ_idx: The circuit index.

        Returns:
            List[List[Tuple[float, float]]]: A list of shots. Each shot consists of a list of qubits.
            The qubits are tuples with two values [I,Q].
            The output structure is  - List[shot index][qubit index] = [I,Q]
        """
        # Randomize samples (width=1)
        qubits_iq_template_rand = [np.nan] * shots
        for shot in range(shots):
            rand_i = self._get_normal_samples_for_shot(circ_qubits)
            rand_q = self._get_normal_samples_for_shot(circ_qubits)
            qubits_iq_template_rand[shot] = np.array([rand_i, rand_q], dtype="float").T

        memory = [[] for _ in range(shots)]

        # The use of idx_shift is to sample 'qubits_iq_rand' correctly
        sample_idx_shift = 0

        # The code generates data as follows:
        # for each experiment, it first checks if it needs to generate data for it. If it does, then the
        # multinomial probability function draws lots for all the shots, and we store this data in the
        # corresponding position in the output list. After that we move on to the next experiment.
        for exp_dict in list_exp_dict:
            # skipping experiments that don't need data generation for this circuit.
            if exp_dict["num_circuits"] <= circ_idx:
                continue

            qubits = list(exp_dict["physical_qubits"])
            prob = self._probability_dict_to_probability_array(
                exp_dict["prob"][circ_idx], len(qubits)
            )
            phase = exp_dict["phase"][circ_idx]
            iq_centers = exp_dict["centers"][circ_idx]
            iq_widths = exp_dict["widths"][circ_idx]
            exp_widths = [iq_widths[i_qubit] for i_qubit in circ_qubits]

            # Rescale samples to appropriate width for the given parallel circuits
            qubits_iq_rand = self._scale_samples_for_widths(qubits_iq_template_rand, exp_widths)

            shot_num = 0

            for output_number, number_of_occurrences in enumerate(
                self._rng.multinomial(shots, prob, size=1)[0]
            ):
                state_str = str(format(output_number, "b").zfill(len(qubits)))
                for _ in range(number_of_occurrences):
                    # the iteration on the string variable state_str starts from the MSB. For
                    # readability, we will reverse the string so the loop will run from the LSB to MSB.
                    for qubit_idx, qubit, char_qubit in zip(
                        range(len(qubits)), qubits, state_str[::-1]
                    ):

                        i_center = iq_centers[qubit][int(char_qubit)][0]
                        q_center = iq_centers[qubit][int(char_qubit)][1]

                        # we use 'sample_idx_shift' to take the sample corresponding to the current qubit
                        # in 'qubits_iq_rand[shot_num]'.
                        point_i = (
                            i_center + qubits_iq_rand[shot_num][qubit_idx + sample_idx_shift][0]
                        )
                        point_q = (
                            q_center + qubits_iq_rand[shot_num][qubit_idx + sample_idx_shift][1]
                        )

                        # Adding phase if not 0.0
                        if not np.allclose(phase, 0.0):
                            complex_iq = (point_i + 1.0j * point_q) * np.exp(1.0j * phase)
                            point_i, point_q = np.real(complex_iq), np.imag(complex_iq)

                        memory[shot_num].append([point_i, point_q])
                    shot_num += 1

            sample_idx_shift = sample_idx_shift + len(qubits)
        return memory

    def _parallel_generate_data(
        self,
        list_exp_dict: List[Dict[str, Union[List, int]]],
        circ_idx: int,
    ) -> Dict[str, Any]:
        """
        Generate data for the circuit.
        Args:
            list_exp_dict (List): A List of dictionaries, each dictionary contains data of an experiment.
            circ_idx (int): The circuit number we simulate.

        Returns:
            A dictionary that's filled with the simulated data.

        Raises:
            QiskitError: Raising an error if in the experiment running option, classified data is
            requested.
        """
        circ_qubit_list = []
        for exp_dict in list_exp_dict:
            if circ_idx < exp_dict["num_circuits"]:
                circ_qubit_list = circ_qubit_list + list(exp_dict["physical_qubits"])

        shots = self.options.get("shots")
        meas_level = self.options.get("meas_level")
        meas_return = self.options.get("meas_return")
        run_result = {}

        if meas_level == MeasLevel.KERNELED:
            memory = self._parallel_draw_iq_shots(list_exp_dict, shots, circ_qubit_list, circ_idx)
            if meas_return == "avg":
                memory = np.average(np.array(memory), axis=0).tolist()

            run_result["memory"] = memory
        else:
            # The backend doesn't currently support 'meas_level = MeasLevel.CLASSIFIED'.
            raise QiskitError("Classified data generator isn't supported for this backend")

        return run_result

    def run(self, run_input: List[QuantumCircuit], **run_options) -> FakeJob:
        """
        Run the IQ backend.

        Args:
            run_input: A list of QuantumCircuit for which the backend will generate
                data.
            **run_options: Experiment running options. The options that are supported
                in this backend are `meas_level`, `meas_return` and `shots`:

                * meas_level: To generate data in the IQ plane, `meas_level` should be
                  assigned 1 or ``MeasLevel.KERNELED``. If `meas_level` is 2 or
                  ``MeasLevel.CLASSIFIED``, the generated data will be in the form
                  of `counts`.
                * meas_return: This option will only take effect if `meas_level` =
                  ``MeasLevel.CLASSIFIED``. It can get either
                  ``MeasReturnType.AVERAGE`` or ``MeasReturnType.SINGLE``. For
                  ``MeasReturnType.SINGLE``, the data of each shot will be stored in
                  the result. For ``MeasReturnType.AVERAGE``, an average of all the
                  shots will be calculated and stored in the result.
                * shots: The number of times the circuit will run.

        Returns:
            FakeJob: A job that contains the simulated data.

        Raises:
            QiskitError: Raised if the user try to run the experiment without setting a helper.
        """

        if not self.experiment_helper:
            raise QiskitError("The backend `experiment_helper` attribute cannot be 'None'.")

        self.options.update_options(**run_options)
        shots = self.options.get("shots")
        meas_level = self.options.get("meas_level")

        result = {
            "backend_name": f"{self.__class__.__name__}",
            "backend_version": "0",
            "qobj_id": "0",
            "job_id": "0",
            "success": True,
            "results": [],
        }

        experiment_data_list = self.experiment_helper.compute_probabilities(run_input)
        for circ_idx, circ in enumerate(run_input):
            run_result = {
                "shots": shots,
                "success": True,
                "header": {"metadata": circ.metadata},
                "meas_level": meas_level,
            }

            run_result["data"] = self._parallel_generate_data(experiment_data_list, circ_idx)
            result["results"].append(run_result)

        return FakeJob(self, Result.from_dict(result))


class MockMultiStateBackend(FakeOpenPulse2QV2):
    """A mock backend for testing with multi-state IQ data.

    .. note::

        This backend does no simulation. It just looks for gates x, x12, and
        x23 and sets the qubit state to the highest possible based on the
        presence of these gates.
    """

    def __init__(
        self,
        iq_centers: list[complex],
        iq_noise: float = 0.1,
        state_noise: float = 0.0,
        rng_seed: int = 0,
    ):
        """
        Initialize the backend.

        Args:
            iq_centers: list of points in the complex plane corresponding to
                different qubit levels.
            iq_noise: Standard deviation of the normally distributed variation in
                output around the IQ centers.
            state_noise: Noise in the probability of the output state. For
                example, 0.2 for a circuit with an x would mean 0.8 probability
                of 1 and 0.2 probability 0 when iq_centers has length 2.
            rng_seed(int): The random seed value.
        """

        if len(iq_centers) > 4:
            raise ValueError("Only 4 qubit levels supported!")
        self.iq_centers = iq_centers
        self.iq_noise = iq_noise
        self.state_noise = state_noise
        self._rng = np.random.default_rng(rng_seed)
        self.simulator = True

        super().__init__()

        if "x" not in self.target:
            self.target.add_instruction(Gate("x", 1, []))
        self.target.add_instruction(Gate("x12", 1, []))
        self.target.add_instruction(Gate("x23", 1, []))

    @classmethod
    def _default_options(cls):
        """Default options of the test backend."""
        return Options(
            shots=1024,
            meas_level=MeasLevel.KERNELED,
            meas_return="single",
        )

    def compute_probabilities(self, circuits: List[QuantumCircuit]) -> List[List[float]]:
        """Return the probability of being in the various states for each circuit"""
        output_dict_list = []
        for circuit in circuits:
            ops = circuit.count_ops()
            if "x23" in ops:
                idx = 3
            elif "x12" in ops:
                idx = 2
            elif "x" in ops:
                idx = 1
            else:
                idx = 0

            probability_outputs = self._rng.random(len(self.iq_centers))
            probability_outputs[idx] = 0.0
            prob_sum = sum(probability_outputs)
            if prob_sum == 0:
                probability_outputs[(idx + 1) % 2] = self.state_noise
            else:
                probability_outputs = self.state_noise * probability_outputs / prob_sum

            probability_outputs[idx] = 1 - self.state_noise

            output_dict_list.append(probability_outputs.tolist())

        return output_dict_list

    @staticmethod
    def _verify_parameters(output_length: int, prob_list: List[float]):
        if output_length != 1:
            raise ValueError(
                f"The output length {output_length} is not 1 (only one measurement supported)."
            )

        if not np.allclose(1, sum(prob_list)):
            raise ValueError("The probabilities given don't sum up to 1.")

    def _draw_iq_shots(
        self,
        prob: List[float],
        shots: int,
    ) -> List[List[float]]:
        """
        Produce an IQ shot.

        Args:
            prob: A list of probabilities for each output.
            shots: The number of times the circuit will run.
        Returns:
            List[List[List[float]]]: A list of shots. Each shot consists of a
            list of qubits (with 1 qubit only). The qubits are lists with two
            values [I,Q]. The output structure is

                List[shot index][qubit index][I,Q]
        """
        # Randomize samples (width=1)
        samples = self.iq_noise * self._rng.normal(0, 1, size=(shots, 2))
        samples = samples[:, 0] + 1j * samples[:, 1]
        samples = samples + self._rng.choice(self.iq_centers, size=(shots,), p=prob)
        memory = [[[np.real(s), np.imag(s)]] for s in samples]

        return memory

    def _generate_data(
        self, prob_list: Dict[str, float], circuit: QuantumCircuit
    ) -> Dict[str, Any]:
        """
        Generate data for the circuit.

        Args:
            prob_list: A list with probabilities for different qubit states
            circuit: The circuit that needs to be simulated.

        Returns:
            A dictionary that's filled with the simulated data. The output format is different between
            measurement level 1 and measurement level 2.
        """
        # The output is proportional to the number of classical bit.
        output_length = int(np.sum([creg.size for creg in circuit.cregs]))
        self._verify_parameters(output_length, prob_list)
        shots = self.options.get("shots")
        meas_return = self.options.get("meas_return")
        run_result = {}

        memory = self._draw_iq_shots(
            prob_list,
            shots,
        )
        if meas_return == "avg":
            memory = np.average(np.array(memory), axis=0).tolist()

        run_result["memory"] = memory
        return run_result

    def run(self, run_input: List[QuantumCircuit], **run_options) -> FakeJob:
        """
        Run the IQ backend.

        Args:
            run_input: A list of QuantumCircuit for which the backend will generate
                data.
            **run_options: Experiment running options. The options that are supported
                in this backend are `meas_level`, `meas_return` and `shots`:

                * meas_level: To generate data in the IQ plane, `meas_level` should be
                  assigned 1 or ``MeasLevel.KERNELED``. If `meas_level` is 2 or
                  ``MeasLevel.CLASSIFIED``, the generated data will be in the form
                  of `counts`.
                * meas_return: This option will only take effect if `meas_level` =
                  ``MeasLevel.CLASSIFIED``. It can get either
                  ``MeasReturnType.AVERAGE`` or ``MeasReturnType.SINGLE``. For
                  ``MeasReturnType.SINGLE``, the data of each shot will be stored in
                  the result. For ``MeasReturnType.AVERAGE``, an average of all the
                  shots will be calculated and stored in the result.
                * shots: The number of times the circuit will run.

        Returns:
            FakeJob: A job that contains the simulated data.

        Raises:
            QiskitError: Raised if the user try to run the experiment without setting a helper.
            ValueError: Raised if ``meas_level`` in ``run_options`` is not 1.
        """

        self.options.update_options(**run_options)
        shots = self.options.get("shots")
        meas_level = self.options.get("meas_level", 1)
        if meas_level != 1:
            raise ValueError("Only level 1 data supported!")

        result = {
            "backend_name": f"{self.__class__.__name__}",
            "backend_version": "0",
            "qobj_id": "0",
            "job_id": "0",
            "success": True,
            "results": [],
        }
        prob_list = self.compute_probabilities(run_input)
        for prob, circ in zip(prob_list, run_input):
            run_result = {
                "shots": shots,
                "success": True,
                "header": {"metadata": circ.metadata},
                "meas_level": meas_level,
            }

            run_result["data"] = self._generate_data(prob, circ)
            result["results"].append(run_result)

        return FakeJob(self, Result.from_dict(result))
