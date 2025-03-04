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

"""Probability and phase functions for the mock IQ backend."""

from abc import abstractmethod
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from qiskit import QuantumCircuit
from qiskit.exceptions import QiskitError
from qiskit_aer import AerSimulator
from qiskit_experiments.framework import BaseExperiment

# Define an IQ point typing class.
IQPoint = Tuple[float, float]


class MockIQExperimentHelper:
    """Abstract class for the MockIQ helper classes.

    Different tests will use experiment specific helper classes which define the pattern
    of the IQ data that is then analyzed.
    """

    def __init__(
        self,
        iq_cluster_centers: Optional[List[Tuple[IQPoint, IQPoint]]] = None,
        iq_cluster_width: Optional[List[float]] = None,
    ):
        """Create a MockIQBackend helper object to define how the backend functions.

        :attr:`iq_cluster_centers` and :attr:`iq_cluster_width` define the base IQ
        cluster centers and standard deviations for each qubit in a
        :class:`MockIQBackend` instance. These are used by :meth:`iq_clusters` by
        default. Subclasses can override :meth:`iq_clusters` to return a modified
        version of :attr:`iq_cluster_centers` and :attr:`iq_cluster_width`.
        :attr:`iq_cluster_centers` is a list of tuples. For a given qubit ``i_qbt`` and
        computational state ``i_state`` (either `0` or `1`), the centers of the IQ
        clusters are found by indexing :attr:`iq_cluster_centers` as follows:

        .. code-block:: python

            iq_center = helper.iq_cluster_centers[i_qbt][i_state]
            center_inphase = iq_center[0]
            center_quadrature = iq_center[1]

        :attr:`iq_cluster_width` is indexed similarly except that there is only one width
        per qubit: i.e., the standard deviation of the IQ cluster for qubit ``i_qbt`` is

        .. code-block:: python

            iq_width = helper.iq_cluster_width[i_qbt]

        Subclasses must call ``super().__init__(iq_cluster_centers,iq_cluster_width)`` so that these
        properties are stored appropriately.

        Args:
            iq_cluster_centers: A list of tuples containing the clusters' centers in the IQ plane. There
                are different centers for different logical values of the qubit. Defaults to a single
                qubit with clusters in quadrants 1 and 3.
            iq_cluster_width: A list of standard deviation values for the sampling of each qubit.
                Defaults to widths of 1.0 for each qubit in :attr:`iq_cluster_centers`.

        """
        self._iq_cluster_centers = (
            iq_cluster_centers if iq_cluster_centers is not None else [((-1.0, -1.0), (1.0, 1.0))]
        )
        self._iq_cluster_width = (
            iq_cluster_width
            if iq_cluster_width is not None
            else [1.0] * len(self._iq_cluster_centers)
        )

    @property
    def iq_cluster_centers(self) -> List[Tuple[IQPoint, IQPoint]]:
        """The base cluster centers in the IQ plane."""
        return self._iq_cluster_centers

    @iq_cluster_centers.setter
    def iq_cluster_centers(self, iq_cluster_centers: List[Tuple[IQPoint, IQPoint]]):
        """Set the base cluster centers in the IQ plane."""
        self._iq_cluster_centers = iq_cluster_centers

    @property
    def iq_cluster_width(self) -> List[float]:
        """The base cluster widths in the IQ plane."""
        return self._iq_cluster_width

    @iq_cluster_width.setter
    def iq_cluster_width(self, iq_cluster_width: List[float]):
        """Set the base cluster widths."""
        self._iq_cluster_width = iq_cluster_width

    @abstractmethod
    def compute_probabilities(self, circuits: List[QuantumCircuit]) -> List[Dict[str, Any]]:
        """
        A function provided by the user which is used to determine the probability of each output of the
        circuit. The function returns a list of dictionaries, each containing output binary strings and
        their probabilities.

        Examples:

        **1 qubit circuit - excited state**

        In this experiment, we want to bring a qubit to its excited state and measure it.
        The circuit:

        .. parsed-literal::

                 ┌───┐┌─┐
            q:   ┤ X ├┤M├
                 └───┘└╥┘
            c: 1/══════╩═
                        0

        The function that calculates the probability for this circuit doesn't need any
        calculation parameters:

        .. code-block::

            @staticmethod
            def compute_probabilities(self, circuits: List[QuantumCircuit])
                -> List[Dict[str, float]]:

                output_dict_list = []
                for circuit in circuits:
                    probability_output_dict = {"1": 1.0, "0": 0.0}
                    output_dict_list.append(probability_output_dict)
                return output_dict_list

        **3 qubit circuit**

        In this experiment, we prepare a Bell state with the first and second qubit.
        In addition, we will bring the third qubit to its excited state.
        The circuit:

        .. parsed-literal::

                    ┌───┐     ┌─┐
            q_0:    ┤ H ├──■──┤M├───
                    └───┘┌─┴─┐└╥┘┌─┐
            q_1:    ─────┤ X ├─╫─┤M├
                    ┌───┐└┬─┬┘ ║ └╥┘
            q_2:    ┤ X ├─┤M├──╫──╫─
                    └───┘ └╥┘  ║  ║
            c:    3/═══════╩═══╩══╩═
                        2   0  1

        When an output string isn't in the probability dictionary, the backend will
        assume its probability is 0.

        .. code-block::

            @staticmethod
            def compute_probabilities(self, circuits: List[QuantumCircuit])
                -> List[Dict[str, float]]:

                output_dict_list = []
                for circuit in circuits:
                    probability_output_dict = {}
                    probability_output_dict["001"] = 0.5
                    probability_output_dict["111"] = 0.5
                    output_dict_list.append(probability_output_dict)
                return output_dict_list
        """

    # pylint: disable=unused-argument
    def iq_phase(self, circuits: List[QuantumCircuit]) -> List[float]:
        """Sub-classes can override this method to introduce a phase in the IQ plane.

        This is needed, to test the resonator spectroscopy where the point in the IQ
        plane has a frequency-dependent phase rotation.
        """
        return [0.0] * len(circuits)

    def iq_clusters(
        self,
        circuits: List[QuantumCircuit],
    ) -> List[Tuple[List[Tuple[IQPoint, IQPoint]], List[float]]]:
        """Returns circuit-specific IQ cluster centers and widths in the IQ plane.

        Subclasses can override this function to modify the centers and widths of IQ clusters based on
        the circuits being simulated by a :class:`MockIQBackend`. The base centers and widths are
        stored internally within the helper object, and can be set in :meth:`__init__` or by modifying
        :attr:`iq_cluster_centers` and :attr:`iq_cluster_width`. The default behavior for
        :meth:`iq_clusters` is to return the centers and widths unmodified for each circuit in
        ``circuits``. Subclasses may return different centers and widths based on the circuits provided.

        The returned list contains a tuple per circuit. Each tuple contains the IQ centers and widths in
        the same format as :attr:`iq_cluster_centers` and :attr:`iq_cluster_width`, passed as
        arguments to :meth:`__init__`. The format of the centers and widths lists, in the argument
        list and in the returned tuples, must match the format of :attr:`iq_cluster_centers` and
        :attr:`iq_cluster_width` in :func:`qiskit_experiments.test.MockIQExperimentHelper.__init__`.

        Args:
            circuits: The quantum circuits for which the clusters should be modified.

        Returns:
            List: A list of tuples containing the circuit-specific IQ centers and widths for the
                provided circuits.
        """
        return [(self.iq_cluster_centers, self.iq_cluster_width)] * len(circuits)


class MockIQParallelExperimentHelper(MockIQExperimentHelper):
    """Helper for Parallel experiment."""

    def __init__(
        self,
        exp_list: List[BaseExperiment],
        exp_helper_list: List[MockIQExperimentHelper],
    ):
        """
        Parallel Experiment Helper initializer. The class assumes `exp_helper_list` is ordered to
        match the corresponding experiment in `exp_list`.

        Note that :meth:`__init__` does not have :attr:`iq_cluster_centers` and :attr:`iq_cluster_width`
        as in :func:`MockIQExperimentHelper.__init__`. This is because the centers and widths for
        :class:`MockIQParallelBackend` are stored in multiple experiment helpers in the list
        `exp_helper_list`.

        Args:
            exp_list(List): List of experiments.
            exp_helper_list(List): Ordered list of :class:`.MockIQExperimentHelper` corresponding to the
             experiments in `exp_list`. Nested parallel experiment aren't supported currently.

        Raises:
            ValueError: Raised if the list are empty or if they don't have the same length.
            QiskitError: Raised if `exp_helper_list` contains an object of type
                ``MockIQParallelExperimentHelper``, because the parallel mock backend currently does not
                support parallel sub-experiments.`.

        Examples:

            **Parallel experiment for Resonator Spectroscopy**

            To run a parallel experiment of Resonator Spectroscopy on two qubits we will create two
            instances of `SpectroscopyHelper` objects (for each experiment) and an instance of
            `ParallelExperimentHelper` with them.


            .. code-block::

                iq_cluster_centers = [
                    ((-1.0, 0.0), (1.0, 0.0)),
                    ((0.0, -1.0), (0.0, 1.0)),
                    ((3.0, 0.0), (5.0, 0.0)),
                    ]

                parallel_backend = MockIQParallelBackend(
                    experiment_helper=None,
                    rng_seed=0,
                )
                parallel_backend._configuration.basis_gates = ["x"]
                parallel_backend._configuration.timing_constraints = {"granularity": 16}

                # experiment parameters
                qubit1 = 0
                qubit2 = 1
                freq01 = parallel_backend.defaults().qubit_freq_est[qubit1]
                freq02 = parallel_backend.defaults().qubit_freq_est[qubit2]

                # experiments initialization
                frequencies1 = np.linspace(freq01 - 10.0e6, freq01 + 10.0e6, 23)
                frequencies2 = np.linspace(freq02 - 10.0e6, freq02 + 10.0e6, 21)

                exp_list = [
                    QubitSpectroscopy(qubit1, frequencies1),
                    QubitSpectroscopy(qubit2, frequencies2),
                ]

                exp_helper_list = [
                    SpectroscopyHelper(iq_cluster_centers=iq_cluster_centers,),
                    SpectroscopyHelper(iq_cluster_centers=iq_cluster_centers,),
                ]
                parallel_helper = ParallelExperimentHelper(exp_list, exp_helper_list)

                parallel_backend.experiment_helper = parallel_helper

                # initializing the parallel experiment
                par_experiment = ParallelExperiment(exp_list, backend=parallel_backend)
                par_experiment.set_run_options(meas_level=MeasLevel.KERNELED, meas_return="single")

                par_data = par_experiment.run().block_for_results()
        """
        # Set ParallelExperimentHelper iq_cluster_[centers,widths] to None as exp_helper_list contains
        # the necessary IQ cluster information.
        super().__init__(None, None)

        # check parameters
        self._verify_parameters(exp_list, exp_helper_list)

        self.exp_helper_list = exp_helper_list
        self.exp_list = exp_list

    def compute_probabilities(
        self,
        circuits: List[QuantumCircuit],
    ) -> List[Dict[str, Any]]:
        """
        Run the compute_probabilities for each helper.

        Args:
            circuits: The quantum circuits for which the probabilities should be computed.

        Returns:
            List: A list of dictionaries containing computed probabilities and data for the given
                circuits.
        """
        # checking for legal parameters before computing output.
        self._verify_parameters(self.exp_list, self.exp_helper_list)

        # Splitting the circuit
        parallel_circ_list = self._parallel_exp_circ_splitter(circuits)
        number_of_experiments = len(self.exp_helper_list)
        prob_help_list = [{} for _ in range(number_of_experiments)]

        for idx, (exp_helper, experiment, experiment_circuits) in enumerate(
            zip(self.exp_helper_list, self.exp_list, parallel_circ_list)
        ):
            # Get centers and widths for experiment_circuits and split into centers and widths lists.
            centers_and_widths = exp_helper.iq_clusters(experiment_circuits)
            exp_centers = [c_and_w[0] for c_and_w in centers_and_widths]
            exp_widths = [c_and_w[1] for c_and_w in centers_and_widths]

            prob_help_list[idx] = {
                "physical_qubits": experiment.physical_qubits,
                "prob": exp_helper.compute_probabilities(experiment_circuits),
                "phase": exp_helper.iq_phase(experiment_circuits),
                "centers": exp_centers,
                "widths": exp_widths,
                "num_circuits": len(experiment_circuits),
            }

        return prob_help_list

    def _verify_parameters(
        self,
        exp_list: List[BaseExperiment] = None,
        exp_helper_list: List[MockIQExperimentHelper] = None,
    ):
        """Check parameters before computing probability"""
        if exp_helper_list is None:
            raise ValueError("Please set the experiment helper list.")
        if exp_list is None:
            raise ValueError("Please set the experiment list.")

        number_of_experiments = len(exp_list)
        number_of_helpers = len(exp_helper_list)

        if number_of_experiments == 0:
            raise ValueError("The experiment list cannot be empty.")
        if number_of_helpers == 0:
            raise ValueError("The experiment helper list cannot be empty.")

        if number_of_experiments != number_of_helpers:
            raise ValueError(
                f"The number of helpers {number_of_experiments} and the number of "
                f"experiment {number_of_helpers} don't match."
            )

        for helper in exp_helper_list:
            # checking there is no nested parallel experiment.
            if isinstance(helper, MockIQParallelExperimentHelper):
                raise QiskitError("Nested parallel experiments aren't currently supported.")

    def _parallel_exp_circ_splitter(self, qc_list: List[QuantumCircuit]):
        """
        Splits quantum circuits to their parallel components.
        Args:
            qc_list: The list of quantum circuits the backend gets as input.

        Returns:
            List: A list for each experiment. Each entry is a list of quantum circuits corresponding to
            that experiment.

        Raises:
            QiskitError: If an instruction is applied with qubits that don't belong to the same
            experiment.
            TypeError: The data type provided doesn't match the expected type (`tuple` or `int`).
        """
        exp_circuits_list = [[] for _ in self.exp_list]
        qubits_expid_map = {exp.physical_qubits: i for i, exp in enumerate(self.exp_list)}

        for qc in qc_list:
            # initialize quantum circuit for each experiment for this instance of circuit to fill
            # with instructions.
            for i in range(len(self.exp_list)):
                # we copy the circuit to ensure that the circuit properties (e.g. calibrations and qubit
                # frequencies) are the same in the new circuit.
                empty_qc = qc.copy_empty_like()
                empty_qc.metadata.clear()
                exp_circuits_list[i].append(empty_qc)

            # fixing metadata
            for exp_idx, sub_metadata in zip(
                qc.metadata["composite_index"],
                qc.metadata["composite_metadata"],
            ):
                exp_circuits_list[exp_idx][-1].metadata = sub_metadata.copy()

            # sorting instructions by qubits indexes and inserting them into a circuit of the relevant
            # experiment
            for data in qc.data:
                inst = data.operation
                qarg = data.qubits
                carg = data.clbits
                qubit_indices = set(qc.find_bit(qr).index for qr in qarg)
                for qubits, exp_idx in qubits_expid_map.items():
                    if qubit_indices.issubset(qubits):
                        exp_circuits_list[exp_idx][-1].append(inst, qarg, carg)
                        break
                else:
                    raise QiskitError(
                        "A gate operates on two qubits that don't belong to the same experiment."
                    )

            # deleting empty circuits
            for exp_circuits in exp_circuits_list:
                # 'exp_circuits' is a list of circuits of a specific experiment
                if not exp_circuits[-1].data:
                    exp_circuits.pop()

        return exp_circuits_list


class MockIQFineDragHelper(MockIQExperimentHelper):
    """Functions needed for Fine Drag Experiment"""

    def __init__(
        self,
        error: float = 0.03,
        iq_cluster_centers: Optional[List[Tuple[IQPoint, IQPoint]]] = None,
        iq_cluster_width: Optional[List[float]] = None,
    ):
        super().__init__(iq_cluster_centers, iq_cluster_width)
        self.error = error

    def compute_probabilities(self, circuits: List[QuantumCircuit]) -> List[Dict[str, float]]:
        """Returns the probability based on error per gate."""

        error = self.error
        output_dict_list = []
        for circuit in circuits:
            probability_output_dict = {}
            n_gates = circuit.count_ops().get("rz", 0) // 2

            # Dictionary of output string vectors and their probability
            probability_output_dict["1"] = 0.5 * np.sin(n_gates * error) + 0.5
            probability_output_dict["0"] = 1 - probability_output_dict["1"]
            output_dict_list.append(probability_output_dict)
        return output_dict_list


class MockIQFineFreqHelper(MockIQExperimentHelper):
    """Functions needed for Fine Frequency experiment on mock IQ backend"""

    def __init__(
        self,
        sx_duration: float = 160,
        freq_shift: float = 0,
        dt: float = 1e-9,
        iq_cluster_centers: Optional[List[Tuple[IQPoint, IQPoint]]] = None,
        iq_cluster_width: Optional[List[float]] = None,
    ):
        """
        Args:
            sx_duration: duration of the single-qubit sx gate.
            freq_shift: the detuning from the ideal frequency that this mock backend will mimic.
            dt: duration of a sample.
        """
        super().__init__(iq_cluster_centers, iq_cluster_width)
        self.sx_duration = sx_duration
        self.freq_shift = freq_shift
        self.dt = dt

    def compute_probabilities(self, circuits: List[QuantumCircuit]) -> List[Dict[str, float]]:
        """Return the probability of being in the excited state."""
        sx_duration = self.sx_duration
        freq_shift = self.freq_shift
        dt = self.dt
        simulator = AerSimulator(method="automatic")
        output_dict_list = []
        for circuit in circuits:
            probability_output_dict = {}
            delay = None
            for instruction in circuit.data:
                if instruction.operation.name == "delay":
                    delay = instruction.operation.duration

            if delay is None:
                probability_output_dict = {"1": 1, "0": 0}
            else:
                reps = delay // sx_duration

                qc = QuantumCircuit(1)
                qc.sx(0)
                qc.rz(np.pi * reps / 2 + 2 * np.pi * freq_shift * delay * dt, 0)
                qc.sx(0)
                qc.measure_all()

                counts = simulator.run(qc, seed_simulator=1).result().get_counts(0)
                probability_output_dict["1"] = counts.get("1", 0) / sum(counts.values())
                probability_output_dict["0"] = 1 - probability_output_dict["1"]
            output_dict_list.append(probability_output_dict)

        return output_dict_list


class MockIQFineAmpHelper(MockIQExperimentHelper):
    """Functions needed for Fine Amplitude experiment on mock IQ backend"""

    def __init__(
        self,
        angle_error: float = 0,
        angle_per_gate: float = 0,
        gate_name: str = "x",
        iq_cluster_centers: Optional[List[Tuple[IQPoint, IQPoint]]] = None,
        iq_cluster_width: Optional[List[float]] = None,
    ):
        """
        Args:
            angle_error: rotation angle error per gate.
            angle_per_gate: the intended rotation angle per gate.
            gate_name: name of the gate that will be counted to determine the total rotation.
        """
        super().__init__(iq_cluster_centers, iq_cluster_width)
        self.angle_error = angle_error
        self.angle_per_gate = angle_per_gate
        self.gate_name = gate_name

    def compute_probabilities(self, circuits: List[QuantumCircuit]) -> List[Dict[str, float]]:
        """Return the probability of being in the excited state."""
        angle_error = self.angle_error
        angle_per_gate = self.angle_per_gate
        gate_name = self.gate_name
        output_dict_list = []
        for circuit in circuits:
            probability_output_dict = {}
            n_ops = circuit.count_ops().get(gate_name, 0)
            angle = n_ops * (angle_per_gate + angle_error)

            if gate_name != "sx":
                angle += np.pi / 2 * circuit.count_ops().get("sx", 0)

            if gate_name != "x":
                angle += np.pi * circuit.count_ops().get("x", 0)

            # Dictionary of output string vectors and their probability
            probability_output_dict["1"] = np.sin(angle / 2) ** 2
            probability_output_dict["0"] = 1 - probability_output_dict["1"]
            output_dict_list.append(probability_output_dict)

        return output_dict_list


class MockIQRamseyXYHelper(MockIQExperimentHelper):
    """Functions needed for Ramsey XY experiment on mock IQ backend"""

    def __init__(
        self,
        t2ramsey: float = 100e-6,
        freq_shift: float = 0,
        iq_cluster_centers: Optional[List[Tuple[IQPoint, IQPoint]]] = None,
        iq_cluster_width: Optional[List[float]] = None,
    ):
        super().__init__(iq_cluster_centers, iq_cluster_width)
        self.t2ramsey = t2ramsey
        self.freq_shift = freq_shift

    def compute_probabilities(self, circuits: List[QuantumCircuit]) -> List[Dict[str, float]]:
        """Return the probability of being in the excited state."""
        t2ramsey = self.t2ramsey
        freq_shift = self.freq_shift
        output_dict_list = []
        for circuit in circuits:
            probability_output_dict = {}
            series = circuit.metadata["series"]
            delay = circuit.metadata["xval"]

            if series == "X":
                phase_offset = 0.0
            else:
                phase_offset = np.pi / 2

            probability_output_dict["1"] = (
                0.5
                * np.exp(-delay / t2ramsey)
                * np.cos(2 * np.pi * delay * freq_shift - phase_offset)
                + 0.5
            )
            probability_output_dict["0"] = 1 - probability_output_dict["1"]
            output_dict_list.append(probability_output_dict)
        return output_dict_list


class MockIQReadoutAngleHelper(MockIQExperimentHelper):
    """Functions needed for Readout angle experiment on mock IQ backend"""

    def compute_probabilities(self, circuits: List[QuantumCircuit]) -> List[Dict[str, float]]:
        """Return the probability of being in the excited state."""
        output_dict_list = []
        for circuit in circuits:
            probability_output_dict = {"1": 1 - circuit.metadata["xval"]}
            probability_output_dict["0"] = 1 - probability_output_dict["1"]
            output_dict_list.append(probability_output_dict)

        return output_dict_list


class MockIQHalfAngleHelper(MockIQExperimentHelper):
    """Functions needed for Half Angle experiment on mock IQ backend"""

    def __init__(
        self,
        error: float = 0,
        iq_cluster_centers: Optional[List[Tuple[IQPoint, IQPoint]]] = None,
        iq_cluster_width: Optional[List[float]] = None,
    ):
        super().__init__(iq_cluster_centers, iq_cluster_width)
        self.error = error

    def compute_probabilities(self, circuits: List[QuantumCircuit]) -> List[Dict[str, float]]:
        """Return the probability of being in the excited state."""
        error = self.error
        output_dict_list = []
        for circuit in circuits:
            probability_output_dict = {}
            n_gates = circuit.metadata["xval"]

            # Dictionary of output string vectors and their probability
            probability_output_dict["1"] = (
                0.5 * np.sin((-1) ** (n_gates + 1) * n_gates * error) + 0.5
            )
            probability_output_dict["0"] = 1 - probability_output_dict["1"]
            output_dict_list.append(probability_output_dict)

        return output_dict_list


class MockIQT1Helper(MockIQExperimentHelper):
    """Functions needed for T1 experiment on mock IQ backend"""

    def __init__(
        self,
        t1: float = None,
        iq_cluster_centers: Optional[List[Tuple[IQPoint, IQPoint]]] = None,
        iq_cluster_width: Optional[List[float]] = None,
    ):
        super().__init__(iq_cluster_centers, iq_cluster_width)
        self._t1 = t1 or 90e-6

    def compute_probabilities(self, circuits: List[QuantumCircuit]) -> List[Dict[str, float]]:
        """Return the probability of being in the excited state."""
        output_dict_list = []
        for circuit in circuits:
            probability_output_dict = {}

            # extracting information from the circuit.
            delay = circuit.metadata["xval"]

            # creating a probability dict.
            probability_output_dict["1"] = np.exp(-delay / self._t1)
            probability_output_dict["0"] = 1 - probability_output_dict["1"]
            output_dict_list.append(probability_output_dict)

        return output_dict_list
