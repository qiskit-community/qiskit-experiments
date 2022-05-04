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
from typing import Dict, List
import numpy as np
from qiskit import QuantumCircuit
from qiskit.providers.aer import AerSimulator


class MockIQExperimentHelper:
    """Abstract class for the MockIQ helper classes"""

    @abstractmethod
    def compute_probabilities(self, circuits: List[QuantumCircuit]) -> List[Dict[str, float]]:
        """
        A function provided by the user which is used to determine the probability of each output of the
        circuit. The function returns a list of dictionaries, each containing output binary strings and
        their probabilities.

        Examples:

            **1 qubit circuit - excited state**

            In this experiment, we want to bring a qubit to its excited state and measure it.
            The circuit:
                         ┌───┐┌─┐
                      q: ┤ X ├┤M├
                         └───┘└╥┘
                    c: 1/══════╩═
                               0

            The function that calculates the probability for this circuit, doesn't need any
            calculation_parameters. It will be as following:

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
                         ┌───┐     ┌─┐
                    q_0: ┤ H ├──■──┤M├───
                         └───┘┌─┴─┐└╥┘┌─┐
                    q_1: ─────┤ X ├─╫─┤M├
                         ┌───┐└┬─┬┘ ║ └╥┘
                    q_2: ┤ X ├─┤M├──╫──╫─
                         └───┘ └╥┘  ║  ║
                    c: 3/═══════╩═══╩══╩═
                                2   0  1

            When an output string isn't in the probability dictionary, the backend will presume its
             probability is 0.

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
    def iq_phase(self, circuit: QuantumCircuit) -> float:
        """Sub-classes can override this method to introduce a phase in the IQ plan.

        This is needed, to test the resonator spectroscopy where the point in the IQ
        plan has a frequency-dependent phase rotation.
        """
        return 0.0


class MockIQDragHelper(MockIQExperimentHelper):
    """Functions needed for test_drag"""

    def __init__(
        self,
        gate_name: str = "Rp",
        ideal_beta: float = 2.0,
        frequency: float = 0.02,
        max_probability: float = 1.0,
        offset_probability: float = 0.0,
    ):
        """
        Args:
            gate_name: name of the gate to count when determining the number of gate repetitions,
            i.e., positive rotation followed by negative rotation, in the circuit.
            ideal_beta: the beta where the minimum of the Drag patterns will be.
            frequency: controls the frequency of the oscillation in the measured Drag pattern.
            max_probability:  a factor to scale the maximum probability of measuring an excited state to
            allow tests to factor in non-ideal situations.
            offset_probability: a constant offset applied to all probabilities to reflect non-ideal
            measurement situations.
        Raises:
            ValueError: if probability value is ot valid.
        """
        if max_probability + offset_probability > 1:
            raise ValueError("Probabilities need to be between 0 and 1.")

        self.gate_name = gate_name
        self.ideal_beta = ideal_beta
        self.frequency = frequency
        self.max_probability = max_probability
        self.offset_probability = offset_probability

    def compute_probabilities(self, circuits: List[QuantumCircuit]) -> List[Dict[str, float]]:
        """Returns the probability based on the beta, number of gates, and leakage."""

        gate_name = self.gate_name
        ideal_beta = self.ideal_beta
        freq = self.frequency
        max_prob = self.max_probability
        offset_prob = self.offset_probability

        if max_prob + offset_prob > 1:
            raise ValueError("Probabilities need to be between 0 and 1.")

        output_dict_list = []
        for circuit in circuits:
            probability_output_dict = {}
            n_gates = circuit.count_ops()[gate_name]
            beta = next(iter(circuit.calibrations[gate_name].keys()))[1][0]

            # Dictionary of output string vectors and their probability
            prob = np.sin(2 * np.pi * n_gates * freq * (beta - ideal_beta) / 4) ** 2
            probability_output_dict["1"] = max_prob * prob + offset_prob
            probability_output_dict["0"] = 1 - probability_output_dict["1"]
            output_dict_list.append(probability_output_dict)
        return output_dict_list


class MockIQFineDragHelper(MockIQExperimentHelper):
    """Functions needed for Fine Drag Experiment"""

    def __init__(self, error: float = 0.03):
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


class MockIQRabiHelper(MockIQExperimentHelper):
    """Functions needed for Rabi experiment on mock IQ backend"""

    def __init__(self, amplitude_to_angle: float = np.pi):
        """
        Args:
            amplitude_to_angle: maps a pulse amplitude to a rotation angle.
        """
        self.amplitude_to_angle = amplitude_to_angle

    def compute_probabilities(self, circuits: List[QuantumCircuit]) -> List[Dict[str, float]]:
        """Returns the probability based on the rotation angle and amplitude_to_angle."""
        amplitude_to_angle = self.amplitude_to_angle
        output_dict_list = []
        for circuit in circuits:
            probability_output_dict = {}
            amp = next(iter(circuit.calibrations["Rabi"].keys()))[1][0]

            # Dictionary of output string vectors and their probability
            probability_output_dict["1"] = np.sin(amplitude_to_angle * amp) ** 2
            probability_output_dict["0"] = 1 - probability_output_dict["1"]
            output_dict_list.append(probability_output_dict)
        return output_dict_list

    def rabi_rate(self) -> float:
        """Returns the rabi rate."""
        return self.amplitude_to_angle / np.pi


class MockIQFineFreqHelper(MockIQExperimentHelper):
    """Functions needed for Fine Frequency experiment on mock IQ backend"""

    def __init__(self, sx_duration: float = 160, freq_shift: float = 0, dt: float = 1e-9):
        """
        Args:
            sx_duration: duration of the single-qubit sx gate.
            freq_shift: the detunning from the ideal frequency that this mock backend will mimic.
            dt: duration of a sample.
        """
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
                if instruction[0].name == "delay":
                    delay = instruction[0].duration

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

    def __init__(self, angle_error: float = 0, angle_per_gate: float = 0, gate_name: str = "x"):
        """
        Args:
            angle_error: rotation angle error per gate.
            angle_per_gate: the intended rotation angle per gate.
            gate_name: name of the gate that will be counted to determine the total rotation.
        """
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

    def __init__(self, t2ramsey: float = 100e-6, freq_shift: float = 0):
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


class MockIQSpectroscopyHelper(MockIQExperimentHelper):
    """Functions needed for Spectroscopy experiment on mock IQ backend"""

    def __init__(self, gate_name: str = "Spec", freq_offset: float = 0.0, line_width: float = 2e6):
        """
        Args:
            gate_name: the gate name to look for when calculating frequency shift.
            freq_offset: frequency offset from resonance that this mock backend will mimic.
            line_width: line width of the resonance of the spectroscopy signal.
        """
        self.freq_offset = freq_offset
        self.line_width = line_width
        self.gate_name = gate_name

    def compute_probabilities(self, circuits: List[QuantumCircuit]) -> List[Dict[str, float]]:
        """Returns the probability based on the parameters provided."""
        freq_offset = self.freq_offset
        line_width = self.line_width
        output_dict_list = []
        for circuit in circuits:
            probability_output_dict = {}
            if self.gate_name == "measure":
                freq_shift = (
                    next(iter(circuit.calibrations[self.gate_name].values())).blocks[0].frequency
                )
            elif self.gate_name == "Spec":
                freq_shift = next(iter(circuit.calibrations[self.gate_name]))[1][0]
            else:
                raise ValueError("The gate name {} isn't supported.".format(str(self.gate_name)))
            delta_freq = freq_shift - freq_offset

            probability_output_dict["1"] = np.abs(1 / (1 + 2.0j * delta_freq / line_width))
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

    def __init__(self, error: float = 0):
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
