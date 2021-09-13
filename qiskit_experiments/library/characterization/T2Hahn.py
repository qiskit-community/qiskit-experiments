from typing import Union, Iterable, Optional, List

import numpy as np
from numpy.random import Generator, default_rng

from qiskit import QuantumCircuit, QiskitError, QuantumRegister, ClassicalRegister
from qiskit.ignis.characterization.characterization_utils import pad_id_gates, time_to_ngates
from qiskit.providers import Backend
from qiskit.quantum_info import Clifford
from qiskit.providers.options import Options
from qiskit.circuit import Gate

from qiskit_experiments.framework import BaseExperiment, ParallelExperiment
from qiskit_experiments.curve_analysis.data_processing import probability
from qiskit.utils import apply_prefix

from qiskit.providers import Backend
from qiskit.test.mock import FakeParis
from qiskit.providers.aer import AerSimulator


class T2Hahn(BaseExperiment):

    # Analysis class for experiment
    #     __analysis_class__ = T2Analysis  # need to add T2Analysis

    @classmethod
    def _default_experiment_options(cls) -> Options:
        """Default experiment options.

        Experiment Options:
            delays (Iterable[float]): Delay times of the experiments.
            unit (str): Unit of the delay times. Supported units are
                's', 'ms', 'us', 'ns', 'ps', 'dt'.
            osc_freq (float): Oscillation frequency offset in Hz.
            n_echos (int); Number of echoes to preform.

        """
        options = super()._default_experiment_options()

        options.delays = None
        options.unit = "s"
        options.osc_freq = 0.0
        options.n_echoes = 1
        options.phase_alt_echo = False

        return options

    def __init__(
            self,
            qubit: Union[int, Iterable[int]],
            delays: Union[List[int], np.array],  # need to change name?
            gate_time: float,
            n_echos: int = 1,
            phase_alt_echo: bool = False,
    ):
        """Initialize a T2 experiment with Hahn echo.

         Args:
                num_of_gates:
                    Each element of the list corresponds to a circuit.
                    `num_of_gates[i]` is the number of identity gates in each section
                    "t" of the pulse sequence in circuit no. i.
                    Must be in an increasing order.
                gate_time: time of running a single identity gate.
                qubits: indices of the qubits whose
                    T\ :sub:`2`:sup:`*`\ 's are to be measured.
                n_echos: number of echo gates (`X` or `Y`).
                phase_alt_echo: if True then alternate the echo between
                    `X` and `Y`.
        """
        # Initialize base experiment
        super().__init__(qubit)
        self._verify_parameters(qubit, delays, n_echos, gate_time)
        self._qubit = qubit
        self._delays = delays
        self._gate_time = gate_time
        self._n_echos = n_echos
        self._phase_alt_echo = phase_alt_echo

        # Set configurable options

    #         self.set_experiment_options(delays=delays, n_echos=n_echos, phase_alt_echo=phase_alt_echo)
    #         self.set_analysis_options(data_processor=probability(
    #             outcome="0" * self.num_qubits))  # Need to rewrite after making the analysis class

    @staticmethod
    def _verify_parameters(self, qubit, delays, n_echos, gate_time):
        """Verify input correctness, raise QiskitError if needed"""
        if any(delay <= 0 for delay in delays):
            raise QiskitError(
                f"The lengths list {delays} should only contain " "positive elements."
            )
        if len(set(delays)) != len(delays):
            raise QiskitError(
                f"The lengths list {delays} should not contain " "duplicate elements."
            )
        if any(delays[idx - 1] >= delays[idx] for idx in range(1, len(delays))):
            raise QiskitError(f"The number of identity gates {delays} should " "be increasing.")

        # if any(qubit < 0 for qubit in qubits):
        #     raise QiskitError(f"The index of the qubits {qubits} should " "be non-negative.")
        if isinstance(qubit, List):
            if len(qubit) != 1:
                raise QiskitError(f"The experiment if for 1 qubit. For multiple qubits, please use "
                                  f"parallel experiments.")
            if qubit[0] < 0:
                raise QiskitError(f"The index of the qubit {qubit[0]} should " "be non-negative.")
        else:
            if qubit < 0:
                raise QiskitError(f"The index of the qubit {qubit} should " "be non-negative.")

        if n_echos < 1:
            raise QiskitError(f"The number of echoes {n_echos} should " "be at least 1.")

        if gate_time <= 0:
            raise QiskitError(f"The gate time {gate_time} should " "be positive.")

    def circuits(self, backend, qubit: Union[List[int], np.array],
                 n_echos: int = 1,
                 phase_alt_echo: bool = False):

        conversion_factor = 1
        #         if self.experiment_options.unit == "dt":
        #             try:
        #                 dt_factor = getattr(backend._configuration, "dt")
        #                 conversion_factor = dt_factor
        #             except AttributeError as no_dt:
        #                 raise AttributeError("Dt parameter is missing in backend configuration") from no_dt
        #         elif self.experiment_options.unit != "s":
        #             apply_prefix(1, self.experiment_options.unit)
        #         xdata = 2 * gate_time * np.array(num_of_gates) * n_echos
        qr = QuantumRegister(max(qubit) + 1)
        cr = ClassicalRegister(len(qubit))
        circuits = []
        for circ_index, delay in enumerate(self._delays):
            circ = QuantumCircuit(max(qubit) + 1, len(qubit))
            circ.name = 't2circuit_' + str(circ_index) + '_0'
            # First Y rotation in 90 degrees
            circ.ry(np.pi / 2, qubit)  # Bring to qubits to X Axis
            # circ = pad_id_gates(circ, qr, qubit, circ_length)  # ids - waiting
            #                 circ.delay(delay, qr[qubit], self.experiment_options.unit)
            circ.delay(delay, qubit, 's')
            circ.rx(np.pi, qubit)

            #                 for echoid in range(n_echos - 1):  # repeat
            #                     circ = pad_id_gates(circ, qr, qubit, 2 * delay)  # ids
            #                     if phase_alt_echo and (not echoid % 2):  # optionally
            #                         circ.x(qr[qubit])  # X
            #                     else:
            #                         circ.y(qr[qubit])

            #                 circ.delay(delay, qr[qubit], self.experiment_options.unit)  # ids
            circ.delay(delay, qubit, 's')
            circ.ry(-np.pi / 2, qubit)  # Y90
            circ.measure(qubit, 0)  # measure
            circ.metadata = {
                "experiment_type": self._type,
                "qubit": self.physical_qubits,
                "xval": delay,
                "unit": 's',
            }
            #             if self.experiment_options.unit == "dt":
            #                 circ.metadata["dt_factor"] = dt_factor
            circuits.append(circ)

        return circuits