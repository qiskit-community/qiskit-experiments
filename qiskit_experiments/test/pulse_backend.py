# This code is part of Qiskit.
#
# (C) Copyright IBM 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""A Pulse simulation backend based on Qiskit-Dynamics"""

import datetime
from itertools import chain
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from qiskit import QuantumCircuit
from qiskit.circuit import CircuitInstruction
from qiskit.circuit.library.standard_gates import RZGate, SXGate, XGate
from qiskit.circuit.measure import Measure
from qiskit.circuit.parameter import Parameter
from qiskit.providers import BackendV2, QubitProperties
from qiskit.providers.models import PulseDefaults  # pylint: disable=no-name-in-module
from qiskit.providers.models.pulsedefaults import Command
from qiskit.providers.options import Options
from qiskit.pulse import Schedule, ScheduleBlock
from qiskit.pulse.transforms import block_to_schedule
from qiskit.qobj.pulse_qobj import PulseQobjInstruction
from qiskit.qobj.utils import MeasLevel, MeasReturnType
from qiskit.quantum_info.states import DensityMatrix, Statevector
from qiskit.result import Result, Counts
from qiskit.transpiler import InstructionProperties, Target

from qiskit_experiments.data_processing.discriminator import BaseDiscriminator
from qiskit_experiments.exceptions import QiskitError
from qiskit_experiments.framework.package_deps import HAS_DYNAMICS, version_is_at_least
from qiskit_experiments.test.utils import FakeJob


@HAS_DYNAMICS.require_in_instance
class PulseBackend(BackendV2):
    r"""Abstract base class for pulse simulation backends in Qiskit Experiments.

    This backend is designed for the tests in Qiskit Experiments as well as for the
    tutorials in Qiskit Experiments. The backend has a Qiskit Dynamics pulse simulator
    which allows it to simulate pulse schedules that are included in the calibrations
    attached to quantum circuits. In addition, sub-classes should implement a set of default
    schedules so that circuits that do not provide calibrations can also run, much like
    the hardware backends. In addition, the backends are also capable of simulating level-
    one (IQ data) and level-two (counts) data. Subclasses of these backends can have an
    optional discriminator so that they can produce counts based on sampled IQ data. If
    a discriminator is not provided then the counts will be produced from a statevector
    or density matrix.

    .. warning::

        Some of the functionality in this backend may move to Qiskit Dynamics and/or be
        refactored. These backends are not intended as a general pulse-simulator backend
        but rather to test the experiments and write short tutorials to demonstrate an
        experiment without having to run on hardware.
    """

    def __init__(
        self,
        static_hamiltonian: np.ndarray,
        hamiltonian_operators: np.ndarray,
        static_dissipators: Optional[np.ndarray] = None,
        dt: float = 0.1 * 1e-9,
        solver_method="RK23",
        seed: int = 0,
        atol: float = None,
        rtol: float = None,
        **kwargs,
    ):
        """Initialize a backend with model information.

        Args:
            static_hamiltonian: Time-independent term in the Hamiltonian.
            hamiltonian_operators: List of time-dependent operators
            static_dissipators: Constant dissipation operators. Defaults to None.
            dt: Sample rate for simulating pulse schedules. Defaults to 0.1*1e-9.
            solver_method: Numerical solver method to use. Check qiskit_dynamics for available
                methods. Defaults to "RK23".
            seed: An optional seed given to the random number generator. If this argument is not
                set then the seed defaults to 0.
            atol: Absolute tolerance during solving.
            rtol: Relative tolerance during solving.
        """
        from qiskit_dynamics import Solver

        super().__init__(
            None,
            name="PulseBackendV2",
            description="A PulseBackend simulator",
            online_date=datetime.datetime.now(datetime.timezone.utc),
            backend_version="0.0.1",
        )

        # subclasses must implements default pulse schedules
        self._defaults = None

        self._target = Target(dt=dt, granularity=16)

        # The RNG to sample IQ data.
        self._rng = np.random.default_rng(seed)

        # The instance to convert pulse schedules to signals for Qiskit Dynamics.
        self.converter = None

        self.solver_method = solver_method

        self.solve_kwargs = {}
        if atol:
            self.solve_kwargs["atol"] = atol
        if rtol:
            self.solve_kwargs["rtol"] = rtol

        self.static_hamiltonian = static_hamiltonian
        self.hamiltonian_operators = hamiltonian_operators
        self.static_dissipators = static_dissipators
        self.solver = Solver(
            static_hamiltonian=self.static_hamiltonian,
            hamiltonian_operators=self.hamiltonian_operators,
            static_dissipators=self.static_dissipators,
            **kwargs,
        )

        self.model_dim = self.solver.model.dim

        if self.static_dissipators is None:
            self.y_0 = np.eye(self.model_dim)
            self.ground_state = np.array([1.0] + [0.0] * (self.model_dim - 1))
        else:
            self.y_0 = np.eye(self.model_dim**2)
            self.ground_state = np.array([1.0] + [0.0] * (self.model_dim**2 - 1))

        self._simulated_pulse_unitaries = {}

        # An internal cache of schedules to unitaries. The key is a hashed string representation.
        self._schedule_cache = {}

        # An optional discriminator that is used to create counts from IQ data.
        self._discriminator = None

    @property
    def target(self):
        """Contains information for circuit transpilation."""
        return self._target

    @property
    def max_circuits(self):
        return None

    def defaults(self):
        """return backend pulse defaults"""
        return self._defaults

    @property
    def discriminator(self) -> BaseDiscriminator:
        """Return the discriminator for the IQ data."""
        return self._discriminator

    @discriminator.setter
    def discriminator(self, discriminator: BaseDiscriminator):
        """Set the discriminator."""
        self._discriminator = discriminator

    @classmethod
    def _default_options(cls) -> Options:
        """Returns the default options of the backend."""
        return Options(
            shots=4000,
            meas_level=MeasLevel.CLASSIFIED,
            meas_return=MeasReturnType.AVERAGE,
            memory=False,
        )

    @property
    def default_pulse_unitaries(self) -> Dict[Tuple, np.array]:
        """Return the default unitary matrices of the backend."""
        return self._simulated_pulse_unitaries

    @default_pulse_unitaries.setter
    def default_pulse_unitaries(self, unitaries: Dict[Tuple, np.array]):
        """Set the default unitary pulses this allows the tests to simulate the pulses only once."""
        self._simulated_pulse_unitaries = unitaries

    @staticmethod
    def _get_info(
        circuit: QuantumCircuit, instruction: CircuitInstruction
    ) -> Tuple[Tuple[int], Tuple[float], str]:
        """Returns information that uniquely describes a circuit instruction.

        Args:
            circuit: The quantum circuit in which the instruction is located. This is needed to
                access the register in the circuit.
            instruction: A gate or operation in a QuantumCircuit

        Returns:
            Tuple of qubit index, gate parameters and instruction name.
        """
        p_dict = instruction.operation
        qubit = tuple(int(circuit.qregs[0].index(qbit)) for qbit in instruction.qubits)
        params = tuple(float(val) for val in p_dict.params)
        return qubit, params, p_dict.name

    def _iq_data(
        self,
        probability: np.ndarray,
        shots: int,
        centers: List[Tuple[float, float]],
        width: float,
        phase: Optional[float] = None,
    ) -> Tuple[List, List]:
        """Generates IQ data for each physical level.

        Args:
            probability: probability of occupation
            shots: Number of shots
            centers: The central I and Q points for each level
            width: Width of IQ data distribution
            phase: Phase of IQ data, by default 0. Defaults to None.

        Returns:
            (I,Q) data.
        """
        counts_n = self._rng.multinomial(shots, probability / sum(probability), size=1).T

        full_i, full_q = [], []

        for idx, count_i in enumerate(counts_n):
            full_i.append(self._rng.normal(loc=centers[idx][0], scale=width, size=count_i))
            full_q.append(self._rng.normal(loc=centers[idx][1], scale=width, size=count_i))

        full_i = list(chain.from_iterable(full_i))
        full_q = list(chain.from_iterable(full_q))

        if phase is not None:
            complex_iq = (full_i + 1.0j * full_q) * np.exp(1.0j * phase)
            full_i, full_q = complex_iq.real, complex_iq.imag

        full_iq = 1e16 * np.array([[full_i], [full_q]]).T
        return full_iq.tolist()

    # pylint: disable=unused-argument
    def _iq_cluster_centers(self, circuit: Optional[QuantumCircuit] = None) -> List[Tuple[float]]:
        """A function to provide the points for the IQ centers when doing readout.

        Subclasses can override this function, for instance, to provide circuit dependent
        IQ cluster centers. If this function is not overridden then the IQ cluster centers returned
        are evenly distributed on the unit sphere in the IQ plane with |0> located at IQ point (0, 1).

        Args:
            circuit: provided so that sub-classes that implement their own IQ simulation
                by overriding this method can access circuit-level data (e.g. for
                ReadoutSpectroscopy simulation).

        Returns:
            Coordinates for IQ centers.
        """
        theta = 2 * np.pi / self.model_dim
        return [(np.cos(idx * theta), np.sin(idx * theta)) for idx in range(self.model_dim)]

    def _state_to_measurement_data(
        self,
        state: np.ndarray,
        shots: int,
        meas_level: MeasLevel,
        meas_return: MeasReturnType,
        memory: bool,
        circuit: QuantumCircuit,
    ) -> Tuple[Union[Union[Dict, Counts, Tuple[List, List]], Any], Optional[Any]]:
        """Convert State objects to IQ data or Counts.

        The counts are produced by sampling from the state vector if no discriminator is
        present. Otherwise, IQ shots are generated and then discriminated based on the
        discriminator.

        Args:
            state: Quantum state information.
            shots: Number of repetitions of each circuit, for sampling.
            meas_level: Measurement level 1 returns IQ data. 2 returns counts.
            meas_return: "single" returns information from every shot. "avg" returns average
                measurement output (averaged over number of shots).
            circuit: The circuit is provided so that :meth:`iq_data` can leverage any circuit-level
                information that it might need to generate the IQ shots.

        Returns:
            Measurement output as either counts or IQ data depending on the run input.

        Raises:
            QiskitError: If unsuported measurement options are provided.
        """
        memory_data = None
        if self.static_dissipators is not None:
            state = state.reshape(self.model_dim, self.model_dim)
            state = DensityMatrix(state / np.trace(state))
        else:
            state = Statevector(state / np.linalg.norm(state))

        if meas_level == MeasLevel.CLASSIFIED:
            if self._discriminator is None:
                if memory:
                    memory_data = state.sample_memory(shots)
                    measurement_data = dict(zip(*np.unique(memory_data, return_counts=True)))
                    memory_data = memory_data.tolist()
                else:
                    measurement_data = state.sample_counts(shots)
            else:
                centers = self._iq_cluster_centers(circuit=circuit)
                iq_data = self._iq_data(state.probabilities(), shots, centers, 0.2)
                memory_data = self._discriminator.predict(iq_data)
                measurement_data = dict(zip(*np.unique(memory_data, return_counts=True)))

        elif meas_level == MeasLevel.KERNELED:
            centers = self._iq_cluster_centers(circuit=circuit)
            measurement_data = self._iq_data(state.probabilities(), shots, centers, 0.2)
            if meas_return == "avg":
                measurement_data = np.average(np.array(measurement_data), axis=0).tolist()
        else:
            raise QiskitError(f"Unsupported measurement level {meas_level}.")

        return measurement_data, memory_data

    def solve(self, schedule: Union[ScheduleBlock, Schedule], qubits: Tuple[int]) -> np.ndarray:
        """Solves for qubit dynamics under the action of a pulse instruction

        Args:
            schedule: Pulse signal
            qubits: (remove after multi-qubit gates is implemented)

        Returns:
            Time-evolution unitary operator
        """
        if len(qubits) > 1:
            raise QiskitError("Multi qubit gates are not yet implemented.")

        if isinstance(schedule, ScheduleBlock):
            schedule = block_to_schedule(schedule)

        signal = self.converter.get_signals(schedule)
        time_f = schedule.duration * self.dt
        unitary = self.solver.solve(
            t_span=[0.0, time_f],
            y0=self.y_0,
            t_eval=[time_f],
            signals=signal,
            method=self.solver_method,
            **self.solve_kwargs,
        ).y[0]

        return unitary

    def run(self, run_input: Union[QuantumCircuit, List[QuantumCircuit]], **run_options) -> FakeJob:
        """Run method takes circuits as input and returns FakeJob with IQ data or counts.

        Args:
            run_input: Circuits to run.
            run_options: Any option that affects the way that the circuits are run. The options
                that are currently supported are ``shots``, ``meas_level``, ``meas_return``,
                and ``memory``.

        Returns:
            FakeJob with simulation data.
        """
        self.options.update_options(**run_options)
        shots = self.options.get("shots", self._options.shots)
        meas_level = self.options.get("meas_level", self._options.meas_level)
        meas_return = self.options.get("meas_return", self._options.meas_return)
        memory = self.options.get("memory", self._options.memory)

        result = {
            "backend_name": f"{self.__class__.__name__}",
            "backend_version": self.backend_version,
            "qobj_id": 0,
            "job_id": 0,
            "success": True,
            "results": [],
        }

        if isinstance(run_input, QuantumCircuit):
            run_input = [run_input]

        for circuit in run_input:
            unitaries = {}

            # 1. Parse the calibrations and simulate any new schedule. Add U to the unitaries.
            for name, schedule in circuit.calibrations.items():
                for (qubits, params), schedule_block in schedule.items():
                    schedule_key = hash(repr(schedule))

                    # Simulate the schedule if not in the cache.
                    if schedule_key not in self._schedule_cache:
                        self._schedule_cache[schedule_key] = self.solve(schedule_block, qubits)

                    unitaries[(name, qubits, params)] = self._schedule_cache[schedule_key]

            # 2. Copy over any remaining instructions to the dict of unitaries.
            for key, unitary in self.default_pulse_unitaries.items():
                if key not in unitaries:
                    unitaries[key] = unitary

            # 3. Multiply the unitaries of the circuit instructions onto the ground state.
            state_t = self.ground_state.copy()
            for instruction in circuit.data:
                qubits, params, inst_name = self._get_info(circuit, instruction)
                if inst_name in ["barrier", "measure"]:
                    continue
                if inst_name == "rz":
                    # Ensures that the action in the qubit space is preserved.
                    unitary = np.diag([np.exp(1.0j * idx * params[0] / 2) for idx in [-1, 1, 3]])
                else:
                    unitary = unitaries[(inst_name, qubits, params)]
                state_t = unitary @ state_t

            # 4. Convert the probabilities to IQ data or counts.
            measurement_data, memory_data = self._state_to_measurement_data(
                state_t, shots, meas_level, meas_return, memory, circuit
            )

            run_result = {
                "shots": shots,
                "success": True,
                "header": {"metadata": circuit.metadata},
                "meas_level": meas_level,
                "meas_return": meas_return,
                "data": {},
            }

            if meas_level == MeasLevel.CLASSIFIED:
                run_result["data"]["counts"] = measurement_data
                if memory_data is not None:
                    run_result["data"]["memory"] = memory_data
            if meas_level == MeasLevel.KERNELED:
                run_result["data"]["memory"] = measurement_data

            result["results"].append(run_result)

        return FakeJob(self, Result.from_dict(result))


@HAS_DYNAMICS.require_in_instance
class SingleTransmonTestBackend(PulseBackend):
    r"""A backend that corresponds to a three level anharmonic transmon qubit.

    The Hamiltonian of the system is

    .. math::
        H = \hbar \sum_{j=1,2} \left[\omega_j |j\rangle\langle j| +
                \mathcal{E}(t) \lambda_j (\sigma_j^+ + \sigma_j^-)\right]

    Here, :math:`\omega_j` is the transition frequency from level :math:`0` to level
    :math:`j`. :math:`\mathcal{E}(t)` is the drive field and :math:`\sigma_j^\pm` are
    the raising and lowering operators between levels :math:`j-1` and :math:`j`.
    """

    def __init__(
        self,
        qubit_frequency: float = 5e9,
        anharmonicity: float = -0.25e9,
        lambda_1: float = 1e9,
        lambda_2: float = 0.8e9,
        gamma_1: float = 1e4,
        noise: bool = True,
        atol: float = None,
        rtol: float = None,
        **kwargs,
    ):
        """Initialise backend with hamiltonian parameters

        Args:
            qubit_frequency: Frequency of the qubit (0-1). Defaults to 5e9.
            anharmonicity: Qubit anharmonicity $\\alpha$ = f12 - f01. Defaults to -0.25e9.
            lambda_1: Strength of 0-1 transition. Defaults to 1e9.
            lambda_2: Strength of 1-2 transition. Defaults to 0.8e9.
            gamma_1: Relaxation rate (1/T1) for 1-0. Defaults to 1e4.
            noise: Defaults to True. If True then T1 dissipation is included in the pulse-simulation.
                The strength is given by ``gamma_1``.
            atol: Absolute tolerance during solving.
            rtol: Relative tolerance during solving.
        """
        from qiskit_dynamics.pulse import InstructionToSignals

        qubit_frequency_02 = 2 * qubit_frequency + anharmonicity
        ket0 = np.array([[1, 0, 0]]).T
        ket1 = np.array([[0, 1, 0]]).T
        ket2 = np.array([[0, 0, 1]]).T

        sigma_m1 = ket0 @ ket1.T.conj()
        sigma_m2 = ket1 @ ket2.T.conj()

        sigma_p1 = sigma_m1.T.conj()
        sigma_p2 = sigma_m2.T.conj()

        p1 = ket1 @ ket1.T.conj()
        p2 = ket2 @ ket2.T.conj()

        drift = 2 * np.pi * (qubit_frequency * p1 + qubit_frequency_02 * p2)
        control = [
            2 * np.pi * (lambda_1 * (sigma_p1 + sigma_m1) + lambda_2 * (sigma_p2 + sigma_m2))
        ]
        r_frame = 2 * np.pi * qubit_frequency * (p1 + 2 * p2)
        t1_dissipator = np.sqrt(gamma_1) * sigma_m1

        self.anharmonicity = anharmonicity
        self.rabi_rate_01 = 8.589
        self.rabi_rate_12 = 6.876

        if noise is True:
            if version_is_at_least("qiskit-dynamics", "0.5.0"):
                solver_args = {
                    "array_library": "numpy",
                    "vectorized": True,
                }
            else:
                solver_args = {
                    "evaluation_mode": "dense_vectorized",
                }
            static_dissipators = [t1_dissipator]
        else:
            if version_is_at_least("qiskit-dynamics", "0.5.0"):
                solver_args = {
                    "array_library": "numpy",
                }
            else:
                solver_args = {
                    "evaluation_mode": "dense",
                }
            static_dissipators = None

        super().__init__(
            static_hamiltonian=drift,
            hamiltonian_operators=control,
            static_dissipators=static_dissipators,
            rotating_frame=r_frame,
            rwa_cutoff_freq=1.9 * qubit_frequency,
            rwa_carrier_freqs=[qubit_frequency],
            atol=atol,
            rtol=rtol,
            **solver_args,
            **kwargs,
        )

        self._defaults = PulseDefaults.from_dict(  # pylint: disable=no-member
            {
                "qubit_freq_est": [qubit_frequency / 1e9],
                "meas_freq_est": [0],
                "buffer": 0,
                "pulse_library": [],
                "cmd_def": [
                    Command.from_dict(
                        {
                            "name": "x",
                            "qubits": [0],
                            "sequence": [
                                PulseQobjInstruction(
                                    name="parametric_pulse",
                                    t0=0,
                                    ch="d0",
                                    label="Xp_d0",
                                    pulse_shape="drag",
                                    parameters={
                                        "amp": (0.5 + 0j) / self.rabi_rate_01,
                                        "beta": 5,
                                        "duration": 160,
                                        "sigma": 40,
                                    },
                                ).to_dict()
                            ],
                        }
                    ).to_dict(),
                    Command.from_dict(
                        {
                            "name": "sx",
                            "qubits": [0],
                            "sequence": [
                                PulseQobjInstruction(
                                    name="parametric_pulse",
                                    t0=0,
                                    ch="d0",
                                    label="X90p_d0",
                                    pulse_shape="drag",
                                    parameters={
                                        "amp": (0.25 + 0j) / self.rabi_rate_01,
                                        "beta": 5,
                                        "duration": 160,
                                        "sigma": 40,
                                    },
                                ).to_dict()
                            ],
                        }
                    ).to_dict(),
                ],
            }
        )
        self._target = Target(
            num_qubits=1,
            qubit_properties=[QubitProperties(frequency=qubit_frequency)],
            dt=self.dt,
            granularity=16,
        )

        measure_props = {
            (0,): InstructionProperties(duration=0, error=0),
        }
        x_props = {
            (0,): InstructionProperties(duration=160e-10, error=0),
        }
        sx_props = {
            (0,): InstructionProperties(duration=160e-10, error=0),
        }
        rz_props = {
            (0,): InstructionProperties(duration=0.0, error=0),
        }
        self._phi = Parameter("phi")
        self._target.add_instruction(Measure(), measure_props)
        self._target.add_instruction(XGate(), x_props)
        self._target.add_instruction(SXGate(), sx_props)
        self._target.add_instruction(RZGate(self._phi), rz_props)

        self.converter = InstructionToSignals(self.dt, carriers={"d0": qubit_frequency})

        default_schedules = [
            self._defaults.instruction_schedule_map.get("x", (0,)),
            self._defaults.instruction_schedule_map.get("sx", (0,)),
        ]
        self._simulated_pulse_unitaries = {
            (schedule.name, (0,), ()): self.solve(schedule, (0,)) for schedule in default_schedules
        }
