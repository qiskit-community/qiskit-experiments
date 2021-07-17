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
Quantum Volume Experiment class.
"""

from typing import Union, Iterable, Optional, List
from numpy.random import Generator, default_rng
from qiskit.providers.backend import Backend
from qiskit.providers.options import Options

try:
    from qiskit import Aer

    HAS_SIMULATION_BACKEND = True
except ImportError:
    HAS_SIMULATION_BACKEND = False

from qiskit import QuantumCircuit
from qiskit.circuit.library import QuantumVolume as QuantumVolumeCircuit
from qiskit import transpile
from qiskit_experiments.base_experiment import BaseExperiment
from .qv_analysis import QuantumVolumeAnalysis


class QuantumVolume(BaseExperiment):
    """Quantum Volume Experiment class

    Experiment Options:
        trials (int): Optional, number of times to generate new Quantum Volume circuits and
                    calculate their heavy output.
    """

    # Analysis class for experiment
    __analysis_class__ = QuantumVolumeAnalysis

    def __init__(
        self,
        qubits: Union[int, Iterable[int]],
        trials: Optional[int] = 100,
        seed: Optional[Union[int, Generator]] = None,
        simulation_backend: Optional[Backend] = None,
    ):
        """Quantum Volume experiment
        Args:
            qubits: the number of qubits or list of
                    physical qubits for the experiment.
            trials: number of trials to run the quantum volume circuit.
            seed: Seed or generator object for random number
                  generation. If None default_rng will be used.
            simulation_backend: the simulator backend to use to generate
                the expected results. the simulator must have a 'save_probabilities' method.
                if None Aer simulator will be used (in case Aer is not installed -
                qiskit.quantum_info will be used).
        """
        super().__init__(qubits)

        # Set configurable options
        self.set_experiment_options(trials=trials)

        if not isinstance(seed, Generator):
            self._rng = default_rng(seed=seed)
        else:
            self._rng = seed

        if not simulation_backend and HAS_SIMULATION_BACKEND:
            self._simulation_backend = Aer.get_backend("aer_simulator")
        else:
            self._simulation_backend = simulation_backend

    @classmethod
    def _default_experiment_options(cls):
        return Options(trials=100)

    def _get_ideal_data(self, circuit, **run_options):
        """
        In case the user does not have Aer installed - use Terra to calculate the ideal state
        Args:
            circuit: the circuit to extract the ideal data from
        Returns:
            dict: the probability for each state in the circuit
        """
        ideal_circuit = circuit.remove_final_measurements(inplace=False)
        if self._simulation_backend:
            ideal_circuit.save_probabilities()
            # always transpile with optimization_level 0, even if the non ideal circuits will run
            # with different optimization level, because we need to compare the results to the
            # exact generated probabilities
            ideal_circuit = transpile(ideal_circuit, self._simulation_backend, optimization_level=0)

            ideal_result = self._simulation_backend.run(ideal_circuit, **run_options).result()
            probabilities = ideal_result.data().get("probabilities")
        else:
            from qiskit.quantum_info import Statevector

            state_vector = Statevector(ideal_circuit)
            probabilities = state_vector.probabilities()
        return probabilities

    def circuits(self, backend: Optional[Backend] = None) -> List[QuantumCircuit]:
        """Return a list of QV circuits, without the measurement instruction
        Args:
            backend (Backend): Optional, a backend object.
        Returns:
            List[QuantumCircuit]: A list of :class:`QuantumCircuit`s.
        """
        circuits = []
        depth = self._num_qubits

        # Note: the trials numbering in the metadata is starting from 1 for each new experiment run
        for trial in range(1, self.experiment_options.trials + 1):
            qv_circ = QuantumVolumeCircuit(depth, depth, seed=self._rng)
            qv_circ.measure_active()
            qv_circ.metadata = {
                "experiment_type": self._type,
                "depth": depth,
                "trial": trial,
                "qubits": self.physical_qubits,
                "ideal_probabilities": self._get_ideal_data(qv_circ),
            }
            circuits.append(qv_circ)
        return circuits
