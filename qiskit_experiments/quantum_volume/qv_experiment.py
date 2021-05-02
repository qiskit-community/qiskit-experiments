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

import copy
from typing import Union, Iterable, Optional
from numpy.random import Generator, default_rng

try:
    from qiskit.providers import BaseBackend
    HAS_SIMULATION_BACKEND = True
except ImportError:
    HAS_SIMULATION_BACKEND = False

from qiskit.circuit.library import QuantumVolume
from qiskit import assemble

from qiskit_experiments.base_experiment import BaseExperiment
from qiskit_experiments.base_experiment import _TRANSPILE_OPTIONS
from qiskit_experiments.experiment_data import ExperimentData
from .qv_analysis import QVAnalysis


class QVExperiment(BaseExperiment):
    """RB Experiment class"""

    # Analysis class for experiment
    __analysis_class__ = QVAnalysis

    # ExperimentData class for the simulations
    __simulation_data__ = ExperimentData

    _trials = 0

    def __init__(
        self,
        qubits: Union[int, Iterable[int]],
        trials: Optional[int] = 1,
        seed: Optional[Union[int, Generator]] = None,
        simulation_backend: Optional[BaseBackend] = None,
    ):
        """Standard randomized benchmarking experiment
        Args:
            qubits: the number of qubits or list of
                    physical qubits for the experiment.
            trials: number of trials to run the quantum volume circuit.
            seed: Seed or generator object for random number
                  generation. If None default_rng will be used.
            simulation_backend: the simulator backend to use to generate
                the expected results. if None BaseBackend will be used.
        """
        if not isinstance(seed, Generator):
            self._rng = default_rng(seed=seed)
        else:
            self._rng = seed
        self._trials = trials
        self._previous_trials = 0
        if not simulation_backend and HAS_SIMULATION_BACKEND:
            self._simulation_backend = BaseBackend
        else:
            self._simulation_backend = simulation_backend
        super().__init__(qubits)

    def run(self, backend, experiment_data=None, **kwargs):
        """Run an experiment and perform analysis.
        Args:
            backend (Backend): The backend to run the experiment on.
            experiment_data (ExperimentData): Optional, add results to existing
                experiment data. If None a new ExperimentData object will be
                returned.
            kwargs: keyword arguments for self.circuit,
                    qiskit.transpile, and backend.run.
        Returns:
            ExperimentData: the experiment data object.
        """
        # Create new experiment data
        if experiment_data is None:
            experiment_data = self.__experiment_data__(self)
        else:
            # count the number of previous trails. assuming that all the data in experiment data is QV data.
            # divide by the depth (num_qubits) and by 2 (because for each trial there is also simulation data)
            self._previous_trials = (len(experiment_data.data) / self._num_qubits) / 2
            self._trials += self._previous_trials

        # Filter kwargs
        run_options = self.__run_defaults__.copy()
        circuit_options = {}
        for key, value in kwargs.items():
            if key in _TRANSPILE_OPTIONS or key in self._circuit_options:
                circuit_options[key] = value
            else:
                run_options[key] = value

        # Generate and run circuits
        circuits = self.transpiled_circuits(backend, **circuit_options)
        backend_circuits = copy.deepcopy(circuits)
        for circ in backend_circuits:
            circ.metadata["is_simulation"] = False
            circ.measure_active()
        qobj = assemble(backend_circuits, backend, **run_options)
        job = backend.run(qobj)

        sim_data = self._get_ideal_data(circuits, run_options)
        # Add Jobs to ExperimentData
        experiment_data.add_data([job, sim_data])

        # Queue analysis of data for when job is finished
        if self.__analysis_class__ is not None:
            # pylint: disable = not-callable
            self.__analysis_class__().run(experiment_data, **kwargs)

        # Return the ExperimentData future
        return experiment_data

    def trials(self):
        """Return number of trials in the experiment"""
        return self._trials

    # pylint: disable = arguments-differ
    def circuits(self, backend=None):
        """Return a list of QV circuits, without the measurement instruction
        Args:
            backend (Backend): Optional, a backend object.
        Returns:
            List[QuantumCircuit]: A list of :class:`QuantumCircuit`s.
        """
        circuits = []

        for depth in range(1, self._num_qubits + 1):
            # continue the trials numbers from previous experiments runs
            for trial in range(self._previous_trials, self._trials):
                qv_circ = QuantumVolume(depth, depth, seed=self._rng)
                qv_circ.metadata = {
                    "experiment_type": self._type,
                    "depth": depth,
                    "trial": trial,
                    "qubits": self.physical_qubits
                }
                circuits.append(qv_circ)

        return circuits

    def _get_ideal_data(self, circuits, run_options):
        """
        in case the user do not have aer installed - use Terra to calculate the ideal state
        Args:
            circuits: the circuits to extract the ideal data from
        Returns:
            dict: data object with the circuit's metadata
                  and the probability for each state in each circuit
        """
        if self._simulation_backend:
            for circuit in circuits:
                circuit.metadata["is_simulation"] = True
                circuit.save_probabilities()
            sim_obj = assemble(circuits, self._simulation_backend, **run_options)
            return self._simulation_backend.run(sim_obj)
        else:
            from qiskit.quantum_info import Statevector
            import numpy as np
            sim_obj = []
            for circuit in circuits:
                circuit.metadata["is_simulation"] = True
                state_vector = Statevector(circuit)
                prob_vector = np.multiply(state_vector, state_vector.conjugate())
                sim_data = {'probabilities': prob_vector,
                            'metadata': circuit.metadata}
                sim_obj.append(sim_data)
            return sim_obj
