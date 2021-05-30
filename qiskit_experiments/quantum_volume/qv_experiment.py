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
from qiskit.circuit.library import QuantumVolume
from qiskit import transpile, assemble
from qiskit import execute
from qiskit.exceptions import QiskitError
from qiskit.providers.basebackend import BaseBackend as LegacyBackend
from qiskit_experiments.base_experiment import BaseExperiment
from qiskit_experiments.experiment_data import ExperimentData
from .qv_analysis import QVAnalysis


class QVExperiment(BaseExperiment):
    """Quantum Volume Experiment class

    Experiment Options:
        trials: number of times to generate new Quantum Volume circuits and calculate their heavy
        output.
    """

    # Analysis class for experiment
    __analysis_class__ = QVAnalysis

    # ExperimentData class for the simulations
    __simulation_data__ = ExperimentData

    def __init__(
        self,
        qubits: Union[int, Iterable[int]],
        trials: Optional[int] = 1,
        seed: Optional[Union[int, Generator]] = None,
        simulation_backend: Optional[Backend] = None,
    ):
        """Standard randomized benchmarking experiment
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

        # Set fixed options
        self._previous_trials = 0
        self._simulation_data = None

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
        return Options(trials=1)

    # pylint: disable = arguments-differ
    def run(
        self,
        backend: "Backend",
        analysis: bool = True,
        experiment_data: Optional[ExperimentData] = None,
        simulation_data: Optional[ExperimentData] = None,
        **run_options,
    ) -> ExperimentData:
        """Run an experiment and perform analysis.
        Args:
            backend (Backend): The backend to run the experiment on.
            analysis: If True run analysis on experiment data.
            experiment_data (ExperimentData): Optional, add results to existing
                experiment data. If None a new ExperimentData object will be
                returned. if given, must proved also simulation_data.
            simulation_data (ExperimentData): Optional, add results to existing
                simulation data. must be used when adding results to existing experiment data.
                If None a new ExperimentData object will be created.
            run_options: backend runtime options used for circuit execution.

        Returns:
            ExperimentData: the experiment data object.

        Raises:
            QiskitError: if experiment data is given but simulation data is not given, or vise versa.
            QiskitError: if experiment data and simulation data does not have the same data length.
        """
        if (experiment_data or simulation_data) and not (experiment_data and simulation_data):
            raise QiskitError(
                "Quantum Volume experiment must have none or both experiment data"
                " and simulation data"
            )
        # Create new experiment data
        if experiment_data is None:
            experiment_data = self.__experiment_data__(self)
        # Create new simulation data
        if simulation_data is None:
            simulation_data = self.__simulation_data__(self)
        if len(experiment_data.data()) != len(simulation_data.data()):
            raise QiskitError(
                "Quantum Volume experiment must have experiment data and simulation data "
                "with the same length"
            )
        # count the number of previous trials.
        # assuming that all the data in experiment data is QV data.
        self._previous_trials = len(experiment_data.data())

        # ideal circuits are used for getting the ideal state using the simulator, without the
        # changes that the transpiler might add to the circuit in order to improve it's performance
        circuits = self.circuits(backend)
        ideal_circuits = []
        for circ in circuits:
            circ.metadata["is_simulation"] = False
            # return new circuit without the measurements
            ideal_circuits.append(circ.remove_final_measurements(inplace=False))
            ideal_circuits[-1].metadata["is_simulation"] = True
        # transpile circuits
        circuits = transpile(circuits, backend, **self.transpile_options.__dict__)

        # Run circuits on backend
        run_opts = copy.copy(self.run_options)
        run_opts.update_options(**run_options)
        run_opts = run_opts.__dict__

        if isinstance(backend, LegacyBackend):
            qobj = assemble(circuits, backend=backend, **run_opts)
            job = backend.run(qobj)
        else:
            job = backend.run(circuits, **run_opts)

        # Add Jobs to ExperimentData
        experiment_data.add_data(job)

        sim_data = self._get_ideal_data(ideal_circuits, **run_opts)
        # Add Jobs to the simulation data
        simulation_data.add_data(sim_data)

        # Queue analysis of data for when job is finished
        if analysis and self.__analysis_class__ is not None:
            self.run_analysis(experiment_data, simulation_data=simulation_data)

        self._simulation_data = simulation_data
        # Return the ExperimentData future
        return experiment_data

    def add_trials(self, additional_trials):
        """
        Add more trials to the experiment
        Args:
            additional_trials (int): The amount of trials to add
        """
        new_trials = self.experiment_options.trials + additional_trials
        self.set_experiment_options(trials=new_trials)

    @property
    def trials(self):
        """Return number of trials in the experiment"""
        return self.experiment_options.trials

    @property
    def simulation_data(self):
        """Return the ideal data of the experiment"""
        return self._simulation_data

    def _get_ideal_data(self, circuits, **run_options):
        """
        in case the user does not have aer installed - use Terra to calculate the ideal state
        Args:
            circuits: the circuits to extract the ideal data from
        Returns:
            dict: data object with the circuit's metadata
                  and the probability for each state in each circuit
        """
        if self._simulation_backend:
            for circuit in circuits:
                circuit.save_probabilities()
            return execute(circuits, backend=self._simulation_backend, **run_options)
        else:
            from qiskit.quantum_info import Statevector

            sim_obj = []
            for circuit in circuits:
                state_vector = Statevector(circuit)
                sim_data = {
                    "probabilities": state_vector.probabilities(),
                    "metadata": circuit.metadata,
                }
                sim_obj.append(sim_data)
            return sim_obj

    def circuits(self, backend: Optional[Backend] = None) -> List[QuantumCircuit]:
        """Return a list of QV circuits, without the measurement instruction
        Args:
            backend (Backend): Optional, a backend object.
        Returns:
            List[QuantumCircuit]: A list of :class:`QuantumCircuit`s.
        """
        circuits = []
        depth = self._num_qubits

        # continue the trials numbers from previous experiments runs
        for trial in range(self._previous_trials + 1, self.experiment_options.trials + 1):
            qv_circ = QuantumVolume(depth, depth, seed=self._rng)
            qv_circ.measure_active()
            qv_circ.metadata = {
                "experiment_type": self._type,
                "depth": depth,
                "trial": trial,
                "qubits": self.physical_qubits,
            }
            circuits.append(qv_circ)

        return circuits
