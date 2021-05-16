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

from typing import Union, Iterable, Optional
from numpy.random import Generator, default_rng
from qiskit.providers import BaseBackend

try:
    from qiskit import Aer

    HAS_SIMULATION_BACKEND = True
except ImportError:
    HAS_SIMULATION_BACKEND = False

from qiskit.circuit.library import QuantumVolume
from qiskit import transpile, assemble
from qiskit import execute
from qiskit.exceptions import QiskitError

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
                the expected results. the simulator must have a 'save_probabilities' method.
                if None Aer simulator will be used (in case Aer is not installed - qiskit-info
                will be used).
        """
        if not isinstance(seed, Generator):
            self._rng = default_rng(seed=seed)
        else:
            self._rng = seed
        self._trials = trials
        self._previous_trials = 0
        if not simulation_backend and HAS_SIMULATION_BACKEND:
            self._simulation_backend = Aer.get_backend("aer_simulator")
        else:
            self._simulation_backend = simulation_backend
        super().__init__(qubits)

    def run(
        self,
        backend: "Backend",
        analysis: bool = True,
        experiment_data: Optional[ExperimentData] = None,
        **kwargs,
    ):
        """Run an experiment and perform analysis.
        Args:
            backend (Backend): The backend to run the experiment on.
            analysis: If True run analysis on experiment data.
            experiment_data (ExperimentData): Optional, add results to existing
                experiment data. If None a new ExperimentData object will be
                returned.
            kwargs: keyword arguments for self.circuit,
                    qiskit.transpile, and backend.run.
        Returns:
            ExperimentData: the experiment data object.
            tuple: If ``return_figures=True`` the output is a pair
                   ``(ExperimentData, figures)`` where ``figures`` is a list of figures.
        """
        # Create new experiment data
        if experiment_data is None:
            experiment_data = self.__experiment_data__(self)
        else:
            # count the number of previous trails.
            # assuming that all the data in experiment data is QV data.
            # divide by 2 (because for each trial there is also simulation data)
            self._previous_trials = int(len(experiment_data.data()) / 2)

        # Filter kwargs
        run_options = self.__run_defaults__.copy()
        circuit_options = {}
        for key, value in kwargs.items():
            if key in _TRANSPILE_OPTIONS or key in self._circuit_options:
                circuit_options[key] = value
            else:
                run_options[key] = value

        # Generate and run circuits
        transpiled_circuits, circuits = self.transpiled_circuits(backend, **circuit_options)
        for circ in transpiled_circuits:
            circ.metadata["is_simulation"] = False
            circ.measure_active()
        qobj = assemble(transpiled_circuits, backend, **run_options)
        job = backend.run(qobj)

        sim_data = self._get_ideal_data(circuits, run_options)
        # Add Jobs to ExperimentData
        experiment_data.add_data([job, sim_data])

        # use 'return_figures' parameter if given
        return_figures = kwargs.get("return_figures", False)
        # Queue analysis of data for when job is finished
        if self.__analysis_class__ is not None:
            if return_figures:
                # pylint: disable = not-callable
                _, figures = self.__analysis_class__().run(experiment_data, **kwargs)
            else:
                self.__analysis_class__().run(experiment_data, **kwargs)

        # Return the ExperimentData future
        if return_figures:
            return experiment_data, figures
        return experiment_data

    def add_trials(self, additional_trials):
        """
        Add more trials to the experiment
        Args:
            additional_trials (int): The amount of trials to add
        """
        self._trials += additional_trials

    @property
    def trials(self):
        """Return number of trials in the experiment"""
        return self._trials

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
            return execute(circuits, backend=self._simulation_backend, **run_options)
        else:
            from qiskit.quantum_info import Statevector
            import numpy as np

            sim_obj = []
            for circuit in circuits:
                circuit.metadata["is_simulation"] = True
                state_vector = Statevector(circuit)
                prob_vector = np.multiply(state_vector, state_vector.conjugate())
                sim_data = {"probabilities": prob_vector, "metadata": circuit.metadata}
                sim_obj.append(sim_data)
            return sim_obj

    # pylint: disable = arguments-differ
    def circuits(self, backend=None):
        """Return a list of QV circuits, without the measurement instruction
        Args:
            backend (Backend): Optional, a backend object.
        Returns:
            List[QuantumCircuit]: A list of :class:`QuantumCircuit`s.
        """
        circuits = []
        depth = self._num_qubits

        # continue the trials numbers from previous experiments runs
        for trial in range(self._previous_trials + 1, self._trials + 1):
            qv_circ = QuantumVolume(depth, depth, seed=self._rng)
            qv_circ.metadata = {
                "experiment_type": self._type,
                "depth": depth,
                "trial": trial,
                "qubits": self.physical_qubits,
            }
            circuits.append(qv_circ)

        return circuits

    def transpiled_circuits(self, backend=None, **kwargs):
        """Return a list of experiment circuits, before and after transpilation.

        Args:
            backend (Backend): Optional, a backend object to use as the
                               argument for the :func:`qiskit.transpile`
                               function.
            kwargs: kwarg options for the :meth:`circuits` method, and
                    :func:`qiskit.transpile` function.

        Returns:
            List[QuantumCircuit]: A list of :class:`QuantumCircuit`s after transpile.
            List[QuantumCircuit]: A list of :class:`QuantumCircuit`s before transpile.

        Raises:
            QiskitError: if an initial layout is specified in the
                         kwarg options for transpilation. The initial
                         layout must be generated from the experiment.

        .. note:
            These circuits should be on qubits ``[0, .., N-1]`` for an
            *N*-qubit experiment. The circuits mapped to physical qubits
            are obtained via the :meth:`transpiled_circuits` method.
        """
        # Filter kwargs to circuit and transpile options
        circuit_options = {}
        transpile_options = self.__transpile_defaults__.copy()
        for key, value in kwargs.items():
            valid_key = False
            if key in self._circuit_options:
                circuit_options[key] = value
                valid_key = True
            if key in _TRANSPILE_OPTIONS:
                transpile_options[key] = value
                valid_key = True
            if not valid_key:
                raise QiskitError(
                    f"{key} is not a valid kwarg for" f" {self.circuits} or {transpile}"
                )
        # Generate circuits
        circuits = self.circuits(backend=backend, **circuit_options)
        transpiled_circuits = []
        for circuit in circuits:
            transpiled_circuits.append(circuit.copy())

        # Transpile circuits
        if "initial_layout" in transpile_options:
            raise QiskitError("Initial layout must be specified by the Experiement.")
        transpile_options["initial_layout"] = self.physical_qubits
        transpiled_circuits = transpile(transpiled_circuits, backend=backend, **transpile_options)

        return transpiled_circuits, circuits
