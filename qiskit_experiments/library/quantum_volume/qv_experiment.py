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

from typing import Union, Sequence, Optional, List
from numpy.random import Generator, default_rng

try:
    from qiskit import Aer

    HAS_SIMULATION_BACKEND = True
except ImportError:
    HAS_SIMULATION_BACKEND = False

from qiskit import QuantumCircuit
from qiskit.circuit.library import QuantumVolume as QuantumVolumeCircuit
from qiskit import transpile
from qiskit.providers.backend import Backend
from qiskit_experiments.framework import BaseExperiment, Options, fix_class_docs
from .qv_analysis import QuantumVolumeAnalysis


@fix_class_docs
class QuantumVolume(BaseExperiment):
    """Quantum Volume Experiment class.

    # section: overview
        Quantum Volume (QV) is a single-number metric that can be measured using a concrete protocol
        on near-term quantum computers of modest size. The QV method quantifies the largest random
        circuit of equal width and depth that the computer successfully implements.
        Quantum computing systems with high-fidelity operations, high connectivity,
        large calibrated gate sets, and circuit rewriting toolchains are expected to
        have higher quantum volumes.

        The Quantum Volume is determined by the largest circuit depth :math:`d_{max}`,
        and equals to :math:`2^{d_{max}}`.
        See `Qiskit Textbook
        <https://qiskit.org/textbook/ch-quantum-hardware/measuring-quantum-volume.html>`_
        for an explanation on the QV protocol.

        In the QV experiment we generate `QV circuits
        <https://qiskit.org/documentation/stubs/qiskit.circuit.library.QuantumVolume.html>`_
        on :math:`d` qubits, which contain :math:`d` layers, where each layer consists of random 2-qubit
        unitary gates from :math:`SU(4)`, followed by a random permutation on the :math:`d` qubits.
        Then these circuits run on the quantum backend and on an ideal simulator
        (either :class:`AerSimulator` or :class:`qiskit.quantum_info.Statevector`).

        A depth :math:`d` QV circuit is successful if it has 'mean heavy-output probability' > 2/3 with
        confidence level > 0.977 (corresponding to z_value = 2), and at least 100 trials have been ran.

        See :class:`QuantumVolumeAnalysis` documentation for additional
        information on QV experiment analysis.

    # section: reference
        .. ref_arxiv:: 1 1811.12926
        .. ref_arxiv:: 2 2008.08571

    """

    # Analysis class for experiment
    __analysis_class__ = QuantumVolumeAnalysis

    def __init__(
        self,
        qubits: Sequence[int],
        backend: Optional[Backend] = None,
        trials: Optional[int] = 100,
        seed: Optional[Union[int, Generator]] = None,
        simulation_backend: Optional[Backend] = None,
    ):
        """Initialize a quantum volume experiment.

        Args:
            qubits: list of physical qubits for the experiment.
            backend: Optional, the backend to run the experiment on.
            trials: The number of trials to run the quantum volume circuit.
            seed: Seed or generator object for random number
                  generation. If None default_rng will be used.
            simulation_backend: The simulator backend to use to generate
                the expected results. the simulator must have a 'save_probabilities'
                method. If None :class:`AerSimulator` simulator will be used
                (in case :class:`AerSimulator` is not
                installed :class:`qiskit.quantum_info.Statevector` will be used).
        """
        super().__init__(qubits, backend=backend)

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
    def _default_experiment_options(cls) -> Options:
        """Default experiment options.

        Experiment Options:
            trials (int): Optional, number of times to generate new Quantum Volume
                circuits and calculate their heavy output.
        """
        options = super()._default_experiment_options()

        options.trials = 100

        return options

    def _get_ideal_data(self, circuit: QuantumCircuit, **run_options) -> List[float]:
        """Return ideal measurement probabilities.

        In case the user does not have Aer installed use Terra to calculate
        the ideal state.

        Args:
            circuit: the circuit to extract the ideal data from
            run_options: backend run options.

        Returns:
            list: list of the probabilities for each state in the circuit (as Numpy array)
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

    def circuits(self) -> List[QuantumCircuit]:
        """Return a list of Quantum Volume circuits.

        Returns:
            A list of :class:`QuantumCircuit`.
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
