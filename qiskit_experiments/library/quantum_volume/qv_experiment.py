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
import numpy as np
from numpy.random import Generator, default_rng
from numpy.random.bit_generator import BitGenerator, SeedSequence

from qiskit.utils.optionals import HAS_AER

from qiskit import QuantumCircuit
from qiskit.circuit.library import quantum_volume
from qiskit import transpile
from qiskit.providers.backend import Backend
from qiskit_experiments.framework import BaseExperiment, Options
from .qv_analysis import QuantumVolumeAnalysis


class QuantumVolume(BaseExperiment):
    """An experiment to measure the largest random square circuit that can be run on a processor.

    # section: overview
        Quantum Volume (QV) is a single-number metric that can be measured using a concrete protocol
        on near-term quantum computers of modest size. The QV method quantifies the largest random
        circuit of equal width and depth that the computer successfully implements.
        Quantum computing systems with high-fidelity operations, high connectivity,
        large calibrated gate sets, and circuit rewriting toolchains are expected to
        have higher quantum volumes.

        The Quantum Volume is determined by the largest circuit depth :math:`d_{max}`,
        and equals to :math:`2^{d_{max}}`.
        See the `Qiskit Textbook
        <https://github.com/Qiskit/textbook/blob/main/notebooks/quantum-hardware/measuring-quantum-volume.ipynb>`_
        for an explanation on the QV protocol.

        In the QV experiment we generate :class:`~qiskit.circuit.library.QuantumVolume` circuits on
        :math:`d` qubits, which contain :math:`d` layers, where each layer consists of random 2-qubit
        unitary gates from :math:`SU(4)`, followed by a random permutation on the :math:`d` qubits.
        Then these circuits run on the quantum backend and on an ideal simulator (either
        :class:`~qiskit_aer.AerSimulator` or :class:`~qiskit.quantum_info.Statevector`).

        A depth :math:`d` QV circuit is successful if it has `mean heavy-output probability` > 2/3 with
        confidence level > 0.977 (corresponding to z_value = 2), and at least 100 trials have been ran.

        See :class:`QuantumVolumeAnalysis` documentation for additional
        information on QV experiment analysis.

    # section: analysis_ref
        :class:`QuantumVolumeAnalysis`

    # section: manual
        :doc:`/manuals/verification/quantum_volume`

    # section: reference
        .. ref_arxiv:: 1 1811.12926
        .. ref_arxiv:: 2 2008.08571

    # section: example
        .. jupyter-execute::
            :hide-code:

            # backend
            from qiskit_aer import AerSimulator
            from qiskit_ibm_runtime.fake_provider import FakeSydneyV2
            backend = AerSimulator.from_backend(FakeSydneyV2())

        .. jupyter-execute::

            from qiskit_experiments.framework import BatchExperiment
            from qiskit_experiments.library import QuantumVolume

            qubits = tuple(range(4)) # Can use specific qubits. for example [2, 4, 7, 10]
            qv_exp = QuantumVolume(qubits, seed=42)
            qv_exp.set_transpile_options(optimization_level=3)
            qv_exp.set_run_options(shots=1000)

            expdata = qv_exp.run(backend=backend).block_for_results()
            display(expdata.figure(0))

            display(expdata.analysis_results(dataframe=True))
    """

    def __init__(
        self,
        physical_qubits: Sequence[int],
        backend: Optional[Backend] = None,
        trials: Optional[int] = 100,
        seed: Optional[Union[int, SeedSequence, BitGenerator, Generator]] = None,
        simulation_backend: Optional[Backend] = None,
    ):
        """Initialize a quantum volume experiment.

        Args:
            physical_qubits: list of physical qubits for the experiment.
            backend: Optional, the backend to run the experiment on.
            trials: The number of trials to run the quantum volume circuit.
            seed: Optional, seed used to initialize ``numpy.random.default_rng``
                  when generating circuits. The ``default_rng`` will be initialized
                  with this seed value every time :meth:`circuits` is called.
            simulation_backend: The simulator backend to use to generate
                the expected results. the simulator must have a 'save_probabilities'
                method. If None, the :class:`qiskit_aer.AerSimulator` simulator will be used
                (in case :external+qiskit_aer:doc:`qiskit-aer <index>` is not
                installed, :class:`qiskit.quantum_info.Statevector` will be used).
        """
        super().__init__(physical_qubits, analysis=QuantumVolumeAnalysis(), backend=backend)

        # Set configurable options
        self.set_experiment_options(trials=trials, seed=seed)

        if not simulation_backend and HAS_AER:
            from qiskit_aer import AerSimulator

            self._simulation_backend = AerSimulator()
        else:
            self._simulation_backend = simulation_backend

    @classmethod
    def _default_experiment_options(cls) -> Options:
        """Default experiment options.

        Experiment Options:
            trials (int): Optional, number of times to generate new Quantum Volume
                circuits and calculate their heavy output.
            seed (None or int or SeedSequence or BitGenerator or Generator): A seed
                used to initialize ``numpy.random.default_rng`` when generating circuits.
                The ``default_rng`` will be initialized with this seed value every time
                :meth:`circuits` is called.
        """
        options = super()._default_experiment_options()

        options.trials = 100
        options.seed = None

        return options

    def _get_ideal_data(self, circuits: List[QuantumCircuit], **run_options) -> List[List[float]]:
        """Return ideal measurement probabilities.

        In case the user does not have Aer installed, use Qiskit's quantum info module
        to calculate the ideal state.

        Args:
            circuits: the circuits to extract the ideal data from
            run_options: backend run options.

        Returns:
            list: list of lists of the probabilities for each state in each circuit.
        """
        # NOTE: this code used to process a single circuit at a time but
        # AerSimulator() generates a backend object that regenerates its Target
        # on the fly each time and the overhead of regenerating the target for
        # each transpile call was significant.
        if self._simulation_backend:
            circuits = [c.copy() for c in circuits]
            for circuit in circuits:
                circuit.save_probabilities()
            # always transpile with optimization_level 0, even if the non ideal circuits will run
            # with different optimization level, because we need to compare the results to the
            # exact generated probabilities
            t_circuits = transpile(circuits, self._simulation_backend, optimization_level=0)

            result = self._simulation_backend.run(t_circuits, **run_options).result()
            probabilities = [
                result.data(i).get("probabilities").tolist() for i, _ in enumerate(t_circuits)
            ]
        else:
            from qiskit.quantum_info import Statevector

            probabilities = [Statevector(c).probabilities().tolist() for c in circuits]
        return probabilities

    def circuits(self) -> List[QuantumCircuit]:
        """Return a list of Quantum Volume circuits.

        Returns:
            A list of :class:`QuantumCircuit`.
        """
        rng = default_rng(seed=self.experiment_options.seed)
        circuits = []
        depth = self._num_qubits

        # Note: the trials numbering in the metadata is starting from 1 for each new experiment run
        for trial in range(1, self.experiment_options.trials + 1):
            # Maximum possible seed to send to quantum_volume()
            # This is a workound that can be replaced with seed=rng once we
            # drop support for qiskit<2.2
            # See https://github.com/Qiskit/qiskit/pull/14586
            max_value = np.iinfo(np.int64).max
            qv_circ = quantum_volume(depth, depth, seed=rng.integers(max_value, dtype=np.int64))
            qv_circ.metadata = {
                "depth": depth,
                "trial": trial,
            }
            circuits.append(qv_circ)
        all_probs = self._get_ideal_data(circuits)
        for qv_circ, probs in zip(circuits, all_probs):
            qv_circ.measure_active()
            qv_circ.metadata["ideal_probabilities"] = probs

        return circuits
