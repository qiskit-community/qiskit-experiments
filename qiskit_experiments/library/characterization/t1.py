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
T1 Experiment class.
"""

from typing import List, Optional, Union, Sequence
import numpy as np

from qiskit import QuantumCircuit
from qiskit.providers.backend import Backend
from qiskit_experiments.framework import BackendTiming, BaseExperiment, Options
from qiskit_experiments.warnings import qubit_deprecate
from qiskit_experiments.library.characterization.analysis.t1_analysis import T1Analysis


class T1(BaseExperiment):
    r"""T1 experiment.

    # section: overview

        Design and analyze experiments for estimating T\ :sub:`1` relaxation time of the qubit.

        Each experiment consists of the following steps:

        1. Circuits generation: the circuits set the qubit in the excited state,
        wait different time intervals, then measure the qubit.

        2. Backend execution: actually running the circuits on the device
        (or simulator).

        3. Analysis of results: deduction of T\ :sub:`1`\ , based on the outcomes,
        by fitting to an exponential curve.

    # section: analysis_ref
        :py:class:`.T1Analysis`

    """

    @classmethod
    def _default_experiment_options(cls) -> Options:
        """Default experiment options.

        Experiment Options:
            delays (Iterable[float]): Delay times of the experiments in seconds.
        """
        options = super()._default_experiment_options()
        options.delays = None
        return options

    @qubit_deprecate()
    def __init__(
        self,
        physical_qubits: Sequence[int],
        delays: Union[List[float], np.array],
        backend: Optional[Backend] = None,
    ):
        """
        Initialize the T1 experiment class

        Args:
            physical_qubits: a single-element sequence containing the qubit whose T1 is to be
                estimated
            delays: delay times of the experiments in seconds
            backend: Optional, the backend to run the experiment on.

        Raises:
            ValueError: if the number of delays is smaller than 3
        """
        # Initialize base experiment
        super().__init__(physical_qubits, analysis=T1Analysis(), backend=backend)

        # Set experiment options
        self.set_experiment_options(delays=delays)

    def circuits(self) -> List[QuantumCircuit]:
        """
        Return a list of experiment circuits

        Returns:
            The experiment circuits
        """
        timing = BackendTiming(self.backend)

        circuits = []
        for delay in self.experiment_options.delays:
            circ = QuantumCircuit(1, 1)
            circ.x(0)
            circ.barrier(0)
            circ.delay(timing.round_delay(time=delay), 0, timing.delay_unit)
            circ.barrier(0)
            circ.measure(0, 0)

            circ.metadata = {
                "experiment_type": self._type,
                "qubit": self.physical_qubits[0],
                "unit": "s",
            }
            circ.metadata["xval"] = timing.delay_time(time=delay)

            circuits.append(circ)

        return circuits

    def _metadata(self):
        metadata = super()._metadata()
        # Store measurement level and meas return if they have been
        # set for the experiment
        for run_opt in ["meas_level", "meas_return"]:
            if hasattr(self.run_options, run_opt):
                metadata[run_opt] = getattr(self.run_options, run_opt)
        return metadata
