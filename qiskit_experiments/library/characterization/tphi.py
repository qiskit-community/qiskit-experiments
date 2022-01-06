# that they have been altered from the originals.
"""
Tphi Experiment class.
"""

from typing import List, Optional, Union
import numpy as np
from enum import Enum

import qiskit
from qiskit.providers import Backend
from qiskit.circuit import QuantumCircuit
from qiskit_experiments.framework import BaseExperiment
from qiskit_experiments.framework.composite.batch_experiment import BatchExperiment
from qiskit_experiments.library.characterization import T1, T2Ramsey
from qiskit_experiments.library.characterization.analysis.tphi_analysis import TphiAnalysis


class Tphi(BatchExperiment):
    """Tphi Experiment Class"""

    __analysis_class__ = TphiAnalysis

    def __init__(
        self,
        qubit: int,
        delays_t1: List[Union[List[float], np.array]],
        delays_t2: List[Union[List[float], np.array]],
        unit: str = "s",
        osc_freq: float = 0.0,
        experiment_type: Optional[str] = None,
    ):
        """Initialize the experiments object.

        Args:
            qubit: the qubit under test
            delays_t1: delay times of the T1 experiment
            delays_t2: delay times of the T2* experiment
            unit: Optional, time unit of `delays`.
            Supported units: 's', 'ms', 'us', 'ns', 'ps', 'dt'.
            The unit is used for both experiments
            osc_freq: the oscillation frequency induced using by the user for T2
            experiment_type: String indicating the experiment type.

        """
        # self._qubit = qubit
        self._delays_t1 = delays_t1
        self._delays_t2 = delays_t2
        self._unit = unit
        self._osc_freq = osc_freq

        expT1 = T1(qubit, self._delays_t1, self._unit)
        expT2 = T2Ramsey(qubit, self._delays_t2, self._unit, self._osc_freq)
        exps = []
        exps.append(expT1)
        exps.append(expT2)

        # Run batch experiments
        batch_exp = super().__init__(exps)

    def run(self, backend, experiment_data, **run_options):
        expdata = super().run(backend, shots=1000)
