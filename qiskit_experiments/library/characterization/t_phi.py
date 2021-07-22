# that they have been altered from the originals.
"""
T_phi Experiment class.
"""

from typing import List, Optional, Union
import numpy as np
from enum import Enum

import qiskit
from qiskit.providers import Backend
from qiskit.circuit import QuantumCircuit
from qiskit_experiments.framework.composite import CompositeExperiment
from qiskit_experiments.library import T1, T2Ramsey

class SubExp(Enum):
    T1 = 0
    T2 = 1

class T_phi():
    """T_phi Experiment Class"""
    
    def __init__(self,
                 qubit: int,
                 delays: List[Union[List[float], np.array]],
                 unit: str = "s",
                 osc_freq: float = 0.0,
                 experiment_type: Optional[str] = None,
                 ):
        """Initialize the experiments object.

        Args:
            qubit: the qubit under test
            delays: delay times of the T1 and for T2 experiments
            unit: Optional, time unit of `delays`.
            Supported units: 's', 'ms', 'us', 'ns', 'ps', 'dt'.
            The unit is used for both experiments
            osc_freq: the oscillation frequency induced using by the user for T2
            experiment_type: String indicating the experiment type.

        """
        self._qubit = qubit
        self._delays = delays
        self._unit = unit
        self._osc_freq = osc_freq
        super().__init__([qubit], experiment_type)
        
        expT1 = T1(self._qubit, self._delays[T1])
        expT2 = T2Ramsey(self._qubit, self._delays[T2], osc_freq)

        t_phi = CompositeExperiment(experiments=[expT1, expT2], qubits= self.qubit,
                                    experiment_type=experiment_type)
