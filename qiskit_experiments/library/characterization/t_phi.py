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
from qiskit_experiments.framework.composite.composite_experiment import CompositeExperiment
from qiskit_experiments.library.characterization import T1, T2Ramsey
from qiskit_experiments.library.characterization.t_phi_analysis import TphiAnalysis

class Tphi(CompositeExperiment):
    """Tphi Experiment Class"""

    __analysis_class__ = TphiAnalysis
    
    def __init__(self,
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
            delays_t2: delay times of the T2*1 experiment
            unit: Optional, time unit of `delays`.
            Supported units: 's', 'ms', 'us', 'ns', 'ps', 'dt'.
            The unit is used for both experiments
            osc_freq: the oscillation frequency induced using by the user for T2
            experiment_type: String indicating the experiment type.

        """
        self._qubit = qubit
        self._delays_t1 = delays_t1
        self._delays_t2 = delays_t2
        self._unit = unit
        self._osc_freq = osc_freq
        
        self._expT1 = T1(self._qubit, self._delays_t1, self._unit)
        self._expT2 = T2Ramsey(self._qubit, self._delays_t2, self._unit, self._osc_freq)

        super().__init__(experiments=[self._expT1, self._expT2],
                         qubits= self._qubit,
                         experiment_type=experiment_type)

    def circuits(self, backend: Optional[Backend] = None) -> List[QuantumCircuit]:
        """
        Return a dictionary of experiment circuits, those of 'T1' and those of 'T2*'
        Args:
            backend: Optional, a backend object

        Returns:
            The experiment circuits
            """
        circuits = {}
        circuits['T1'] = self._expT1.circuits()
        circuits['T2*'] = self._expT2.circuits()
        return circuits

    def run(self, backends, experiment_data, **run_options):
        expdata = {}
        expdata['T1'] = super().run(self, backends['T1'], experiment_data, **run_options)
        expdata['T2*'] = super().run(self, backends['T2*'], experiment_data, **run_options)
