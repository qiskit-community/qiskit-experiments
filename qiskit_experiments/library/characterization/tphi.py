# that they have been altered from the originals.
"""
Tphi Experiment class.
"""

from typing import List, Optional, Union
import numpy as np

from qiskit.providers import Backend
from qiskit_experiments.framework.composite.batch_experiment import BatchExperiment
from qiskit_experiments.library.characterization import T1, T2Ramsey
from qiskit_experiments.library.characterization.analysis.tphi_analysis import TphiAnalysis


class Tphi(BatchExperiment):
    r"""Tphi Experiment Class
    Tphi is defined as follows:

    .. math::
    1/T_{\phi} = 1/2T_1 + 1/T_{2*}


    """

    def __init__(
        self,
        qubit: int,
        delays_t1: List[Union[List[float], np.array]],
        delays_t2: List[Union[List[float], np.array]],
        unit: str = "s",
        osc_freq: float = 0.0,
        backend: Optional[Backend] = None,
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
            backend = Optional, the backend on which to run the experiment

        """
        self.exps = []
        self.exps.append(T1(qubit, delays_t1, unit))
        self.exps.append(T2Ramsey(qubit, delays_t2, unit, osc_freq))
        # Run batch experiments
        super().__init__(experiments=self.exps, analysis=TphiAnalysis(), backend=backend)
