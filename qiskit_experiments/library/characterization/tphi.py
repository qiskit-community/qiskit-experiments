# that they have been altered from the originals.
"""
Tphi Experiment class.
"""

from typing import List, Optional, Union
import numpy as np

from qiskit.providers import Backend
from qiskit_experiments.framework import Options
from qiskit_experiments.framework.composite.batch_experiment import BatchExperiment
from qiskit_experiments.library.characterization import T1, T2Ramsey
from qiskit_experiments.library.characterization.analysis.tphi_analysis import TphiAnalysis


class Tphi(BatchExperiment):
    r"""Tphi Experiment Class

    # section: overview

        Tphi is defined as follows:

        .. math::

        1/T_{\phi} = 1/{2T_1} + 1/T_{2*}

    # section: analysis_ref
        :py:class:`TphiAnalysis`
    """

    @classmethod
    def _default_experiment_options(cls) -> Options:
        """Default experiment options.

        Experiment Options:
            delays_t1 (Iterable[float]): Delay times of the t1 experiment in seconds.
            delays_t2 (Iterable[float]): Delay times of the t2Ramsey experiment in seconds.
        """
        options = super()._default_experiment_options()
        options.delays_t1 = None
        options.delays_t2 = None
        return options

    def __init__(
        self,
        qubit: int,
        delays_t1: List[Union[List[float], np.array]],
        delays_t2: List[Union[List[float], np.array]],
        osc_freq: float = 0.0,
        backend: Optional[Backend] = None,
    ):
        """Initialize the experiments object.

        Args:
            qubit: the qubit under test
            delays_t1: delay times of the T1 experiment
            delays_t2: delay times of the T2* experiment
            osc_freq: the oscillation frequency induced using by the user for T2Ramsey
            backend = Optional, the backend on which to run the experiment

        """
        self.set_experiment_options = self._default_experiment_options()
        # Set experiment options
        self.set_experiment_options.delays_t1 = delays_t1
        self.set_experiment_options.delays_t2 = delays_t2

        self.exps = []
        self.exps.append(T1(qubit, self.set_experiment_options.delays_t1))
        self.exps.append(T2Ramsey(qubit, self.set_experiment_options.delays_t2, osc_freq))
        # Run batch experiments
        super().__init__(experiments=self.exps, analysis=TphiAnalysis(), backend=backend)
