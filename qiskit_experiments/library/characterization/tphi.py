# This code is part of Qiskit.
#
# (C) Copyright IBM 2021, 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
"""
Tphi Experiment class.
"""

from typing import List, Optional, Union
import numpy as np

from qiskit.providers import Backend
from qiskit.test.mock import FakeBackend
from qiskit_experiments.framework import Options
from qiskit_experiments.framework.composite.batch_experiment import BatchExperiment
from qiskit_experiments.library.characterization import T1, T2Ramsey
from qiskit_experiments.library.characterization.analysis.tphi_analysis import TphiAnalysis


class Tphi(BatchExperiment):
    r"""Tphi Experiment Class

    # section: overview

        Tphi is defined as follows:

        :math:`1/T_\phi = 1/T_{2*} - 1/2T_1`.

        For more details, see :py:class:`T1` and :py:class:`T2Ramsey`

    # section: analysis_ref
        :py:class:`TphiAnalysis`

    # section: reference
        .. ref_arxiv:: 1 1904.06560v5

    # section: tutorial
        :doc:`/tutorials/tphi_characterization`
    """

    @classmethod
    def _default_experiment_options(cls) -> Options:
        """Default experiment options.

        Experiment Options:
            delays_t1 (Iterable[float]): Delay times of the t1 experiment in seconds.
            delays_t2 (Iterable[float]): Delay times of the t2ramsey experiment in seconds.
        """
        options = super()._default_experiment_options()
        options.delays_t1 = None
        options.delays_t2 = None
        return options

    @classmethod
    def _default_transpile_options(cls) -> Options:
        options = super()._default_transpile_options()
        return options

    @classmethod
    def _default_run_options(cls) -> Options:
        """Default option values for the experiment :meth:`run` method."""
        options = super()._default_run_options()

        options.shots = 2000
        return options

    def __init__(
        self,
        qubit: int,
        delays_t1: List[Union[List[float], np.array]],
        delays_t2: List[Union[List[float], np.array]],
        osc_freq: float = 0.0,
        backend: Optional[Backend] = None,
    ):
        """Initialize the experiment object.

        Args:
            qubit: the qubit under test
            delays_t1: delay times of the T1 experiment
            delays_t2: delay times of the T2* experiment
            osc_freq: the oscillation frequency induced using by the user for T2Ramsey
            backend: Optional, the backend on which to run the experiment

        """
        self.set_experiment_options = self._default_experiment_options()
        # Set experiment options
        self.set_experiment_options.delays_t1 = delays_t1
        self.set_experiment_options.delays_t2 = delays_t2

        self.exps = []
        self.exps.append(
            T1(qubit=qubit, delays=self.set_experiment_options.delays_t1, backend=backend)
        )
        self.exps.append(
            T2Ramsey(
                qubit=qubit,
                delays=self.set_experiment_options.delays_t2,
                backend=backend,
                osc_freq=osc_freq,
            )
        )
        # Run batch experiments
        super().__init__(experiments=self.exps, analysis=TphiAnalysis(), backend=backend)

    def _set_backend(self, backend: Backend):
        super()._set_backend(backend)

        # Scheduling parameters
        if not self._backend.configuration().simulator and not isinstance(backend, FakeBackend):
            timing_constraints = getattr(self.transpile_options, "timing_constraints", {})
            if "acquire_alignment" not in timing_constraints:
                timing_constraints["acquire_alignment"] = 16
            scheduling_method = getattr(self.transpile_options, "scheduling_method", "alap")
            self.set_transpile_options(
                timing_constraints=timing_constraints, scheduling_method=scheduling_method
            )
