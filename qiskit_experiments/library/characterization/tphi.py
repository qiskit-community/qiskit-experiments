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

from qiskit import QiskitError
from qiskit.providers import Backend
from qiskit_experiments.framework.composite.batch_experiment import BatchExperiment
from qiskit_experiments.library.characterization import (
    T1,
    T2Ramsey,
    TphiAnalysis,
)


class Tphi(BatchExperiment):
    r"""Tphi Experiment Class

    # section: overview

        :math:`\Gamma_\varphi` is defined as the rate of pure dephasing
        or depolarization in the :math:`x - y` plane.
        We compute :math:`\Gamma_\varphi` by computing :math:`\Gamma_2*`, the transverse relaxation rate,
        and subtracting :math:`\Gamma_1`, the longitudinal relaxation rate. The pure dephasing time
        is defined by :math:`T_\varphi = 1/\Gamma_\varphi`. Or more precisely,

        :math:`1/T_\varphi = 1/T_{2*} - 1/2T_1`.

        For more details, see :py:class:`T1` and :py:class:`T2Ramsey`

    # section: analysis_ref
        :py:class:`TphiAnalysis`

    # section: reference
        .. ref_arxiv:: 1 1904.06560v5

    # section: tutorial
        :doc:`/tutorials/tphi_characterization`
    """

    def set_experiment_options(self, **fields):
        """Set the experiment options.
        Args:
            fields: The fields defining the options

        Raises:
             QiskitError : Error for invalid input option.
        """
        # propagate options to the sub-experiments.
        for key in fields:
            if key == "delays_t1":
                self.component_experiment(0).set_experiment_options(delays=fields["delays_t1"])
            elif key == "delays_t2":
                self.component_experiment(1).set_experiment_options(delays=fields["delays_t2"])
            elif key == "osc_freq":
                self.component_experiment(1).set_experiment_options(osc_freq=fields["osc_freq"])
            else:
                raise QiskitError(f"Tphi experiment does not support option {key}")

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

        exp_t1 = T1(qubit=qubit, delays=delays_t1, backend=backend)
        exp_t2 = T2Ramsey(
            qubit=qubit,
            delays=delays_t2,
            backend=backend,
            osc_freq=osc_freq,
        )
        analysis = TphiAnalysis([exp_t1.analysis, exp_t2.analysis])

        # Create batch experiment
        super().__init__([exp_t1, exp_t2], backend=backend, analysis=analysis)
        self.set_experiment_options(delays_t1=delays_t1, delays_t2=delays_t2)
