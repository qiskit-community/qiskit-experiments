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

from typing import List, Optional, Union, Sequence
import numpy as np

from qiskit import QiskitError
from qiskit.providers import Backend
from qiskit_experiments.framework.composite.batch_experiment import BatchExperiment
from qiskit_experiments.warnings import qubit_deprecate
from qiskit_experiments.library.characterization import (
    T1,
    T2Ramsey,
    T2Hahn,
    TphiAnalysis,
)


class Tphi(BatchExperiment):
    r"""Tphi Experiment Class

    # section: overview

        :math:`T_\varphi`, or :math:`1/\Gamma_\varphi`, is the pure dephasing time of
        depolarization in the :math:`x - y` plane of the Bloch sphere. We compute
        :math:`\Gamma_\varphi` by computing :math:`\Gamma_2`, the transverse relaxation
        rate, and subtracting :math:`\Gamma_1`, the longitudinal relaxation rate. It
        follows that

        :math:`1/T_\varphi = 1/T_2 - 1/2T_1`.

        The transverse relaxation rate can be estimated by either T2 or T2* experiments.

    # section: analysis_ref
        :py:class:`TphiAnalysis`

    # section: reference
        .. ref_arxiv:: 1 1904.06560

    # section: tutorial
        :doc:`/tutorials/tphi_characterization`

    # section: see_also
        qiskit_experiments.library.characterization.t1
        qiskit_experiments.library.characterization.t2ramsey
        qiskit_experiments.library.characterization.t2hahn

    """

    def set_experiment_options(self, **fields):
        """Set the experiment options.
        Args:
            fields: The fields defining the options.

        Raises:
             QiskitError: Invalid input option.
        """
        # propagate options to the sub-experiments.
        for key in fields:
            if key == "delays_t1":
                self.component_experiment(0).set_experiment_options(delays=fields["delays_t1"])
            elif key == "delays_t2":
                self.component_experiment(1).set_experiment_options(delays=fields["delays_t2"])
            elif key == "osc_freq" and isinstance(self.component_experiment(1), T2Ramsey):
                self.component_experiment(1).set_experiment_options(osc_freq=fields["osc_freq"])
            elif key == "num_echoes" and isinstance(self.component_experiment(1), T2Hahn):
                self.component_experiment(1).set_experiment_options(num_echoes=fields["num_echoes"])
            else:
                raise QiskitError(f"Tphi experiment does not support option {key}")

    @qubit_deprecate()
    def __init__(
        self,
        physical_qubits: Sequence[int],
        delays_t1: List[Union[List[float], np.array]],
        delays_t2: List[Union[List[float], np.array]],
        t2star: bool = True,
        osc_freq: float = 0.0,
        num_echoes: int = 1,
        backend: Optional[Backend] = None,
    ):
        """Initialize the experiment object.

        Args:
            physical_qubits: a single-element sequence containing the qubit under test
            t2star: Whether to use T2* for the transverse relaxation time. If True,
                the T2Ramsey is used. If False, the T2Hahn experiment is used. False by default.
            delays_t1: Delay times of the T1 experiment.
            delays_t2: Delay times of the T2 experiment.
            osc_freq: The oscillation frequency induced using by the user for T2Ramsey.
            num_echoes: The number of echoes to perform for T2Hahn.
            backend: Optional, the backend on which to run the experiment
        """

        exp_t1 = T1(physical_qubits=physical_qubits, delays=delays_t1, backend=backend)

        exp_options = {"delays_t1": delays_t1, "delays_t2": delays_t2}

        if t2star:
            exp_t2 = T2Ramsey(
                physical_qubits=physical_qubits,
                delays=delays_t2,
                backend=backend,
                osc_freq=osc_freq,
            )
            exp_options["osc_freq"] = osc_freq
        else:
            exp_t2 = T2Hahn(
                physical_qubits=physical_qubits,
                delays=delays_t2,
                backend=backend,
                num_echoes=num_echoes,
            )
            exp_options["num_echoes"] = num_echoes
        analysis = TphiAnalysis([exp_t1.analysis, exp_t2.analysis])

        # Create batch experiment
        super().__init__([exp_t1, exp_t2], backend=backend, analysis=analysis)
        self.set_experiment_options(**exp_options)
