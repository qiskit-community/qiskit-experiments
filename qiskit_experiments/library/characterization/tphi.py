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
from qiskit_experiments.library.characterization import (
    T1,
    T2Ramsey,
    T2Hahn,
    TphiAnalysis,
)


class Tphi(BatchExperiment):
    r"""An experiment to measure the qubit dephasing rate in the :math:`x - y` plane.

    # section: overview

    :math:`T_\varphi`, or :math:`1/\Gamma_\varphi`, is the pure dephasing time in
    the :math:`x - y` plane of the Bloch sphere. We compute :math:`\Gamma_\varphi`
    by computing :math:`\Gamma_2`, the transverse relaxation rate, and subtracting
    :math:`\Gamma_1`, the longitudinal relaxation rate. It follows that

    :math:`1/T_\varphi = 1/T_2 - 1/2T_1`.

    The transverse relaxation rate can be estimated by either :math:`T_2` or
    :math:`T_2^*` experiments. In superconducting qubits, :math:`T_2^*` tends to be
    significantly smaller than :math:`T_1`, so :math:`T_2` is usually used.

    .. note::
        In 0.5.0, this experiment changed from using :math:`T_2^*` as the default
        to :math:`T_2`.

    # section: analysis_ref
        :class:`.TPhiAnalysis`

    # section: example
        .. jupyter-execute::
            :hide-code:

            # backend
            from qiskit_ibm_runtime.fake_provider import FakeManilaV2
            from qiskit_aer import AerSimulator
            from qiskit_aer.noise import NoiseModel

            noise_model = NoiseModel.from_backend(FakeManilaV2(),
                                                  thermal_relaxation=True,
                                                  gate_error=False,
                                                  readout_error=False,
                                                 )

            backend = AerSimulator.from_backend(FakeManilaV2(), noise_model=noise_model)

        .. jupyter-execute::

            import numpy as np
            import qiskit
            from qiskit_experiments.library.characterization import Tphi

            delays_t1 = np.arange(1e-6, 300e-6, 10e-6)
            delays_t2 = np.arange(1e-6, 50e-6, 2e-6)

            exp = Tphi(physical_qubits=(0, ),
                        delays_t1=delays_t1,
                        delays_t2=delays_t2,
                        backend=backend
                        )

            exp_data = exp.run().block_for_results()
            display(exp_data.figure(0))
            display(exp_data.figure(1))
            exp_data.analysis_results(dataframe=True)

    # section: reference
        .. ref_arxiv:: 1 1904.06560

    # section: manual
        :doc:`/manuals/characterization/tphi`

    # section: see_also
        * :py:class:`qiskit_experiments.library.characterization.T1`
        * :py:class:`qiskit_experiments.library.characterization.T2Ramsey`
        * :py:class:`qiskit_experiments.library.characterization.T2Hahn`

    """

    def __init__(
        self,
        physical_qubits: Sequence[int],
        delays_t1: List[Union[List[float], np.array]],
        delays_t2: List[Union[List[float], np.array]],
        t2type: str = "hahn",
        osc_freq: float = 0.0,
        num_echoes: int = 1,
        backend: Optional[Backend] = None,
    ):
        """Initialize the experiment object.

        Args:
            physical_qubits: A single-element sequence containing the qubit under test.
            t2type: What type of T2/T2* experiment to use. Can be either "ramsey" for
                :class:`.T2Ramsey` to be used, or "hahn" for :class:`.T2Hahn`. Defaults
                to "hahn".
            delays_t1: Delay times of the T1 experiment.
            delays_t2: Delay times of the T2 experiment.
            osc_freq: The oscillation frequency induced for T2Ramsey. Only used when
                ``t2type`` is set to "ramsey".
            num_echoes: The number of echoes to perform for T2Hahn. Only used when
                ``t2type`` is set to "hahn".
            backend: Optional, the backend on which to run the experiment.

        Raises:
            QiskitError: If an invalid ``t2type`` is provided.
        """

        exp_t1 = T1(physical_qubits=physical_qubits, delays=delays_t1, backend=backend)

        exp_options = {"delays_t1": delays_t1, "delays_t2": delays_t2}

        if t2type == "ramsey":
            exp_t2 = T2Ramsey(
                physical_qubits=physical_qubits,
                delays=delays_t2,
                backend=backend,
                osc_freq=osc_freq,
            )
            exp_options["osc_freq"] = osc_freq
        elif t2type == "hahn":
            exp_t2 = T2Hahn(
                physical_qubits=physical_qubits,
                delays=delays_t2,
                backend=backend,
                num_echoes=num_echoes,
            )
            exp_options["num_echoes"] = num_echoes
        else:
            raise QiskitError(f"Invalid T2 experiment type {t2type} specified.")

        analysis = TphiAnalysis([exp_t1.analysis, exp_t2.analysis])

        # Create batch experiment
        super().__init__(
            [exp_t1, exp_t2],
            flatten_results=True,
            backend=backend,
            analysis=analysis,
        )
        self.set_experiment_options(**exp_options)

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
                raise QiskitError(f"Tphi experiment does not support option {key}.")
