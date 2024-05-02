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
T2Ramsey Experiment class.

"""

from typing import List, Union, Optional, Sequence
import numpy as np

import qiskit
from qiskit import QuantumCircuit
from qiskit.providers.backend import Backend

from qiskit_experiments.framework import BackendTiming, BaseExperiment, Options
from qiskit_experiments.library.characterization.analysis.t2ramsey_analysis import T2RamseyAnalysis


class T2Ramsey(BaseExperiment):
    r"""An experiment to measure the Ramsey frequency and the qubit dephasing time
    sensitive to inhomogeneous broadening.

    # section: overview

        This experiment is used to estimate two properties for a single qubit:
        :math:`T_2^*` and Ramsey frequency. :math:`T_2^*` is the dephasing time
        or the transverse relaxation time of the qubit on the Bloch sphere as a result
        of both energy relaxation and pure dephasing in the transverse plane. Unlike
        :math:`T_2`, which is measured by :class:`.T2Hahn`, :math:`T_2^*` is sensitive
        to inhomogenous broadening.

        This experiment consists of a series of circuits of the form

        .. parsed-literal::

                 ┌───┐┌──────────────┐┌──────┐ ░ ┌───┐ ░ ┌─┐
            q_0: ┤ H ├┤   DELAY(t)   ├┤ P(λ) ├─░─┤ H ├─░─┤M├
                 └───┘└──────────────┘└──────┘ ░ └───┘ ░ └╥┘
            c: 1/═════════════════════════════════════════╩═
                                                        0

        for each *t* from the specified delay times, where
        :math:`\lambda =2 \pi \times {osc\_freq}`,
        and the delays are specified by the user.
        The circuits are run on the device or on a simulator backend.

    # section: manual
        :doc:`/manuals/characterization/t2ramsey`

    # section: analysis_ref
        :class:`T2RamseyAnalysis`

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
            from qiskit_experiments.library import T2Ramsey

            delays = list(np.arange(1.00e-6, 50.0e-6, 2.00e-6))
            exp = T2Ramsey(physical_qubits=(0, ), delays=delays, backend=backend, osc_freq=1.0e5)

            exp_data = exp.run().block_for_results()
            display(exp_data.figure(0))
            exp_data.analysis_results(dataframe=True)

    # section: reference
        .. ref_arxiv:: 1 1904.06560
    """

    @classmethod
    def _default_experiment_options(cls) -> Options:
        """Default experiment options.

        Experiment Options:
            delays (Iterable[float]): Delay times of the experiments in seconds.
            osc_freq (float): Oscillation frequency offset in Hz.
        """
        options = super()._default_experiment_options()

        options.delays = None
        options.osc_freq = 0.0

        return options

    def __init__(
        self,
        physical_qubits: Sequence[int],
        delays: Union[List[float], np.array],
        backend: Optional[Backend] = None,
        osc_freq: float = 0.0,
    ):
        """
        Initialize the T2Ramsey class.

        Args:
            physical_qubits: a single-element sequence containing the qubit under test.
            delays: delay times of the experiments in seconds.
            backend: Optional, the backend to run the experiment on.
            osc_freq: the oscillation frequency induced by the user.
                The frequency is given in Hz.

        """
        super().__init__(physical_qubits, analysis=T2RamseyAnalysis(), backend=backend)
        self.set_experiment_options(delays=delays, osc_freq=osc_freq)

    def circuits(self) -> List[QuantumCircuit]:
        """Return a list of experiment circuits.

        Each circuit consists of a Hadamard gate, followed by a fixed delay,
        a phase gate (with a linear phase), and an additional Hadamard gate.

        Returns:
            The experiment circuits
        """
        timing = BackendTiming(self.backend)

        circuits = []
        for delay in self.experiment_options.delays:
            rotation_angle = (
                2 * np.pi * self.experiment_options.osc_freq * timing.delay_time(time=delay)
            )

            circ = qiskit.QuantumCircuit(1, 1)
            circ.sx(0)  # Brings the qubit to the X Axis
            circ.delay(timing.round_delay(time=delay), 0, timing.delay_unit)
            circ.rz(rotation_angle, 0)
            circ.barrier(0)
            circ.sx(0)
            circ.barrier(0)
            circ.measure(0, 0)

            circ.metadata = {"xval": timing.delay_time(time=delay)}

            circuits.append(circ)

        return circuits

    def _metadata(self):
        metadata = super()._metadata()
        # Store measurement level and meas return if they have been
        # set for the experiment
        for run_opt in ["meas_level", "meas_return"]:
            if hasattr(self.run_options, run_opt):
                metadata[run_opt] = getattr(self.run_options, run_opt)
        metadata["osc_freq"] = self.experiment_options.osc_freq
        return metadata
