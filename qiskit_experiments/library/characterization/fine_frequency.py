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

"""Fine frequency characterization experiment."""

from typing import List, Optional, Sequence
import numpy as np

from qiskit import QuantumCircuit
from qiskit.providers.backend import Backend

from qiskit_experiments.framework import BaseExperiment, Options
from qiskit_experiments.curve_analysis.standard_analysis import ErrorAmplificationAnalysis


class FineFrequency(BaseExperiment):
    r"""An experiment to make a fine measurement of the qubit frequency.

    # section: overview

        The fine frequency characterization experiment measures the qubit frequency by moving
        to the equator of the Bloch sphere with a sx gate and idling for a time
        :math:`n\cdot \tau` where :math:`\tau` is the duration of the single-qubit gate and
        :math:`n` is an integer which ranges from zero to a maximal value in integer steps.
        The idle time is followed by a rz rotation with an angle :math:`n\pi/2` and a final
        sx gate. If the frequency of the pulses match the frequency of the qubit then the
        sequence :math:`n\in[0,1,2,3,4,...]` will result in the sequence of measured qubit
        populations :math:`[1, 0.5, 0, 0.5, 1, ...]` due to the rz rotation. If the frequency
        of the pulses do not match the qubit frequency then the qubit will precess in the
        drive frame during the idle time and phase errors will build-up. By fitting the measured
        points we can extract the error in the qubit frequency. The circuit that are run are

        .. parsed-literal::

                    ┌────┐┌────────────────┐┌──────────┐┌────┐ ░ ┌─┐
                 q: ┤ √X ├┤ Delay(n*τ[dt]) ├┤ Rz(nπ/2) ├┤ √X ├─░─┤M├
                    └────┘└────────────────┘└──────────┘└────┘ ░ └╥┘
            meas: 1/══════════════════════════════════════════════╩═
                                                                  0
    # section: analysis_ref
        :class:`~qiskit_experiments.curve_analysis.ErrorAmplificationAnalysis`

    # section: example
        .. jupyter-execute::
            :hide-code:

            # backend
            from qiskit_experiments.test import T2HahnBackend

            # AerSimulator can not mimic a freqeuncy offset
            backend = T2HahnBackend(frequency=1e5)


        .. jupyter-execute::

            from qiskit_experiments.library.characterization import FineFrequency

            exp = FineFrequency([0], delay_duration=int(30e-9 / backend.dt), backend=backend)
            exp_data = exp.run().block_for_results()
            display(exp_data.figure(0))
            exp_data.analysis_results(dataframe=True)
    """

    def __init__(
        self,
        physical_qubits: Sequence[int],
        delay_duration: int,
        backend: Optional[Backend] = None,
        repetitions: Optional[List[int]] = None,
    ):
        """Setup a fine frequency experiment on the given qubit.

        Args:
            physical_qubits: List containing the qubit on which to run the fine
                frequency characterization experiment.
            delay_duration: The duration of the delay at :math:`n=1` in dt.
            backend: Optional, the backend to run the experiment on.
            repetitions: The number of repetitions, if not given then the default value
                from the experiment default options will be used.
        """
        analysis = ErrorAmplificationAnalysis()
        analysis.set_options(
            normalization=True,
            fixed_parameters={
                "angle_per_gate": np.pi / 2,
                "phase_offset": 0.0,
            },
        )

        super().__init__(physical_qubits, analysis=analysis, backend=backend)

        if repetitions is not None:
            self.set_experiment_options(repetitions=repetitions)

        self.set_experiment_options(delay_duration=delay_duration)

    @classmethod
    def _default_experiment_options(cls) -> Options:
        r"""Default values for the fine frequency experiment.

        Experiment Options:
            repetitions (List[int]): A list of the number of times that the delay is repeated.
            delay_duration (int): The duration of the delay as the number of ``dt`` s it contains.
                The total length of the delay in units of ``dt`` will be n times ``delay_duration``
                where n also  determines the rotation angle of the ``RZGate`` by :math:`n \pi/2`.
        """
        options = super()._default_experiment_options()
        options.repetitions = list(range(40))
        options.delay_duration = None

        return options

    @classmethod
    def _default_transpile_options(cls) -> Options:
        """Default transpiler options."""
        options = super()._default_transpile_options()
        options.inst_map = None
        options.basis_gates = ["sx", "rz", "delay"]
        return options

    def _pre_circuit(self) -> QuantumCircuit:
        """A method that subclasses can override to perform gates before the main sequence."""
        return QuantumCircuit(1)

    def circuits(self) -> List[QuantumCircuit]:
        """Return the list of quantum circuits to run."""

        circuits = []

        # The main sequence
        for repetition in self.experiment_options.repetitions:
            circuit = self._pre_circuit()
            circuit.sx(0)

            circuit.delay(duration=self.experiment_options.delay_duration * repetition, unit="dt")

            circuit.rz(np.pi * repetition / 2, 0)
            circuit.sx(0)
            circuit.measure_all()

            circuit.metadata = {"xval": repetition}

            circuits.append(circuit)

        return circuits

    def _metadata(self):
        metadata = super()._metadata()
        # Store measurement level and meas return if they have been
        # set for the experiment
        for run_opt in ["meas_level", "meas_return"]:
            if hasattr(self.run_options, run_opt):
                metadata[run_opt] = getattr(self.run_options, run_opt)
        return metadata
