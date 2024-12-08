# This code is part of Qiskit.
#
# (C) Copyright IBM 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Multi state discrimination experiment."""

import warnings
from typing import Dict, List, Optional, Sequence

from qiskit import QuantumCircuit
from qiskit.circuit import Gate
from qiskit.providers import Backend
from qiskit.providers.options import Options
from qiskit.pulse import ScheduleBlock
from qiskit.qobj.utils import MeasLevel, MeasReturnType
from qiskit_experiments.framework import BaseExperiment
from qiskit_experiments.library.characterization import MultiStateDiscriminationAnalysis


class MultiStateDiscrimination(BaseExperiment):
    r"""An experiment that discriminates between the first :math:`n` energy states.

    # section: overview

        The experiment creates :math:`n` circuits that prepare, respectively, the energy states
        :math:`|0\rangle,\cdots,|n-1\rangle`. For, e.g., :math:`n=4` the circuits are of the form

        .. parsed-literal::

            Circuit preparing :math:`|0\rangle`

                       ░ ┌─┐
                   q: ─░─┤M├
                       ░ └╥┘
                meas: ════╩═

            ...

            Circuit preparing :math:`|3\rangle`

                      ┌───┐┌─────┐┌─────┐ ░ ┌─┐
                   q: ┤ X ├┤ x12 ├┤ x23 ├─░─┤M├
                      └───┘└─────┘└─────┘ ░ └╥┘
                meas: ═══════════════════════╩═

    # section: analysis_ref
        :class:`MultiStateDiscriminationAnalysis`

    # section: example
        .. jupyter-execute::
            :hide-code:

            # backend
            from qiskit_experiments.test.pulse_backend import SingleTransmonTestBackend
            backend = SingleTransmonTestBackend(5.2e9,-.25e9, 1e9, 0.8e9, 1e4, noise=False, seed=199)

        .. jupyter-execute::

            from qiskit_experiments.library.characterization import MultiStateDiscrimination

            exp=MultiStateDiscrimination((0,), backend=backend)

            exp_data=exp.run().block_for_results()
            display(exp_data.figure(0))
            exp_data.analysis_results(dataframe=True)

    # section: reference
        `Qiskit Textbook\
        <https://github.com/Qiskit/textbook/blob/main/notebooks/quantum-hardware-pulses/accessing_higher_energy_states.ipynb>`_

    """

    @classmethod
    def _default_run_options(cls) -> Options:
        """Default option values for the experiment :meth:`run` method."""
        options = super()._default_run_options()

        options.meas_level = MeasLevel.KERNELED
        options.meas_return = MeasReturnType.SINGLE

        return options

    @classmethod
    def _default_experiment_options(cls) -> Options:
        """Default values for the number of states if none is given.

        Experiment Options:
            n_states (int): The number of states to discriminate.
            schedules (dict): A dictionary of the schedules for the gates in the experiment. Each key is
                a gate name of the form ``xii+1`` which should implement an x-rotation between level
                ``i`` and ``i+1``.

        """
        options = super()._default_experiment_options()
        options.n_states = 2
        options.schedules = None

        return options

    def __init__(
        self,
        physical_qubits: Sequence[int],
        backend: Optional[Backend] = None,
        n_states: Optional[int] = None,
        schedules: Optional[Dict[str, ScheduleBlock]] = None,
    ):
        """Setup an experiment to prepare different energy states on a given qubit.

        Args:
            physical_qubits: A single-element sequence containing the qubit on which to run the
                experiment.
            backend: Optional, the backend to run the experiment on.
            n_states: The number of energy levels to prepare.
            schedules: The schedules of the x gates between neighboring energy levels.
        """

        super().__init__(
            physical_qubits, analysis=MultiStateDiscriminationAnalysis(), backend=backend
        )

        self.experiment_options.schedules = schedules

        if n_states is not None:
            self.set_experiment_options(n_states=n_states)

    def circuits(self) -> List[QuantumCircuit]:
        """
        Create the circuits for the multi state discrimination experiment.

        Returns:
            A list of circuits preparing the different energy states.
        """
        warnings.warn(
            (
                "Setting pulse schedules for x gates is deprecated  as of "
                "version 0.8 due to the deprecation of Qiskit Pulse. It will be "
                "removed in a future release."
            ),
            DeprecationWarning,
        )
        circuits = []
        for level in range(self.experiment_options.n_states):
            circuit = QuantumCircuit(1)

            # Prepare |1>
            if level >= 1:
                circuit.x(0)

            # Prepare higher energy states
            if level >= 2:
                for idx in range(1, level):
                    gate_name = f"x{idx}{idx + 1}"
                    gate = Gate(name=gate_name, num_qubits=1, params=[])
                    circuit.append(gate, (0,))
                    if self.experiment_options.schedules is not None:
                        circuit.add_calibration(
                            gate_name,
                            self._physical_qubits,
                            self.experiment_options.schedules[gate_name],
                        )

            # label the circuit
            circuit.metadata = {"label": level}

            circuit.measure_all()
            circuits.append(circuit)

        return circuits
