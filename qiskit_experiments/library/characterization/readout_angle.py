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
Readout Angle Experiment class.
"""

from typing import List, Optional, Sequence

from qiskit.circuit import QuantumCircuit
from qiskit.qobj.utils import MeasLevel
from qiskit.providers.backend import Backend

from qiskit_experiments.framework import BaseExperiment, Options
from qiskit_experiments.library.characterization.analysis.readout_angle_analysis import (
    ReadoutAngleAnalysis,
)


class ReadoutAngle(BaseExperiment):
    r"""
    An experiment to measure the angle between ground and excited state IQ clusters.

    # section: overview

        Design and analyze experiments for estimating readout angle of the qubit.
        The readout angle is the average of two angles: the angle of the IQ
        cluster center of the ground state, and the angle of the IQ cluster center
        of the excited state.

        Each experiment consists of the following steps:

        1. Circuits generation: two circuits, the first circuit measures the qubit
           in the ground state, the second circuit sets the qubit in the excited state
           and measures it. Measurements are in level 1 (kerneled).
        2. Backend execution: actually running the circuits on the device
           (or a simulator that supports level 1 measurements). The backend returns
           the cluster centers of the ground and excited states.
        3. Analysis of results: return the average of the angles of the two centers.

    # section: analysis_ref
        :class:`ReadoutAngleAnalysis`

    # section: example
        .. jupyter-execute::
            :hide-code:

            # backend
            from qiskit_experiments.test.pulse_backend import SingleTransmonTestBackend
            backend = SingleTransmonTestBackend(5.2e9,-.25e9, 1e9, 0.8e9, 1e4, noise=False, seed=199)

        .. jupyter-execute::

            from qiskit_experiments.library.characterization import ReadoutAngle

            exp = ReadoutAngle((0,), backend=backend)

            exp_data = exp.run().block_for_results()
            display(exp_data.figure(0))
            exp_data.analysis_results(dataframe=True)
    """

    @classmethod
    def _default_run_options(cls) -> Options:
        """Default run options."""
        options = super()._default_run_options()

        options.meas_level = MeasLevel.KERNELED
        options.meas_return = "avg"

        return options

    def __init__(
        self,
        physical_qubits: Sequence[int],
        backend: Optional[Backend] = None,
    ):
        """
        Initialize the readout angle experiment class

        Args:
            physical_qubits: a single-element sequence containing the qubit whose readout angle is to be
                estimated
            backend: Optional, the backend to run the experiment on.
        """
        # Initialize base experiment
        super().__init__(physical_qubits, analysis=ReadoutAngleAnalysis(), backend=backend)

    def circuits(self) -> List[QuantumCircuit]:
        """
        Return a list of experiment circuits

        Returns:
            The experiment circuits
        """
        circ0 = QuantumCircuit(1, 1)
        circ0.measure(0, 0)
        circ0.metadata = {"xval": 0}

        circ1 = QuantumCircuit(1, 1)
        circ1.x(0)
        circ1.measure(0, 0)
        circ1.metadata = {"xval": 1}

        return [circ0, circ1]
