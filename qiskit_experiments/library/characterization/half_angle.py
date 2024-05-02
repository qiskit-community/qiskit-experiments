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

"""Half angle characterization."""

from typing import List, Optional, Sequence
import numpy as np

from qiskit import QuantumCircuit
from qiskit.providers import Backend

from qiskit_experiments.framework import BaseExperiment, Options
from qiskit_experiments.curve_analysis.standard_analysis import ErrorAmplificationAnalysis
from qiskit_experiments.curve_analysis import ParameterRepr


class HalfAngle(BaseExperiment):
    r"""An experiment class to measure the amount by which sx and x are not parallel.

    # section: overview

        This experiment runs circuits that repeat blocks of :code:`sx - sx - y` gates
        inserted in a Ramsey type experiment, i.e. the full gate sequence is thus
        :code:`Ry(π/2) - [sx - sx - y] ^ n - sx` where :code:`n` is varied.

        .. parsed-literal::

                    ┌─────────┐┌────┐┌────┐┌───┐   ┌────┐┌────┐┌───┐┌────┐ ░ ┌─┐
               q_0: ┤ Ry(π/2) ├┤ sx ├┤ sx ├┤ y ├...┤ sx ├┤ sx ├┤ y ├┤ sx ├─░─┤M├
                    └─────────┘└────┘└────┘└───┘   └────┘└────┘└───┘└────┘ ░ └╥┘
            meas: 1/════════════════════════════...═══════════════════════════╩═
                                                                              0

        This sequence measures angle errors where the axis of the :code:`sx` and :code:`x`
        rotation are not parallel. A similar experiment is described in Ref.~[1] where the
        gate sequence :code:`x - y` is repeated to amplify errors caused by non-orthogonal
        :code:`x` and :code:`y` rotation axes.

        One cause of such errors is non-linearity in the microwave mixer used
        to produce the pulses for the ``x`` and ``sx`` gates. Typically, these
        gates are calibrated to have the same duration and so have different
        pulse amplitudes. Non-linearities in the mixer's skew can cause the
        angle to differ for these different pulse amplitudes.

        The way the experiment works is that the initial ``Ry(π/2)`` puts the
        qubit close to the :math:`+X` state, with a deviation :math:`δθ`, due
        to the misalignment between ``sx`` and ``x`` (``Ry(π/2)`` is
        implemented with ``sx`` as described below). The first ``sx - sx`` do
        nothing as they should be rotations about the axis the qubit is
        pointing along. The first ``y`` then mirrors the qubit about the
        :math:`y` axis in the :math:`xy` plane of the Bloch sphere, so the
        :math:`δθ` deviation from :math:`+X` becomes a :math:`-δθ` from
        :math:`-X`. The next ``sx - sx`` sequence rotates about the axis that
        is :math:`+δθ` rotated in the :math:`xy` plane from :math:`+X`, which
        takes the deviation from :math:`-X` from :math:`-δθ` to :math:`+3 δθ`.
        Then the next ``y`` mirrors this across the :math:`y` axis, taking the
        state to :math:`-3 δθ` from :math:`+X`. This pattern continues with
        each iteration, with the angular deviation in units of :math:`δθ`
        following the sequence 1, 3, 5, 7, 9, etc. from :math:`+X` and
        :math:`-X`. The final ``sx`` rotation serves mainly to rotate these
        deviations from :math:`+X` and :math:`-X` in the :math:`xy` plane into
        deviations out of the :math:`xy` plane, so that they appear as a signal
        in the :math:`Z` basis.  Because ``sx`` has a :math:`δθ` deviation from
        ``x``, the final ``sx`` adds an extra :math:`δθ` to the deviations, so
        the pattern ends up as 2, 4, 6, 8, etc., meaning that each iteration
        adds :math:`2 δθ` to the deviation from the equator of the Bloch sphere
        (with the sign alternating due to the ``y`` gates, so the deviations
        are really -2, 4, -6, 8, etc.).

        For the implementation of the circuits, the experiment uses ``Rz(π/2) -
        sx - Rz(-π/2)`` to implement the ``Ry(π/2)`` and ``Rz(π/2) - x -
        Rz(-π/2)`` to implement the ``y``. So the experiment makes use of only
        ``sx``, ``x``, ``Rz(π/2)``, and ``Rz(-π/2)`` gates. For the
        experiment's analysis to be valid, it is important that the ``sx`` and
        ``x`` gates are not replaced (such as by a transpiler pass that
        replaces ``x`` with ``sx - sx``), as it is the angle between them which
        is being inferred. It is assumed that the angle between ``x`` and
        ``Rz`` is exactly :math:`π/2`.

    # section: analysis_ref
        :class:`.ErrorAmplificationAnalysis`

    # section: example
        .. jupyter-execute::
            :hide-code:

            # backend
            from qiskit_experiments.test.pulse_backend import SingleTransmonTestBackend
            backend = SingleTransmonTestBackend(5.2e9,-.25e9, 1e9, 0.8e9, 1e4, noise=False, seed=199)

        .. jupyter-execute::

            from qiskit_experiments.library.characterization import HalfAngle

            exp = HalfAngle((0,), backend=backend)

            exp_data = exp.run().block_for_results()
            display(exp_data.figure(0))
            exp_data.analysis_results(dataframe=True)

    # section: reference
        .. ref_arxiv:: 1 1504.06597
    """

    @classmethod
    def _default_experiment_options(cls) -> Options:
        r"""Default values for the half angle experiment.

        Experiment Options:
            repetitions (List[int]): A list of the number of times that the gate
                sequence :code:`[sx sx y]` is repeated.
        """
        options = super()._default_experiment_options()
        options.repetitions = list(range(15))
        return options

    def __init__(self, physical_qubits: Sequence[int], backend: Optional[Backend] = None):
        """Setup a half angle experiment on the given qubit.

        Args:
            physical_qubits: List containing the qubits on which to run the
                fine amplitude calibration experiment.
            backend: Optional, the backend to run the experiment on.
        """
        analysis = ErrorAmplificationAnalysis()

        default_bounds = analysis.options.bounds
        default_bounds.update({"d_theta": (-np.pi / 2, np.pi / 2)})

        analysis.set_options(
            fixed_parameters={
                "angle_per_gate": np.pi,
                "phase_offset": -np.pi / 2,
                "amp": 1.0,
            },
            result_parameters=[ParameterRepr("d_theta", "d_hac", "rad")],
            normalization=True,
            bounds=default_bounds,
        )

        super().__init__(physical_qubits, analysis=analysis, backend=backend)

    @staticmethod
    def _pre_circuit() -> QuantumCircuit:
        """Return the preparation circuit for the experiment."""
        return QuantumCircuit(1)

    def circuits(self) -> List[QuantumCircuit]:
        """Create the circuits for the half angle calibration experiment."""

        circuits = []

        for repetition in self.experiment_options.repetitions:
            circuit = self._pre_circuit()

            # First ry gate
            circuit.rz(np.pi / 2, 0)
            circuit.sx(0)
            circuit.rz(-np.pi / 2, 0)

            # Error amplifying sequence
            for _ in range(repetition):
                circuit.sx(0)
                circuit.sx(0)
                circuit.rz(np.pi / 2, 0)
                circuit.x(0)
                circuit.rz(-np.pi / 2, 0)

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
