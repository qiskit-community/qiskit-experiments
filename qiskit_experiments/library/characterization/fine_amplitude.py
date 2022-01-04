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

"""Fine amplitude characterization experiment."""

from typing import List, Optional
import numpy as np

from qiskit import QuantumCircuit
from qiskit.circuit import Gate
from qiskit.circuit.library import XGate, SXGate
from qiskit.providers.backend import Backend
from qiskit_experiments.framework import BaseExperiment, Options
from qiskit_experiments.library.characterization.analysis import FineAmplitudeAnalysis
from qiskit_experiments.exceptions import CalibrationError


class FineAmplitude(BaseExperiment):
    r"""Error amplifying fine amplitude calibration experiment.

    # section: overview

        The :class:`FineAmplitude` calibration experiment repeats N times a gate with a pulse
        to amplify the under-/over-rotations in the gate to determine the optimal amplitude.
        The circuits are therefore of the form:

        .. parsed-literal::

                       ┌──────┐       ┌──────┐ ░ ┌─┐
                  q_0: ┤ Gate ├─ ... ─┤ Gate ├─░─┤M├
                       └──────┘       └──────┘ ░ └╥┘
            measure: 1/═════════ ... ═════════════╩═
                                                  0

        Here, Gate is the name of the gate which will be repeated. The user can optionally add a
        square-root of X pulse before the gates are repeated. This square-root of X pulse allows
        the analysis to differentiate between over rotations and under rotations in the case of
        pi-pulses. Importantly, the resulting data is analyzed by a fit to a cosine function in
        which we try to determine the over/under rotation given an intended rotation angle per
        gate which must also be specified by the user.

        Error amplifying experiments are most sensitive to angle errors when we measure points along
        the equator of the Bloch sphere. This is why users should insert a square-root of X pulse
        before running calibrations for :math:`\pm\pi` rotations. When all data points are close to
        the equator, it is difficult for a fitter to infer the overall scale of the error. When
        calibrating a :math:`pi` rotation, one can use ``add_xp_circuit = True`` to insert one
        circuit that puts the qubit in the excited state to set the scale for the other circuits.
        Furthermore, when running calibrations for :math:`\pm\pi/2` rotations users are advised
        to use an odd number of repetitions, e.g. [1, 2, 3, 5, 7, ...] to ensure that the ideal
        points are on the equator of the Bloch sphere. Note the presence of two repetitions which
        allows us to prepare the excited state. Therefore, ``add_xp_circuit = True`` is not needed
        in this case.

    # section: example

        The steps to run a fine amplitude experiment are

        .. code-block:: python

            qubit = 3
            amp_cal = FineAmplitude(qubit, SXGate())
            amp_cal.set_experiment_options(
                angle_per_gate=np.pi/2,
                add_xp_circuit=False,
                add_sx=False
            )
            amp_cal.run(backend)

        Note that there are subclasses of :class:`FineAmplitude` such as :class:`FineSXAmplitude`
        that set the appropriate options by default.

    # section: analysis_ref
        :py:class:`FineAmplitudeAnalysis`

    # section: reference
        .. ref_arxiv:: 1 1504.06597

    # section: tutorial
        :doc:`/tutorials/fine_calibrations`

    """

    @classmethod
    def _default_experiment_options(cls) -> Options:
        r"""Default values for the fine amplitude experiment.

        Experiment Options:
            repetitions (List[int]): A list of the number of times that the gate is repeated.
            gate_type (Gate): This is a gate class such as XGate, so that one can obtain a gate
                by doing :code:`options.gate_class()`.
            normalization (bool): If set to True the DataProcessor will normalized the
                measured signal to the interval [0, 1]. Defaults to True.
            add_sx (bool): If True then the circuits will start with an sx gate. This is typically
                needed when calibrating pulses with a target rotation angle of :math:`\pi`. The
                default value is False.
            add_xp_circuit (bool): If set to True then a circuit with only an X gate will also be
                run. This allows the analysis class to determine the correct sign for the amplitude.
        """
        options = super()._default_experiment_options()
        options.repetitions = list(range(15))
        options.gate = None
        options.normalization = True
        options.add_sx = False
        options.add_xp_circuit = True

        return options

    def __init__(self, qubit: int, gate: Gate, backend: Optional[Backend] = None):
        """Setup a fine amplitude experiment on the given qubit.

        Args:
            qubit: The qubit on which to run the fine amplitude calibration experiment.
            gate: The gate that will be repeated.
            backend: Optional, the backend to run the experiment on.
        """
        super().__init__([qubit], analysis=FineAmplitudeAnalysis(), backend=backend)
        self.set_experiment_options(gate=gate)

    def _pre_circuit(self) -> QuantumCircuit:
        """Return a preparation circuit.

        This method can be overridden by subclasses e.g. to calibrate gates on
        transitions other than the 0 <-> 1 transition.
        """
        circuit = QuantumCircuit(1)

        if self.experiment_options.add_sx:
            circuit.sx(0)

        return circuit

    def circuits(self) -> List[QuantumCircuit]:
        """Create the circuits for the fine amplitude calibration experiment.

        Returns:
            A list of circuits with a variable number of gates.

        Raises:
            CalibrationError: If the analysis options do not contain the angle_per_gate.
        """
        # Prepare the circuits.
        repetitions = self.experiment_options.get("repetitions")

        circuits = []

        if self.experiment_options.add_xp_circuit:
            # Note that the rotation error in this xval will be overweighted when calibrating xp
            # because it will be treated as a half pulse instead of a full pulse. However, since
            # the qubit population is first-order insensitive to rotation errors for an xp pulse
            # this point won't contribute much to inferring the angle error.
            angle_per_gate = self.analysis.options.get("angle_per_gate", None)
            phase_offset = self.analysis.options.get("phase_offset")

            if angle_per_gate is None:
                raise CalibrationError(
                    f"Unknown angle_per_gate for {self.__class__.__name__}. "
                    "Please set it in the analysis options."
                )

            circuit = QuantumCircuit(1)
            circuit.x(0)
            circuit.measure_all()

            circuit.metadata = {
                "experiment_type": self._type,
                "qubits": self.physical_qubits,
                "xval": (np.pi - phase_offset) / angle_per_gate,
                "unit": "gate number",
            }

            circuits.append(circuit)

        for repetition in repetitions:
            circuit = self._pre_circuit()

            for _ in range(repetition):
                circuit.append(self.experiment_options.gate, (0,))

            circuit.measure_all()

            circuit.metadata = {
                "experiment_type": self._type,
                "qubits": self.physical_qubits,
                "xval": repetition,
                "unit": "gate number",
            }

            circuits.append(circuit)

        return circuits


class FineXAmplitude(FineAmplitude):
    r"""A fine amplitude experiment with all the options set for the :math:`\pi`-rotation.

    # section: overview

        :class:`FineXAmplitude` is a subclass of :class:`FineAmplitude` and is used to set
        the appropriate values for the default options.
    """

    def __init__(self, qubit: int, backend: Optional[Backend] = None):
        """Initialize the experiment."""
        super().__init__(qubit, XGate(), backend=backend)
        # Set default analysis options
        self.analysis.set_options(
            angle_per_gate=np.pi,
            phase_offset=np.pi / 2,
            amp=1,
        )

    @classmethod
    def _default_experiment_options(cls) -> Options:
        r"""Default values for the fine amplitude experiment.

        Experiment Options:
            gate (Gate): Gate to characterize. Defaults to an XGate.
            add_sx (bool): This option is True by default when calibrating gates with a target
                angle per gate of :math:`\pi` as this increases the sensitivity of the
                experiment.
            add_xp_circuit (bool): This option is True by default when calibrating gates with
                a target angle per gate of :math:`\pi`.
        """
        options = super()._default_experiment_options()
        options.gate = XGate()
        options.add_sx = True
        options.add_xp_circuit = True
        return options


class FineSXAmplitude(FineAmplitude):
    r"""A fine amplitude experiment with all the options set for the :math:`\pi/2`-rotation.

    # section: overview

        :class:`FineSXAmplitude` is a subclass of :class:`FineAmplitude` and is used to set
        the appropriate values for the default options.
    """

    def __init__(self, qubit: int, backend: Optional[Backend] = None):
        """Initialize the experiment."""
        super().__init__(qubit, SXGate(), backend=backend)
        # Set default analysis options
        self.analysis.set_options(
            angle_per_gate=np.pi / 2,
            phase_offset=np.pi,
        )

    @classmethod
    def _default_experiment_options(cls) -> Options:
        r"""Default values for the fine amplitude experiment.

        Experiment Options:
            gate (Gate): FineSXAmplitude calibrates an SXGate.
            add_sx (bool): This option is False by default when calibrating gates with a target
                angle per gate of :math:`\pi/2` as it is not necessary in this case.
            add_xp_circuit (bool): This option is False by default when calibrating gates with
                a target angle per gate of :math:`\pi/2`.
            repetitions (List[int]): By default the repetitions take on odd numbers for
                :math:`\pi/2` target angles as this ideally prepares states on the equator of
                the Bloch sphere. Note that the repetitions include two repetitions which
                plays the same role as including a circuit with an X gate.
        """
        options = super()._default_experiment_options()
        options.gate = SXGate()
        options.add_sx = False
        options.add_xp_circuit = False
        options.repetitions = [0, 1, 2, 3, 5, 7, 9, 11, 13, 15, 17, 21, 23, 25]

        return options
