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

from typing import List, Optional, Union, Iterable, Dict, Tuple

import numpy as np
import qiskit
from qiskit.circuit import QuantumCircuit, Parameter, ParameterExpression
from qiskit.providers import Backend, Options
from qiskit.pulse.schedule import ScheduleBlock
from qiskit.utils import apply_prefix

from qiskit_experiments.base_experiment import BaseExperiment
from .t2ramsey_analysis import T2RamseyAnalysis, RamseyXYAnalysis


class T2Ramsey(BaseExperiment):
    """T2Ramsey class"""

    __analysis_class__ = T2RamseyAnalysis

    def __init__(
        self,
        qubit: int,
        delays: Union[List[float], np.array],
        unit: str = "s",
        osc_freq: float = 0.0,
        experiment_type: Optional[str] = None,
    ):
        """Initialize the T2Ramsey class.

        Args:
            qubit: the qubit under test
            delays: delay times of the experiments
            unit: Optional, time unit of `delays`.
            Supported units: 's', 'ms', 'us', 'ns', 'ps', 'dt'.
            The unit is used for both T2Ramsey and the frequency
            osc_freq: the oscillation frequency induced using by the user
            experiment_type: String indicating the experiment type.
        """

        self._qubit = qubit
        self._delays = delays
        self._unit = unit
        self._osc_freq = osc_freq
        super().__init__([qubit], experiment_type)

    def circuits(self, backend: Optional[Backend] = None) -> List[QuantumCircuit]:
        """Return a list of experiment circuits.

        Each circuit consists of a Hadamard gate, followed by a fixed delay,
        a phase gate (with a linear phase), and an additional Hadamard gate.

        Args:
            backend: Optional, a backend object

        Returns:
            The experiment circuits

        Raises:
            AttributeError: if unit is dt but dt parameter is missing in the backend configuration
        """
        if self._unit == "dt":
            try:
                dt_factor = getattr(backend._configuration, "dt")
            except AttributeError as no_dt:
                raise AttributeError("Dt parameter is missing in backend configuration") from no_dt

        circuits = []
        for delay in self._delays:
            circ = qiskit.QuantumCircuit(1, 1)
            circ.h(0)
            circ.delay(delay, 0, self._unit)
            circ.p(2 * np.pi * self._osc_freq, 0)
            circ.barrier(0)
            circ.h(0)
            circ.barrier(0)
            circ.measure(0, 0)

            circ.metadata = {
                "experiment_type": self._type,
                "qubit": self._qubit,
                "osc_freq": self._osc_freq,
                "xval": delay,
                "unit": self._unit,
            }
            if self._unit == "dt":
                circ.metadata["dt_factor"] = dt_factor

            circuits.append(circ)

        return circuits


class RamseyXY(BaseExperiment):
    r"""Ramsey experiment for frequency calibration.

    This experiment differs from the :class:`~qiskit_experiments.characterization.\
    t2ramsey.T2Ramsey` by the sensitivity to the sign of frequency error.
    This experiment consists of following two circuits:

    .. parsed-literal::

        (Ramsey X) Second pulse is SX

                   ┌────┐┌─────────────┐┌────┐ ░ ┌─┐
              q_0: ┤ √X ├┤ Delay(τ[s]) ├┤ √X ├─░─┤M├
                   └────┘└─────────────┘└────┘ ░ └╥┘
        measure: 1/═══════════════════════════════╩═
                                                  0

        (Ramsey Y) Second pulse is SY

                   ┌────┐┌─────────────┐┌──────────┐┌────┐ ░ ┌─┐
              q_0: ┤ √X ├┤ Delay(τ[s]) ├┤ Rz(-π/2) ├┤ √X ├─░─┤M├
                   └────┘└─────────────┘└──────────┘└────┘ ░ └╥┘
        measure: 1/═══════════════════════════════════════════╩═
                                                              0

    The first (second) circuit measures :math:`\sigma_Y` (:math:`\sigma_X`) expectation value,
    so this experiment draws the dynamics of the Bloch vector as if drawing a Lissajous figure.

    Given the control electronics tracks the frame of qubit at the reference frequency
    which is slightly differ from the true qubit frequency by :math:`\Delta\omega`,
    and the IQ mixer skew can be ignored, we can describe the dynamics of
    two circuits as follows.

    The Hamiltonian during the ``Delay`` instruction can be written as
    :math:`H^R = - \frac{1}{2} \Delta\omega` in the rotating frame,
    and the propagator will be :math:`U(\tau) = \exp(-iH^R\tau)` where :math:`\tau` is the
    duration of the delay. By scanning this duration, we can get

    .. math::

        {\cal E}_x(\tau)
            = {\rm Re} {\rm Tr}\left( \sigma_Y U \rho U^\dagger \right)
            &= - \cos(\Delta\omega\tau) = \sin(\Delta\omega\tau - \frac{\pi}{2}), \\
        {\cal E}_y(\tau)
            = {\rm Re} {\rm Tr}\left( \sigma_X U \rho U^\dagger \right)
            &= \sin(\Delta\omega\tau),

    where :math:`\rho = | L \rangle` prepared by the :math:`\sqrt{\rm X}` gate in the beginning.

    Note that phase difference of these two outcomes :math:`{\cal E}_x, {\cal E}_y` depends on
    the frequency offset :math:`\Delta\omega`, which is also sensitive
    to the sign of it, in contrast to the standard Ramsey experiment which
    usually consists only of the first circuit,
    i.e. :math:`\cos(-\Delta\omega\tau) = \cos(\Delta\omega\tau)`.
    """

    __analysis_class__ = RamseyXYAnalysis

    def __init__(
            self,
            qubit: int,
            delays: Union[Iterable[float], np.ndarray],
            unit: str = "s",
            offset_frequency: float = 0.,
    ):
        """Initialize new experiment.

        Args:
            qubit: The qubit under test.
            delays: Delay times of the experiments.
            unit: Time unit of `delays`. One of 's', 'ms', 'us', 'ns', 'ps', 'dt'.
            offset_frequency: Frequency offset for this experiment.

        .. warning::

            ``offset_frequency`` doesn't really change the qubit frequency for now,
            because of the missing oscillator frequency upgrade pass in the circuit execution.
        """
        super().__init__([qubit])

        default_options = self.experiment_options
        default_options.delays = delays
        default_options.unit = unit
        default_options.offset_frequency = offset_frequency

        self.set_experiment_options(**default_options.__dict__)

    @classmethod
    def _default_experiment_options(cls) -> Options:
        return Options(
            delays=None,
            unit="s",
            offset_frequency=0.,
            calibrations={
                "sx": None,
                "measure": None,
            },
        )

    def circuits(self, backend: Optional[Backend] = None) -> List[QuantumCircuit]:
        """Return a list of experiment circuits.

        Args:
            backend: Optional, a backend object.

        Returns:
            A list of :class:`QuantumCircuit`.
        """
        options = self.experiment_options

        p_delay = Parameter("delay")

        ram_x = QuantumCircuit(1)
        ram_x.sx(0)
        ram_x.delay(p_delay, 0, unit=options.unit)
        ram_x.sx(0)
        ram_x.measure_active()

        ram_y = QuantumCircuit(1)
        ram_y.sx(0)
        ram_y.delay(p_delay, 0, unit=options.unit)
        ram_y.rz(-np.pi/2, 0)
        ram_y.sx(0)
        ram_y.measure_active()

        circs = []
        for delay in options.delays:

            # format delay to SI unit for analysis
            if options.unit == "dt":
                xval = delay * backend.configuration().dt
            else:
                if options.unit == "s":
                    xval = delay
                else:
                    xval = apply_prefix(delay, options.unit)

            metadata = {
                "experiment_type": self._type,
                "qubits": (self.physical_qubits[0], ),
                "delay": delay,
                "unit": options.unit,
                "xval": xval,
            }

            # create ramsey x
            assigned_x = ram_x.assign_parameters({p_delay: delay}, inplace=False)
            assigned_x.metadata = metadata.copy()
            assigned_x.metadata["post_pulse"] = "x"

            # create ramsey y
            assigned_y = ram_y.assign_parameters({p_delay: delay}, inplace=False)
            assigned_y.metadata = metadata.copy()
            assigned_y.metadata["post_pulse"] = "y"

            circs.extend([assigned_x, assigned_y])

        return circs

    def _postprocess_transpiled_circuits(self, circuits, backend, **run_options):
        """Attach gate calibration if available.

        TODO need discussion,
        - Ramsey should be able to run without calibration, same for T1 T2
        - However we should be able to add calibration to any gate
        """
        for circ in circuits:
            apply_calibration(self.experiment_options.calibrations, circ)


CalibrationEntry = Dict[
    Tuple[Tuple[int, ...], Tuple[Union[complex, ParameterExpression], ...]],
    ScheduleBlock
]


def apply_calibration(
        calibrations: Dict[str, CalibrationEntry],
        circuit: QuantumCircuit,
) -> QuantumCircuit:
    """A helper function to add calibration.

    TODO need discussion
    """
    gates = list(circuit.count_ops().values())

    for gate in gates:
        entries = calibrations.get(gate, None)
        if not entries:
            continue
        for qubit_parameters, schedule in entries.items():
            circuit.add_calibration(
                gate=gate,
                qubits=qubit_parameters[0],
                schedule=schedule,
                params=qubit_parameters[1],
            )
    return circuit
