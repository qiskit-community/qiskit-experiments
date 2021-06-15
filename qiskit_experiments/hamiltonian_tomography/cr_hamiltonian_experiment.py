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
Cross resonance Hamiltonian tomography.
"""

from typing import Tuple, List, Optional, Union

import numpy as np
from qiskit import QuantumCircuit, pulse, circuit
from qiskit.exceptions import QiskitError
from qiskit.providers import Backend
from qiskit.providers.options import Options
from qiskit.utils import apply_prefix

from qiskit_experiments import BaseExperiment
from .cr_hamiltonian_analysis import CRHamiltonianAnalysis


class CRHamiltonianTomography(BaseExperiment):
    r"""Cross resonance Hamiltonian tomography experiment.

    This experiment assumes the 2 qubit Hamiltonian in the form

    .. math::

        H = \frac{I \otimes A}{2} + \frac{Z \otimes B}{2}

    where :math:`A` and :math:`B` are Pauli operator :math:`\in {X, Y, Z}`.

    This experiment is performed by stretching a pulse duration of the cross resonance
    and measuring the target qubit by projecting onto x, y, and z basis.
    Control qubit is prepared in both ground and excited state and the experiment is
    repeated for both control qubit states.
    See [SarahSheldon2016]_ for more technical details.

    In this experiment, user can override the cross resonance pulse schedule.
    This schedule defaults to a single cross resonance pulse with the flat-topped
    Gaussian envelope (:class:`~qiskit.pulse.GaussianSquare`).
    This can be replaced with the echoed CR with following code:

    .. code-block::

        # define parameter object. this should be defined with name="duration".
        duration = Parameter("duration")

        with build(backend, default_alignment="sequential") as ecr_sched:
            u_channel = control_channels(0, 1)[0]
            play(GaussianSquare(duration, 0.1, 64, duration-4*64), u_channel)
            x(0)
            play(GaussianSquare(duration, -0.1, 64, duration-4*64), u_channel)
            x(0)

        # create experiment with ECR pulse schedule.
        durations = np.asarray([300, 400, 500], dtype=int)

        ham_tomo = CRHamiltonianTomography(
            qubits=[0, 1],
            durations=durations,
            cr_gate_schedule=ecr_sched,
            x_values=durations * 2 * backend.configuration().dt
        )

    Note that schedule registered to the ``CRHamiltonianTomography`` should contain at least
    one parameter object with ``name="duration"`` otherwise it raises an error.
    In above example, the effect of rising and falling edges are not considered in x values.

    .. [SarahSheldon2016] https://arxiv.org/abs/1603.04821
    """
    __analysis_class__ = CRHamiltonianAnalysis

    def __init__(
            self,
            qubits: Tuple[int, int],
            durations: np.ndarray,
            cr_gate_schedule: Optional[Union[pulse.Schedule, pulse.ScheduleBlock]] = None,
            x_values: Optional[np.ndarray] = None
    ):
        r"""Create new CR Hamiltonian tomography experiment.

        Args:
            qubits: The qubit on which to run tomography.
            durations: The pulse durations to scan. The maximum value should be sufficiently long
                so that at least one cycle of oscillation of
                the Bloch vector can be observed in the Bloch sphere of the target qubit.
                Typically 100s ns to 10s μs depending on the drive power and
                qubit-qubit coupling strength.
            cr_gate_schedule: CR gate schedule to measure. Defaults to a single cross resonance
                pulse with the flat-topped Gaussian envelope.
            x_values: Net interaction time of the cross resonance gate. This value is used
                to calculate Pauli coefficients of the CR Hamiltonian.
                This defaults to the net duration of the single CR pulse.
                The net CR pulse duration is determined by

                .. math::

                    \tau' = \tau - 2 {\rm (risefall)} \sigma + \sqrt{2\pi\sigma^2}

                where :math:`\sigma` is the standard deviation of Gaussian edges and
                `\tau` represents the total duration of the CR pulse.
                If ``cr_gate_schedule`` is overridden without ``x_values``,
                this will be identical to ``durations`` in units of second.

        Raises:
            QiskitError: When qubit number is not 2.
            QiskitError: When ``cr_gate_schedule`` is specified without parameter named "duration".
            QiskitError: When length of ``x_values`` and ``durations`` are different.
        """
        super().__init__(qubits)

        if len(qubits) != 2:
            raise QiskitError(
                f"{self.__class__.__name__} is 2 qubit experiment. len({qubits}) != 2."
            )

        if cr_gate_schedule is not None and not cr_gate_schedule.get_parameters("duration"):
            raise QiskitError(
                "Target cross resonance schedule doesn't contain the parameter `duration`."
            )

        if x_values is not None and len(durations) != len(x_values):
            raise QiskitError(
                "Length of x values are not equivalent to durations; "
                f"{len(self.experiment_options.durations)} != {len(x_values)}."
            )

        self._cr_gate_schedule = cr_gate_schedule
        self._x_values = x_values

        self.set_experiment_options(durations=durations)

    @classmethod
    def _default_experiment_options(cls) -> Options:
        """Default option values used for CR Hamiltonian tomography."""
        return Options(
            amp=0.1,
            sigma=64,
            risefall=2,
            durations=None,
            alignment=16,
            unit="dt",
            dt=None,
        )

    def circuits(self, backend: Optional[Backend] = None) -> List[QuantumCircuit]:
        """Create the circuits for the CR Hamiltonian experiment.

        Args:
            backend: A backend object.

        Returns:
            circuits: The circuits that will run the spectroscopy experiment.

        Raises:
            QiskitError: When `dt` information is not provided.
        """
        if self.experiment_options.dt is None:
            try:
                self.set_experiment_options(dt=backend.configuration().dt)
            except AttributeError as ex:
                raise QiskitError(
                    "Duration is not set and the backend doesn't provide dt. "
                    "Set time resolution of the system to proceed."
                ) from ex

        durations_dt = np.asarray(
            [self._to_samples(dur) for dur in self.experiment_options.durations], dtype=int
        )

        if not self._cr_gate_schedule:
            # defaults to the CR1 sequence
            par_duration = circuit.Parameter("duration")
            amp = self.experiment_options.amp
            sigma_dt = self._to_samples(self.experiment_options.sigma, require_integer=False)
            risefall = self.experiment_options.risefall

            with pulse.build(backend=backend) as cr_sched:
                with pulse.align_left():
                    pulse.play(
                        pulse.GaussianSquare(
                            duration=par_duration,
                            amp=amp,
                            sigma=sigma_dt,
                            width=par_duration - 2 * risefall * sigma_dt,
                        ),
                        pulse.control_channels(*self.physical_qubits)[0]
                    )
                    pulse.delay(par_duration, pulse.drive_channel(self.physical_qubits[0]))
                    pulse.delay(par_duration, pulse.drive_channel(self.physical_qubits[1]))
            cr_sched.metadata = {
                "amplitude": amp,
                "sigma": sigma_dt,
                "risefall": risefall,
            }

            if self._x_values is None:
                flat_top_width_dt = durations_dt - 2 * risefall * sigma_dt
                net_risefall_dt = np.sqrt(2 * np.pi * sigma_dt ** 2)
                x_values = (flat_top_width_dt + net_risefall_dt) * self.experiment_options.dt
            else:
                x_values = self._x_values
        else:
            # user defined CR sequence
            cr_sched = self._cr_gate_schedule

            if self._x_values is None:
                x_values = durations_dt * self.experiment_options.dt
            else:
                x_values = self._x_values

        par_durations = list(cr_sched.get_parameters("duration"))

        # create tomography circuits
        cr_gate = circuit.Gate(name="cr_gate", num_qubits=2, params=par_durations)
        temp_circs = list()
        for control_state in (0, 1):
            for meas_basis in ("x", "y", "z"):
                tomo_circ = circuit.QuantumCircuit(2, 1)
                if control_state:
                    tomo_circ.x(0)
                tomo_circ.append(cr_gate, [0, 1])
                if meas_basis == "x":
                    tomo_circ.h(1)
                elif meas_basis == "y":
                    tomo_circ.sdg(1)
                    tomo_circ.h(1)
                elif meas_basis == "z":
                    tomo_circ.id(1)
                tomo_circ.measure(1, 0)

                # add pulse gate schedule
                tomo_circ.add_calibration(
                    gate=cr_gate,
                    qubits=self.physical_qubits,
                    schedule=cr_sched,
                    params=par_durations,
                )
                # add metadata
                tomo_circ.metadata = {"control_state": control_state, "meas_basis": meas_basis}

                temp_circs.append(tomo_circ)

        experiment_circs = list()
        alignment = self.experiment_options.alignment
        for x_val, duration in zip(x_values, durations_dt):
            # consider hardware sample alignment constraint
            duration = int(alignment * np.round(duration / alignment))

            for temp_circ in temp_circs:
                value_dict = {par_dur: duration for par_dur in par_durations}
                experiment_circ = temp_circ.assign_parameters(value_dict, inplace=False)

                # add more information to metadata
                experiment_metadata = {
                    "experiment_type": self._type,
                    "qubits": self.physical_qubits,
                    "xval": x_val,
                    "dt": self.experiment_options.dt,
                    "duration": duration,
                }
                experiment_metadata.update(experiment_circ.metadata)
                experiment_metadata.update(cr_sched.metadata)

                experiment_circ.metadata = experiment_metadata

                experiment_circs.append(experiment_circ)

        return experiment_circs

    def _to_samples(self, value, require_integer: bool = True):
        """A helper function to convert SI pulse length to samples.

        Args:
            value: A value to convert.
            require_integer: Set ``True`` to return integer value.

        Returns:
            Given value in samples.
        """
        unit = self.experiment_options.unit
        if unit != "dt":
            if unit != "s":
                value = apply_prefix(value, unit) / self.experiment_options.dt
            else:
                value = value / self.experiment_options.dt

        if require_integer:
            return int(np.round(value))
        else:
            return value
