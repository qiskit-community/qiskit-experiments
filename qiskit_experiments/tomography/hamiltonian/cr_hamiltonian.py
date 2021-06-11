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

from typing import Tuple, List, Dict, Any, Optional, Union

import numpy as np
from qiskit import QuantumCircuit, pulse, circuit
from qiskit.exceptions import QiskitError
from qiskit.providers import Backend
from qiskit.providers.options import Options
from qiskit.utils import apply_prefix

from qiskit_experiments import BaseExperiment
from qiskit_experiments.analysis import CurveAnalysis, SeriesDef, CurveAnalysisResult
from qiskit_experiments.analysis.utils import get_opt_value, get_opt_error, frequency_guess
from qiskit_experiments.analysis.data_processing import expectation_value


def oscillation_x(x: np.ndarray, px: float, py: float, pz: float, b: float):
    """Fit function for x basis oscillation."""
    omega = np.sqrt(px**2 + py**2 + pz**2)
    return (-pz * px + pz * px * np.cos(omega * x) + omega * py * np.sin(omega * x)) / omega**2 + b


def oscillation_y(x: np.ndarray, px: float, py: float, pz: float, b: float):
    """Fit function for y basis oscillation."""
    omega = np.sqrt(px**2 + py**2 + pz**2)
    return (pz * py - pz * py * np.cos(omega * x) - omega * px * np.sin(omega * x)) / omega**2 + b


def oscillation_z(x: np.ndarray, px: float, py: float, pz: float, b: float):
    """Fit function for z basis oscillation."""
    omega = np.sqrt(px**2 + py**2 + pz**2)
    return (pz**2 + (px**2 + py**2) * np.cos(omega * x)) / omega**2 + b


class CRHamiltonianAnalysis(CurveAnalysis):

    __series__ = [
        SeriesDef(
            name="x|c=0",
            fit_func=lambda x, px0, px1, py0, py1, pz0, pz1, b: oscillation_x(
                x, px=px0, py=py0, pz=pz0, b=b
            ),
            filter_kwargs={"control_state": 0, "meas_basis": "x"},
            plot_color="blue",
            plot_symbol="o",
            canvas=0,
        ),
        SeriesDef(
            name="y|c=0",
            fit_func=lambda x, px0, px1, py0, py1, pz0, pz1, b: oscillation_y(
                x, px=px0, py=py0, pz=pz0, b=b
            ),
            filter_kwargs={"control_state": 0, "meas_basis": "y"},
            plot_color="blue",
            plot_symbol="o",
            canvas=1,
        ),
        SeriesDef(
            name="z|c=0",
            fit_func=lambda x, px0, px1, py0, py1, pz0, pz1, b: oscillation_z(
                x, px=px0, py=py0, pz=pz0, b=b
            ),
            filter_kwargs={"control_state": 0, "meas_basis": "z"},
            plot_color="blue",
            plot_symbol="o",
            canvas=2,
        ),
        SeriesDef(
            name="x|c=1",
            fit_func=lambda x, px0, px1, py0, py1, pz0, pz1, b: oscillation_x(
                x, px=px1, py=py1, pz=pz1, b=b
            ),
            filter_kwargs={"control_state": 1, "meas_basis": "x"},
            plot_color="red",
            plot_symbol="^",
            canvas=0,
        ),
        SeriesDef(
            name="y|c=1",
            fit_func=lambda x, px0, px1, py0, py1, pz0, pz1, b: oscillation_y(
                x, px=px1, py=py1, pz=pz1, b=b
            ),
            filter_kwargs={"control_state": 1, "meas_basis": "y"},
            plot_color="red",
            plot_symbol="^",
            canvas=1,
        ),
        SeriesDef(
            name="z|c=1",
            fit_func=lambda x, px0, px1, py0, py1, pz0, pz1, b: oscillation_z(
                x, px=px1, py=py1, pz=pz1, b=b
            ),
            filter_kwargs={"control_state": 1, "meas_basis": "z"},
            plot_color="red",
            plot_symbol="^",
            canvas=2,
        ),
    ]

    @classmethod
    def _default_options(cls):
        """Return default data processing options.

        See :meth:`~qiskit_experiment.analysis.CurveAnalysis._default_options` for
        descriptions of analysis options.
        """
        default_options = super()._default_options()
        default_options.data_processor = expectation_value()
        default_options.fig_size = (8, 10)
        default_options.xlabel = "CR duration (sec)"
        default_options.ylabel = r"$\langle\sigma_{X, Y, Z}\rangle$"
        default_options.p0 = {
            "px0": None, "px1": None, "py0": None, "py1": None, "pz0": None, "pz1": None, "b": None
        }
        default_options.bounds = {
            "px0": None, "px1": None, "py0": None, "py1": None, "pz0": None, "pz1": None, "b": None
        }
        default_options.fit_reports = {
            "IX": r"$\omega_{IX}$",
            "IY": r"$\omega_{IY}$",
            "IZ": r"$\omega_{IZ}$",
            "ZX": r"$\omega_{ZX}$",
            "ZY": r"$\omega_{ZY}$",
            "ZZ": r"$\omega_{ZZ}$",
        }

        return default_options

    def _setup_fitting(self, **options) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """Fitter options."""
        user_p0 = self._get_option("p0")
        user_bounds = self._get_option("bounds")

        init_guess = dict()
        for control in (0, 1):
            # initial guess of pz
            ts, exp_z, _ = self._subset_data(
                name=f"z|c={control}",
                data_index=self._data_index,
                x_values=self._x_values,
                y_values=self._y_values,
                y_sigmas=self._y_sigmas,
            )
            zrange_mean = np.mean(np.percentile(exp_z, [10, 90]))
            if zrange_mean < 0:
                pz_guess = 0.
            else:
                omega = 2 * np.pi * frequency_guess(ts, exp_z, method="FFT")
                pz_guess = omega * np.sqrt(zrange_mean)

            # initial guess of py
            ts, exp_x, _ = self._subset_data(
                name=f"x|c={control}",
                data_index=self._data_index,
                x_values=self._x_values,
                y_values=self._y_values,
                y_sigmas=self._y_sigmas,
            )
            py_guess = (exp_x[1] - exp_x[0]) / (ts[1] - ts[0])

            # initial guess of px
            ts, exp_y, _ = self._subset_data(
                name=f"y|c={control}",
                data_index=self._data_index,
                x_values=self._x_values,
                y_values=self._y_values,
                y_sigmas=self._y_sigmas,
            )
            px_guess = - (exp_y[1] - exp_y[0]) / (ts[1] - ts[0])

            init_guess[f"px{control}"] = user_p0[f"px{control}"] or px_guess
            init_guess[f"py{control}"] = user_p0[f"py{control}"] or py_guess
            init_guess[f"pz{control}"] = user_p0[f"pz{control}"] or pz_guess
        init_guess["b"] = user_p0["b"] or 0.

        fit_option = {
            "p0": init_guess,
            "bounds": {
                "px0": user_bounds["px0"] or (-np.inf, np.inf),
                "py0": user_bounds["py0"] or (-np.inf, np.inf),
                "pz0": user_bounds["pz0"] or (-np.inf, np.inf),
                "px1": user_bounds["px1"] or (-np.inf, np.inf),
                "py1": user_bounds["py1"] or (-np.inf, np.inf),
                "pz1": user_bounds["pz1"] or (-np.inf, np.inf),
                "b": user_bounds["b"] or (-0.1, 0.1),
            },
        }
        fit_option.update(options)

        return fit_option

    def _post_processing(self, analysis_result: CurveAnalysisResult) -> CurveAnalysisResult:
        """Calculate Hamiltonian coefficients."""

        for control in ("Z", "I"):
            for target in ("X", "Y", "Z"):
                p0_val = get_opt_value(analysis_result, f"p{target.lower()}0") / (2 * np.pi)
                p1_val = get_opt_value(analysis_result, f"p{target.lower()}1") / (2 * np.pi)
                p0_err = get_opt_error(analysis_result, f"p{target.lower()}0") / (2 * np.pi)
                p1_err = get_opt_error(analysis_result, f"p{target.lower()}1") / (2 * np.pi)
                if control == "Z":
                    coef = 0.5 * (p0_val - p1_val)
                else:
                    coef = 0.5 * (p0_val + p1_val)
                analysis_result[f"{control}{target}"] = coef
                analysis_result[f"{control}{target}_err"] = np.sqrt(p0_err**2 + p1_err**2)

        return analysis_result


class CRHamiltonianTomography(BaseExperiment):
    """Cross resonance Hamiltonian tomography experiment.

    This experiment is performed by stretching a pulse duration of the cross resonance
    and measuring the target qubit by projecting onto x, y, and z basis.
    Control qubit is prepared in both ground and excited state and above experiment is
    repeated for both control qubit states.
    See [1]_ for more technical details.

    In this experiment, user can override the cross resonance pulse schedule.
    This schedule defaults to a single cross resonance pulse with the flat-topped
    Gaussian envelope. This can be replaced with echoed CR with following code:

    .. code-block::

        duration = Parameter("duration")
        with build(backend, default_alignment="sequential") as ecr_sched:
            u_channel = control_channels(0, 1)[0]
            play(GaussianSquare(duration, 0.1, 64, duration-4*64), u_channel)
            x(0)
            play(GaussianSquare(duration, -0.1, 64, duration-4*64), u_channel)
            x(0)

        durations = np.asarray([300, 400, 500], dtype=int)

        ham_tomo = CRHamiltonianTomography(
            qubits=[0, 1],
            durations=durations,
            cr_gate_schedule=ecr_sched,
            x_values=durations * 2 * backend.configuration().dt
        )

    Note that schedule attached to the ``CRHamiltonianTomography`` should contain at least
    one parameter object with ``name="duration"`` otherwise it raises an error.
    In above example, the effect of rising and falling edges are not considered in x values.

    .. [1] https://arxiv.org/abs/1603.04821
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
            durations: The pulse durations to scan.
            cr_gate_schedule: CR gate schedule to measure. Defaults to a single cross resonance
                pulse with the flat-topped Gaussian envelope.
            x_values: Net interaction time of the cross resonance gate. This value is used
                to calculate Pauli coefficients of the CR Hamiltonian.
                This defaults to the net duration of the default CR pulse.
                The net CR pulse duration is determined by

                .. math::

                    \tau' = \tau - 2 {\rm (risefall)} \sigma + \sqrt{2\pi\sigma^2}

                where :math:`\sigma` is the standard deviation of Gaussian edges and
                `\tau` represents the total duration of the CR pulse.
                If `cr_gate_schedule` is specified without `x_values`,
                this will be identical to `durations` in units of second.

        Raises:
            QiskitError:
                - When `cr_gate_schedule` is specified without parameter named "`duration`".
                - When length of `x_values` and `durations` are different.
        """
        super().__init__(qubits)

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
        """
        self.set_experiment_options(dt=backend.configuration().dt)

        durations_dt = np.asarray(
            [self._to_samples(dur) for dur in self.experiment_options.durations], dtype=int
        )

        if not self._cr_gate_schedule:
            # defaults to the CR1 sequence
            par_duration = circuit.Parameter("duration")
            amp = self.experiment_options.amp
            sigma_dt = self._to_samples(self.experiment_options.sigma)
            risefall = self.experiment_options.risefall

            with pulse.build(backend=backend) as cr_sched:
                pulse.play(
                    pulse.GaussianSquare(
                        duration=par_duration,
                        amp=amp,
                        sigma=sigma_dt,
                        width=par_duration - 2 * risefall * sigma_dt,
                    ),
                    pulse.control_channels(*self.physical_qubits)[0]
                )
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

    def _to_samples(self, value):
        """A helper function to convert SI pulse length to samples.

        Args:
            value: A value to convert.

        Returns:
            Given value in samples.
        """
        unit = self.experiment_options.unit
        if unit != "dt":
            return int(np.round(apply_prefix(value, unit) / self.experiment_options.dt))
        else:
            return int(value)
