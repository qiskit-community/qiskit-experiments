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

"""Rabi amplitude experiment."""

from typing import Any, Dict, List, Optional, Union
import numpy as np

from qiskit import QiskitError, QuantumCircuit
from qiskit.circuit import Gate, Parameter
from qiskit.qobj.utils import MeasLevel
from qiskit.providers import Backend
import qiskit.pulse as pulse
from qiskit.providers.options import Options

from qiskit_experiments.analysis import (
    CurveAnalysis,
    CurveAnalysisResult,
    SeriesDef,
    fit_function,
    guess,
    get_opt_value,
    get_opt_error,
)
from qiskit_experiments.base_experiment import BaseExperiment


class RabiAnalysis(CurveAnalysis):
    r"""Rabi analysis class based on a fit to a cosine function.

    Analyse a Rabi experiment by fitting it to a cosine function

    .. math::
        y = amp \cos\left(2 \pi {\rm freq} x + {\rm phase}\right) + baseline

    Fit Parameters
        - :math:`amp`: Amplitude of the oscillation.
        - :math:`baseline`: Base line.
        - :math:`{\rm freq}`: Frequency of the oscillation. This is the fit parameter of interest.
        - :math:`{\rm phase}`: Phase of the oscillation.

    Initial Guesses
        - :math:`amp`: Calculated by :func:`~qiskit_experiments.analysis.guess.max_height`.
        - :math:`baseline`: Calculated by :func:`~qiskit_experiments.analysis.\
          guess.constant_sinusoidal_offset`.
        - :math:`{\rm freq}`: Calculated by :func:`~qiskit_experiments.analysis.\
          guess.frequency`.
        - :math:`{\rm phase}`: Zero.

    Bounds
        - :math:`amp`: [-2, 2] scaled to the maximum signal value.
        - :math:`baseline`: [-1, 1] scaled to the maximum signal value.
        - :math:`{\rm freq}`: [0, inf].
        - :math:`{\rm phase}`: [-pi, pi].
    """

    __series__ = [
        SeriesDef(
            fit_func=lambda x, amp, freq, phase, baseline: fit_function.cos(
                x, amp=amp, freq=freq, phase=phase, baseline=baseline
            ),
            plot_color="blue",
        )
    ]

    @classmethod
    def _default_options(cls):
        """Return the default analysis options.

        See :meth:`~qiskit_experiment.analysis.CurveAnalysis._default_options` for
        descriptions of analysis options.
        """
        default_options = super()._default_options()
        default_options.p0 = {"amp": None, "freq": None, "phase": None, "baseline": None}
        default_options.bounds = {"amp": None, "freq": None, "phase": None, "baseline": None}
        default_options.fit_reports = {"freq": "rate"}
        default_options.xlabel = "Amplitude"
        default_options.ylabel = "Signal (arb. units)"

        return default_options

    def _setup_fitting(self, **options) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """Fitter options."""
        user_p0 = self._get_option("p0")
        user_bounds = self._get_option("bounds")

        curve_data = self._data()

        max_abs_y = np.max(np.abs(curve_data.y))

        f_guess = guess.frequency(curve_data.x, curve_data.y)
        b_guess = guess.constant_sinusoidal_offset(curve_data.y)
        a_guess, _ = guess.max_height(curve_data.y - b_guess, absolute=True)

        if user_p0["phase"] is not None:
            p_guesses = [user_p0["phase"]]
        else:
            p_guesses = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4, np.pi]

        fit_options = []
        for p_guess in p_guesses:
            fit_option = {
                "p0": {
                    "amp": user_p0["amp"] or a_guess,
                    "freq": user_p0["freq"] or f_guess,
                    "phase": p_guess,
                    "baseline": user_p0["baseline"] or b_guess,
                },
                "bounds": {
                    "amp": user_bounds["amp"] or (-2 * max_abs_y, 2 * max_abs_y),
                    "freq": user_bounds["freq"] or (0, np.inf),
                    "phase": user_bounds["phase"] or (-np.pi, np.pi),
                    "baseline": user_bounds["baseline"] or (-1 * max_abs_y, 1 * max_abs_y),
                },
            }
            fit_option.update(options)
            fit_options.append(fit_option)

        return fit_options

    def _post_analysis(self, analysis_result: CurveAnalysisResult) -> CurveAnalysisResult:
        """Algorithmic criteria for whether the fit is good or bad.

        A good fit has:
            - a reduced chi-squared lower than three,
            - more than a quarter of a full period,
            - less than 10 full periods, and
            - an error on the fit frequency lower than the fit frequency.
        """
        fit_freq = get_opt_value(analysis_result, "freq")
        fit_freq_err = get_opt_error(analysis_result, "freq")

        criteria = [
            analysis_result["reduced_chisq"] < 3,
            1.0 / 4.0 < fit_freq < 10.0,
            (fit_freq_err is None or (fit_freq_err < fit_freq)),
        ]

        if all(criteria):
            analysis_result["quality"] = "computer_good"
        else:
            analysis_result["quality"] = "computer_bad"

        return analysis_result


class Rabi(BaseExperiment):
    """An experiment that scans the amplitude of a pulse to calibrate rotations between 0 and 1.

    The circuits that are run have a custom rabi gate with the pulse schedule attached to it
    through the calibrations. The circuits are of the form:

    .. parsed-literal::

                   ┌───────────┐ ░ ┌─┐
              q_0: ┤ Rabi(amp) ├─░─┤M├
                   └───────────┘ ░ └╥┘
        measure: 1/═════════════════╩═
                                    0
    """

    __analysis_class__ = RabiAnalysis
    __rabi_gate_name__ = "Rabi"

    @classmethod
    def _default_run_options(cls) -> Options:
        """Default option values for the experiment :meth:`run` method."""
        return Options(
            meas_level=MeasLevel.KERNELED,
            meas_return="single",
        )

    @classmethod
    def _default_experiment_options(cls) -> Options:
        """Default values for the pulse if no schedule is given.

        Users can set a schedule by doing

        .. code-block::

            rabi.set_experiment_options(schedule=rabi_schedule)

        """
        return Options(
            duration=160,
            sigma=40,
            amplitudes=np.linspace(-0.95, 0.95, 51),
            schedule=None,
        )

    @classmethod
    def _default_analysis_options(cls) -> Options:
        """Default analysis options."""
        options = super()._default_analysis_options()
        options.normalization = True

        return options

    def __init__(self, qubit: int):
        """Setup a Rabi experiment on the given qubit.

        The parameters of the Gaussian Rabi pulse can be specified at run-time.
        The rabi pulse has the following parameters:
            - duration: The duration of the rabi pulse in samples, the default is 160 samples.
            - sigma: The standard deviation of the pulse, the default is duration 40.
            - amplitudes: The amplitude that are scanned in the experiment, default  is
                np.linspace(-0.95, 0.95, 51)

        Args:
            qubit: The qubit on which to run the Rabi experiment.
        """
        super().__init__([qubit])

    def _template_circuit(self, amp_param) -> QuantumCircuit:
        """Return the template quantum circuit."""
        gate = Gate(name=self.__rabi_gate_name__, num_qubits=1, params=[amp_param])

        circuit = QuantumCircuit(1)
        circuit.append(gate, (0,))
        circuit.measure_active()

        return circuit

    def _default_gate_schedule(self, backend: Optional[Backend] = None):
        """Create the default schedule for the Rabi gate."""
        amp = Parameter("amp")
        with pulse.build(backend=backend, name="rabi") as default_schedule:
            pulse.play(
                pulse.Gaussian(
                    duration=self.experiment_options.duration,
                    amp=amp,
                    sigma=self.experiment_options.sigma,
                ),
                pulse.DriveChannel(self.physical_qubits[0]),
            )

        return default_schedule

    def circuits(self, backend: Optional[Backend] = None) -> List[QuantumCircuit]:
        """Create the circuits for the Rabi experiment.

        Args:
            backend: A backend object.

        Returns:
            A list of circuits with a rabi gate with an attached schedule. Each schedule
            will have a different value of the scanned amplitude.

        Raises:
            QiskitError:
                - If the user-provided schedule does not contain a channel with an index
                  that matches the qubit on which to run the Rabi experiment.
                - If the user provided schedule has more than one free parameter.
        """
        schedule = self.experiment_options.get("schedule", None)

        if schedule is None:
            schedule = self._default_gate_schedule(backend=backend)
        else:
            if self.physical_qubits[0] not in set(ch.index for ch in schedule.channels):
                raise QiskitError(
                    f"User provided schedule {schedule.name} does not contain a channel "
                    "for the qubit on which to run Rabi."
                )

        if len(schedule.parameters) != 1:
            raise QiskitError("Schedule in Rabi must have exactly one free parameter.")

        param = next(iter(schedule.parameters))

        # Create template circuit
        circuit = self._template_circuit(param)
        circuit.add_calibration(
            self.__rabi_gate_name__, (self.physical_qubits[0],), schedule, params=[param]
        )

        # Create the circuits to run
        circs = []
        for amp in self.experiment_options.amplitudes:
            amp = np.round(amp, decimals=6)
            assigned_circ = circuit.assign_parameters({param: amp}, inplace=False)
            assigned_circ.metadata = {
                "experiment_type": self._type,
                "qubits": (self.physical_qubits[0],),
                "xval": amp,
                "unit": "arb. unit",
                "amplitude": amp,
                "schedule": str(schedule),
            }

            if backend:
                assigned_circ.metadata["dt"] = getattr(backend.configuration(), "dt", "n.a.")

            circs.append(assigned_circ)

        return circs


class EFRabi(Rabi):
    """An experiment that scans the amplitude of a pulse to calibrate rotations between 1 and 2.

    This experiment is similar to the Rabi experiment between the ground and first excited state.
    The difference is in the initial X gate used to populate the first excited state. The Rabi pulse
    is then applied on the 1 <-> 2 transition (sometimes also labeled the e <-> f transition) which
    implies that frequency shift instructions are used. The necessary frequency shift (typically the
    qubit anharmonicity) should be specified through the experiment options.

    The circuits are of the form:

    .. parsed-literal::

                   ┌───┐┌───────────┐ ░ ┌─┐
              q_0: ┤ X ├┤ Rabi(amp) ├─░─┤M├
                   └───┘└───────────┘ ░ └╥┘
        measure: 1/══════════════════════╩═
                                         0
    """

    @classmethod
    def _default_experiment_options(cls) -> Options:
        """Default values for the pulse if no schedule is given.

        Users can set a schedule by doing

        .. code-block::

            ef_rabi.set_experiment_options(schedule=rabi_schedule)

        """
        return Options(
            duration=160,
            sigma=40,
            amplitudes=np.linspace(-0.95, 0.95, 51),
            schedule=None,
            normalization=True,
            frequency_shift=None,
        )

    def _default_gate_schedule(self, backend: Optional[Backend] = None):
        """Create the default schedule for the EFRabi gate with a frequency shift to the 1-2
        transition."""

        if self.experiment_options.frequency_shift is None:
            try:
                anharm, _ = backend.properties.qubit_property(self.physical_qubits[0])[
                    "anharmonicity"
                ]
                self.set_experiment_options(frequency_shift=anharm)
            except KeyError as key_err:
                raise QiskitError(
                    f"The backend {backend} does not provide an anharmonicity for qubit "
                    f"{self.physical_qubits[0]}. Use EFRabi.set_experiment_options(frequency_shift="
                    f"anharmonicity) to manually set the correct frequency for the 1-2 transition."
                ) from key_err
            except AttributeError as att_err:
                raise QiskitError(
                    "When creating the default schedule without passing a backend, the frequency needs "
                    "to be set manually through EFRabi.set_experiment_options(frequency_shift=..)."
                ) from att_err

        amp = Parameter("amp")
        with pulse.build(backend=backend, name=self.__rabi_gate_name__) as default_schedule:
            with pulse.frequency_offset(
                self.experiment_options.frequency_shift,
                pulse.DriveChannel(self.physical_qubits[0]),
            ):
                pulse.play(
                    pulse.Gaussian(
                        duration=self.experiment_options.duration,
                        amp=amp,
                        sigma=self.experiment_options.sigma,
                    ),
                    pulse.DriveChannel(self.physical_qubits[0]),
                )

        return default_schedule

    def _template_circuit(self, amp_param) -> QuantumCircuit:
        """Return the template quantum circuit."""
        circuit = QuantumCircuit(1)
        circuit.x(0)
        circuit.append(Gate(name=self.__rabi_gate_name__, num_qubits=1, params=[amp_param]), (0,))
        circuit.measure_active()

        return circuit
