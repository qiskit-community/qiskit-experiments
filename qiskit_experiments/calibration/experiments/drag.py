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

"""Rough drag pulse calibration experiment."""

from typing import Any, Dict, List, Optional, Union
import numpy as np

from qiskit import QuantumCircuit
from qiskit.circuit import Gate, Parameter
from qiskit.qobj.utils import MeasLevel
from qiskit.providers import Backend
import qiskit.pulse as pulse
from qiskit.providers.options import Options

from qiskit_experiments.analysis import (
    CurveAnalysis,
    CurveAnalysisResult,
    SeriesDef,
    get_opt_value,
    get_opt_error,
)
from qiskit_experiments.base_experiment import BaseExperiment
from qiskit_experiments.data_processing.processor_library import get_to_signal_processor
from qiskit_experiments.calibration.exceptions import CalibrationError
from qiskit_experiments.analysis.fit_function import cos


class DragCalAnalysis(CurveAnalysis):
    r"""Drag calibration analysis based on a fit to a cosine function.

    Analyse a Drag calibration experiment by fitting three series each to a cosine function.
    The three functions share the phase parameter (i.e. beta) but each have their own amplitude,
    baseline, and frequency parameters (which therefore depend on the number of repetitions of
    xp-xm). Several initial guesses are tried if the user does not provide one.

    .. math::
        y = amp_i \cos\left(2 \pi {\rm freq_i} x - 2 \pi {\rm beta}\right) + {\rm base}_i

    Fit Parameters
        - :math:`{\rm amp}_i`: Amplitude of series :math:`i`.
        - :math:`{\rm base}_i`: Base line of series :math:`i`.
        - :math:`{\rm freq}_i`: Frequency of the :math:`i`th oscillation.
        - :math:`{\rm beta}`: Common beta offset. This is the parameter of interest.

    Initial Guesses
        - :math:`{\rm amp}_i`: The maximum y value less the minimum y value. 0.5 is also tried.
        - :math:`{\rm base}_i`: The average of the data. 0.5 is also tried.
        - :math:`{\rm freq}_i`: The frequency with the highest power spectral density.
        - :math:`{\rm beta}`: Linearly spaced between the maximum and minimum scanned beta.

    Bounds
        - :math:`{\rm amp}_i`: [-2, 2] scaled to the maximum signal value.
        - :math:`{\rm base}_i`: [-1, 1] scaled to the maximum signal value.
        - :math:`{\rm freq}_i`: [0, inf].
        - :math:`{\rm beta}`: [-min scan range, max scan range].

    """

    __series__ = [
        SeriesDef(
            fit_func=lambda x, amp0, amp1, amp2, freq0, freq1, freq2, beta, base0, base1, base2: cos(
                x, amp=amp0, freq=freq0, phase=-2 * np.pi * freq0 * beta, baseline=base0
            ),
            plot_color="blue",
            name="series-0",
            filter_kwargs={"series": 0},
            plot_symbol="o",
        ),
        SeriesDef(
            fit_func=lambda x, amp0, amp1, amp2, freq0, freq1, freq2, beta, base0, base1, base2: cos(
                x, amp=amp1, freq=freq1, phase=-2 * np.pi * freq1 * beta, baseline=base1
            ),
            plot_color="green",
            name="series-1",
            filter_kwargs={"series": 1},
            plot_symbol="^",
        ),
        SeriesDef(
            fit_func=lambda x, amp0, amp1, amp2, freq0, freq1, freq2, beta, base0, base1, base2: cos(
                x, amp=amp2, freq=freq2, phase=-2 * np.pi * freq2 * beta, baseline=base2
            ),
            plot_color="red",
            name="series-2",
            filter_kwargs={"series": 2},
            plot_symbol="v",
        ),
    ]

    @classmethod
    def _default_options(cls):
        """Return the default analysis options.

        See :meth:`~qiskit_experiment.analysis.CurveAnalysis._default_options` for
        descriptions of analysis options.
        """
        default_options = super()._default_options()
        default_options.p0 = {
            "amp0": None,
            "amp1": None,
            "amp2": None,
            "freq0": None,
            "freq1": None,
            "freq2": None,
            "beta": None,
            "base0": None,
            "base1": None,
            "base2": None,
        }
        default_options.bounds = {
            "amp0": None,
            "amp1": None,
            "amp2": None,
            "freq0": None,
            "freq1": None,
            "freq2": None,
            "beta": None,
            "base0": None,
            "base1": None,
            "base2": None,
        }
        default_options.fit_reports = {"beta": "beta"}
        default_options.xlabel = "Beta"
        default_options.ylabel = "Signal (arb. units)"

        return default_options

    def _setup_fitting(self, **options) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """Compute the initial guesses."""
        user_p0 = self._get_option("p0")
        user_bounds = self._get_option("bounds")

        # Use a fast Fourier transform to guess the frequency.
        x_data = self._data("series-0").x
        delta_beta = x_data[1] - x_data[0]

        min_beta, max_beta = min(x_data), max(x_data)

        freq_guess = []
        amp_guesses = []
        base_guesses = []
        for series in ["series-0", "series-1", "series-2"]:
            y_data = self._data(series).y
            fft = np.abs(np.fft.fft(y_data - np.average(y_data)))
            freqs = np.linspace(0.0, 1.0 / (2.0 * delta_beta), len(fft))
            freq_guess.append(freqs[np.argmax(fft[0 : len(fft) // 2])])

            b_guess = np.average(y_data)
            amp_guesses.append(np.max(y_data) - np.min(y_data) - b_guess)
            base_guesses.append(b_guess)

        if user_p0.get("beta", None) is not None:
            p_guesses = [user_p0["beta"]]
        else:
            p_guesses = np.linspace(min_beta, max_beta, 20)

        user_amps = None
        if all(user_p0.get("amp" + idx, None) is not None for idx in ["0", "1", "2"]):
            user_amps = list(user_p0["amp" + idx] for idx in ["0", "1", "2"])

        user_base = None
        if all(user_p0.get("base" + idx, None) is not None for idx in ["0", "1", "2"]):
            user_base = list(user_p0["base" + idx] for idx in ["0", "1", "2"])

        # Drag curves can sometimes be very flat, i.e. averages of y-data
        # and min-max do not always make good initial guesses. We therefore add
        # 0.5 to the initial guesses.
        guesses = [([0.5] * 3, [0.5] * 3), (amp_guesses, base_guesses)]

        if user_amps is not None and user_base is not None:
            guesses.append((user_amps, user_base))

        max_abs_y = np.max(np.abs(self._data().y))

        fit_options = []
        for amp_guess, b_guess in guesses:
            for p_guess in p_guesses:
                fit_option = {
                    "p0": {
                        "amp0": amp_guess[0],
                        "amp1": amp_guess[1],
                        "amp2": amp_guess[2],
                        "freq0": user_p0.get("freq0", None) or freq_guess[0],
                        "freq1": user_p0.get("freq1", None) or freq_guess[1],
                        "freq2": user_p0.get("freq2", None) or freq_guess[2],
                        "beta": p_guess,
                        "base0": b_guess[0],
                        "base1": b_guess[1],
                        "base2": b_guess[2],
                    },
                    "bounds": {
                        "amp0": user_bounds.get("amp0", None) or (-2 * max_abs_y, 2 * max_abs_y),
                        "amp1": user_bounds.get("amp1", None) or (-2 * max_abs_y, 2 * max_abs_y),
                        "amp2": user_bounds.get("amp2", None) or (-2 * max_abs_y, 2 * max_abs_y),
                        "freq0": user_bounds.get("freq0", None) or (0, np.inf),
                        "freq1": user_bounds.get("freq1", None) or (0, np.inf),
                        "freq2": user_bounds.get("freq2", None) or (0, np.inf),
                        "beta": user_bounds.get("beta", None) or (min_beta, max_beta),
                        "base0": user_bounds.get("base0", None) or (-1 * max_abs_y, 1 * max_abs_y),
                        "base1": user_bounds.get("base1", None) or (-1 * max_abs_y, 1 * max_abs_y),
                        "base2": user_bounds.get("base2", None) or (-1 * max_abs_y, 1 * max_abs_y),
                    },
                }

                fit_option.update(options)
                fit_options.append(fit_option)

        return fit_options

    def _post_analysis(self, analysis_result: CurveAnalysisResult) -> CurveAnalysisResult:
        """Algorithmic criteria for whether the fit is good or bad.

        A good fit has:
            - a reduced chi-squared lower than three,
            - a drag parameter value contained in the range of x-values, and
            - an error on the drag beta smaller than the beta.
        """

        fit_beta = get_opt_value(analysis_result, "beta")
        fit_beta_err = get_opt_error(analysis_result, "beta")

        x_data = self._data("series-0").x
        min_x, max_x = min(x_data), max(x_data)

        criteria = [
            analysis_result["reduced_chisq"] < 3,
            min_x <= fit_beta <= max_x,
            fit_beta_err < abs(fit_beta),
        ]

        if all(criteria):
            analysis_result["quality"] = "computer_good"
        else:
            analysis_result["quality"] = "computer_bad"

        return analysis_result


class DragCal(BaseExperiment):
    """An experiment that scans the drag parameter to find the optimal value.

    The Drag calibration will run several series of circuits. In a given circuit
    a xp(β) - xm(β) block is repeated :math:`N` times. As example the circuit of a single
    repetition, i.e. :math:`N=1`, is shown below.

    .. parsed-literal::

                   ┌───────┐ ░ ┌───────┐ ░ ┌─┐
              q_0: ┤ xp(β) ├─░─┤ xm(β) ├─░─┤M├
                   └───────┘ ░ └───────┘ ░ └╥┘
        measure: 1/═════════════════════════╩═
                                            0

    Here, the xp gate and the xm gate are intended to be pi and -pi rotations about the
    x-axis of the Bloch sphere. The parameter β is scanned to find the value that minimizes
    the leakage to the second excited state. Note that the analysis class requires this
    experiment to run with three repetition numbers.
    """

    __analysis_class__ = DragCalAnalysis

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
        Users can set the xm and xp schedules with

        .. code-block::

            drag.set_experiment_options(xp=xp_schedule, xm=xm_schedule)
        """
        options = super()._default_experiment_options()

        options.xp = None
        options.xm = None
        options.amp = 0.2
        options.duration = 160
        options.sigma = 40
        options.reps = [1, 3, 5]
        options.betas = np.linspace(-5, 5, 51)

        return options

    # pylint: disable=arguments-differ
    def set_experiment_options(self, reps: Optional[List] = None, **fields):
        """Raise if reps has a length different from three.

        Raises:
            CalibrationError: if the number of repetitions is different from three.
        """

        if reps is None:
            reps = [1, 3, 5]

        if len(reps) != 3:
            raise CalibrationError(
                "As longs as analysis series cannot be dynamically updated "
                f"{self.__class__.__name__} must use exactly three repetition numbers. "
                f"Received {reps} with length {len(reps)} != 3."
            )

        super().set_experiment_options(reps=reps, **fields)

    def __init__(self, qubit: int):
        """
        Args:
            qubit: The qubit for which to run the Drag calibration.
        """

        super().__init__([qubit])

    def circuits(self, backend: Optional[Backend] = None) -> List[QuantumCircuit]:
        """Create the circuits for the Drag calibration.

        Args:
            backend: A backend object.

        Returns:
            circuits: The circuits that will run the Drag calibration.

        Raises:
            CalibrationError:
                - If the beta parameters in the xp and xm pulses are not the same.
                - If either the xp or xm pulse do not have at least one Drag pulse.
                - If the number of different repetition series is not three.
        """
        # TODO this is temporary logic.
        self.set_analysis_options(
            data_processor=get_to_signal_processor(
                meas_level=self.run_options.meas_level,
                meas_return=self.run_options.meas_return,
            ),
        )

        xm = self.experiment_options.xm
        xp = self.experiment_options.xp

        if xp is None:
            beta = Parameter("β")
            with pulse.build(backend=backend, name="xp") as xp:
                pulse.play(
                    pulse.Drag(
                        duration=self.experiment_options.duration,
                        amp=self.experiment_options.amp,
                        sigma=self.experiment_options.sigma,
                        beta=beta,
                    ),
                    pulse.DriveChannel(self._physical_qubits[0]),
                )

            with pulse.build(backend=backend, name="xm") as xm:
                pulse.play(
                    pulse.Drag(
                        duration=self.experiment_options.duration,
                        amp=-self.experiment_options.amp,
                        sigma=self.experiment_options.sigma,
                        beta=beta,
                    ),
                    pulse.DriveChannel(self._physical_qubits[0]),
                )

        # Now check that we are dealing with Drag pulses.
        for schedule in [xp, xm]:
            for block in schedule.blocks:
                if isinstance(block, pulse.Play):
                    if isinstance(block.pulse, pulse.Drag):
                        break
            else:
                raise CalibrationError(f"No Drag pulse found in {schedule.name}.")

        beta_xp = next(iter(xp.parameters))
        beta_xm = next(iter(xm.parameters))

        if beta_xp != beta_xm:
            raise CalibrationError(
                f"Beta for xp and xm in {self.__class__.__name__} calibration are not identical."
            )

        xp_gate = Gate(name="xp", num_qubits=1, params=[beta_xp])
        xm_gate = Gate(name="xm", num_qubits=1, params=[beta_xp])

        reps = self.experiment_options.reps
        if len(reps) != 3:
            raise CalibrationError(
                f"The number of repetitions for {self.__class__.__name__} must be three. "
                "This constraint can be removed once CurveFitting supports a dynamic number "
                "of series."
            )

        circuits = []
        for beta in self.experiment_options.betas:

            beta = np.round(beta, decimals=6)

            for idx, rep in enumerate(reps):
                circuit = QuantumCircuit(1)
                for index in range(rep):
                    circuit.append(xp_gate, (0,))
                    circuit.barrier()
                    circuit.append(xm_gate, (0,))
                    if index != rep - 1:
                        circuit.barrier()

                circuit.measure_active()
                circuit.assign_parameters({beta_xp: beta}, inplace=True)

                xm_ = xm.assign_parameters({beta_xp: beta}, inplace=False)
                xp_ = xp.assign_parameters({beta_xp: beta}, inplace=False)

                circuit.add_calibration("xp", (self.physical_qubits[0],), xp_, params=[beta])
                circuit.add_calibration("xm", (self.physical_qubits[0],), xm_, params=[beta])

                circuit.metadata = {
                    "experiment_type": self._type,
                    "qubits": (self.physical_qubits[0],),
                    "xval": beta,
                    "series": idx,
                }

                circuits.append(circuit)

        return circuits
