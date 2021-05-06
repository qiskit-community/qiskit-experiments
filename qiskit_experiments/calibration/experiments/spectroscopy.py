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

"""Spectroscopy experiment class."""

from typing import List, Optional, Tuple, Union
import numpy as np

from qiskit import QuantumCircuit
from qiskit.circuit import Gate
import qiskit.pulse as pulse
from qiskit.qobj.utils import MeasLevel
from qiskit_experiments.analysis.curve_fitting import curve_fit
from qiskit_experiments.base_analysis import BaseAnalysis
from qiskit_experiments.base_experiment import BaseExperiment
from qiskit_experiments import AnalysisResult
from qiskit_experiments.data_processing.data_processor import DataProcessor
from qiskit_experiments.data_processing.nodes import ToReal
from qiskit_experiments.data_processing.nodes import Probability


class SpectroscopyAnalysis(BaseAnalysis):
    """Class to analysis a spectroscopy experiment."""

    # pylint: disable=arguments-differ
    def _run_analysis(
        self,
        experiment_data,
        data_processor=None,
        meas_level=MeasLevel.KERNELED,
        amp_guess: float = None,
        gamma_guesses: List[float] = None,
        freq_guess: float = None,
        offset_guess: float = None,
        amplitude_bounds: List[float] = None,
        width_bounds: List[float] = None,
        freq_bounds: List[float] = None,
        offset_bounds: List[float] = None,
    ) -> Tuple[AnalysisResult, None]:
        """
        Analyse a spectroscopy experiment by fitting the data to a Lorentz function.
        The fit function is:

        .. math::

            a * ( g**2 / ((x-x0)**2 + g**2)) + b

        Here, :math:`x` is the frequency. The analysis loops over the initial guesses
        of the width parameter :math:`g`.

        Args:
            experiment_data: The experiment data to analyze.
            data_processor: The data processor with which to process the data.
            meas_level: The measurement level of the experiment data.
            amp_guess: The amplitude of the Lorentz function, i.e. :math:`a`. If not
                provided, this will default to the maximum absolute value of the ydata.
            gamma_guesses: The guesses for the width parameter of the Lorentz distribution,
                i.e. :math:`g`. If it is not given this will default to an array of ten
                points linearly spaced between zero and the full width of the data.
            freq_guess: A guess for the frequency of the peak :math:`x0`. If not provided
                this guess will default to the location of the highest absolute data point.
            offset_guess: A guess for the magnitude :math:`b` offset of the fit function.
                If not provided, the initial guess defaults to the average of the ydata.
            amplitude_bounds: Bounds on the amplitude of the Lorentz function as a list of
                two floats. The default bounds are [0, 1.1*max(ydata)]
            width_bounds: Bounds on the width of the Lorentz function as a list of two floats.
                The default values are [0, frequency range].
            freq_bounds: Bounds on the center frequency as a list of two floats. The default
                values are 90% of the lower end of the frequency and 110% of the upper end of
                the frequency.
            offset_bounds: Bounds on the offset of the Lorentz function as a list of two floats.
                The default values are the minimum and maximum of the ydata.

        Returns:
            The analysis result with the estimated peak frequency.

        Raises:
            ValueError: If the measurement level is not supported.
        """

        # Pick a data processor.
        if data_processor is None:
            if meas_level == MeasLevel.CLASSIFIED:
                data_processor = DataProcessor("counts", [Probability("1")])
            elif meas_level == MeasLevel.KERNELED:
                data_processor = DataProcessor("memory", [ToReal()])
            else:
                raise ValueError("Unsupported measurement level.")

        y_sigmas = np.array([data_processor(datum) for datum in experiment_data.data])
        sigmas = y_sigmas[:, 1]
        ydata = abs(y_sigmas[:, 0])
        xdata = np.array([datum["metadata"]["xval"] for datum in experiment_data.data])

        if not offset_guess:
            offset_guess = np.average(ydata)
        if not amp_guess:
            amp_guess = np.max(ydata)
        if not freq_guess:
            peak_idx = np.argmax(ydata)
            freq_guess = xdata[peak_idx]
        if not gamma_guesses:
            gamma_guesses = np.linspace(0, abs(xdata[-1] - xdata[0]), 10)
        if amplitude_bounds is None:
            amplitude_bounds = [0.0, 1.1 * max(ydata)]
        if width_bounds is None:
            width_bounds = [0, abs(xdata[-1] - xdata[0])]
        if freq_bounds is None:
            freq_bounds = [0.9 * xdata[0], 1.1 * xdata[-1]]
        if offset_bounds is None:
            offset_bounds = [np.min(ydata), np.max(ydata)]

        best_fit = None

        lower = np.array([amplitude_bounds[0], width_bounds[0], freq_bounds[0], offset_bounds[0]])
        upper = np.array([amplitude_bounds[1], width_bounds[1], freq_bounds[1], offset_bounds[1]])

        for gamma_guess in gamma_guesses:
            fit_result = curve_fit(
                lambda x, a, g, x0, b: a * (g ** 2 / ((x - x0) ** 2 + g ** 2)) + b,
                xdata,
                np.array(ydata),
                np.array([amp_guess, gamma_guess, freq_guess, offset_guess]),
                np.array(sigmas),
                (lower, upper),
            )

            if not best_fit:
                best_fit = fit_result
            else:
                if fit_result["reduced_chisq"] < best_fit["reduced_chisq"]:
                    best_fit = fit_result

        analysis_result = AnalysisResult(
            {
                "value": best_fit["popt"][2],
                "stderr": best_fit["popt_err"][2],
                "unit": experiment_data.data[0]["metadata"].get("unit", "Hz"),
                "label": "Spectroscopy",
                "fit": best_fit,
                "quality": self._fit_quality(
                    best_fit["popt"],
                    best_fit["popt_err"],
                    best_fit["reduced_chisq"],
                    xdata[0],
                    xdata[-1],
                ),
            }
        )

        return analysis_result, None

    @staticmethod
    def _fit_quality(fit_out, fit_err, reduced_chisq, min_freq, max_freq) -> str:
        """
        Algorithmic criteria for whether the fit is good or bad.
        A good fit has a small reduced chi-squared and the peak must be
        within the scanned frequency range.

        Args:
            fit_out: Value of the fit.
            fit_err: Errors on the fit value.
            reduced_chisq: Reduced chi-squared of the fit.
            min_freq: Minimum frequency in the spectroscopy.
            max_freq: Maximum frequency in the spectroscopy.

        Returns:
            computer_bad or computer_good if the fit passes or fails, respectively.
        """

        if (
            min_freq <= fit_out[2] <= max_freq
            and fit_out[1] < (max_freq - min_freq)
            and reduced_chisq < 3
            and (fit_err[2] is None or fit_err[2] < fit_out[2])
        ):
            return "computer_good"
        else:
            return "computer_bad"


class Spectroscopy(BaseExperiment):
    """Class the runs spectroscopy by sweeping the qubit frequency."""

    __analysis_class__ = SpectroscopyAnalysis

    # Supported units for spectroscopy.
    __units__ = {"Hz": 1.0, "kHz": 1.0e3, "MHz": 1.0e6, "GHz": 1.0e9}

    # default run options
    __run_defaults__ = {"meas_level": MeasLevel.KERNELED}

    def __init__(
        self, qubit: int, frequency_shifts: Union[List[float], np.array], unit: Optional[str] = "Hz"
    ):
        """
        A spectroscopy experiment run by shifting the frequency of the qubit.
        The parameters of the GaussianSquare spectroscopy pulse are specified at run-time.
        The spectroscopy pulse has the following parameters:
        - amp: The amplitude of the pulse must be between 0 and 1, the default is 0.1.
        - duration: The duration of the spectroscopy pulse in samples, the default is 1000 samples.
        - sigma: The standard deviation of the pulse, the default is 5 x duration.
        - width: The width of the flat-top in the pulse, the default is 0, i.e. a Gaussian.

        Args:
            qubit: The qubit on which to run spectroscopy.
            frequency_shifts: The frequencies to scan in the experiment.
            unit: The unit in which the user specifies the frequencies. Can be one
                of 'Hz', 'kHz', 'MHz', 'GHz'. Internally, all frequencies will be converted
                to 'Hz'.

        Raises:
            ValueError: if there are less than three frequency shifts or if the unit is not known.

        """
        if len(frequency_shifts) < 3:
            raise ValueError("Spectroscopy requires at least three frequencies.")

        if unit not in self.__units__:
            raise ValueError(f"Unsupported unit: {unit}.")

        self._frequency_shifts = [freq * self.__units__[unit] for freq in frequency_shifts]

        super().__init__([qubit], circuit_options=("amp", "duration", "sigma", "width"))

    # pylint: disable=unused-argument
    def circuits(self, backend: Optional["Backend"] = None, **circuit_options):
        """
        Create the circuit for the spectroscopy experiment. The circuits are based on a
        GaussianSquare pulse and a frequency_shift instruction encapsulated in a gate.

        Args:
            backend: A backend object.
            circuit_options: Key word arguments to run the circuits. The circuit options are
                - amp: The amplitude of the GaussianSquare pulse, defaults to 0.1.
                - duration: The duration of the GaussianSquare pulse, defaults to 10240.
                - sigma: The standard deviation of the GaussianSquare pulse, defaults to one
                    fith of the duration.
                - width: The width of the flat top in the GaussianSquare pulse, defaults to 0.

        Returns:
            circuits: The circuits that will run the spectroscopy experiment.
        """

        amp = circuit_options.get("amp", 0.1)
        duration = circuit_options.get("duration", 1024)
        sigma = circuit_options.get("sigma", duration / 5)
        width = circuit_options.get("width", 0)

        drive = pulse.DriveChannel(self._physical_qubits[0])

        circs = []

        for freq_shift in self._frequency_shifts:
            with pulse.build(name=f"Frequency shift{freq_shift}") as sched:
                pulse.shift_frequency(freq_shift, drive)
                pulse.play(pulse.GaussianSquare(duration, amp, sigma, width), drive)

            gate = Gate(name="Spec", num_qubits=1, params=[])

            circuit = QuantumCircuit(1)
            circuit.append(gate, (0,))
            circuit.add_calibration(gate, (self._physical_qubits[0],), sched)
            circuit.measure_active()

            circuit.metadata = {
                "experiment_type": self._type,
                "qubit": self._physical_qubits[0],
                "xval": freq_shift,
                "unit": "Hz",
            }

            circs.append(circuit)

        return circs
