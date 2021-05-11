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
from qiskit_experiments.analysis.plotting import plot_curve_fit, plot_scatter

try:
    from matplotlib import pyplot as plt

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


class SpectroscopyAnalysis(BaseAnalysis):
    """Class to analysis a spectroscopy experiment."""

    # pylint: disable=arguments-differ
    def _run_analysis(
        self,
        experiment_data,
        data_processor: Optional[callable] = None,
        meas_level: Optional[int] = MeasLevel.CLASSIFIED,
        amp_guess: Optional[float] = None,
        sigma_guesses: Optional[List[float]] = None,
        freq_guess: Optional[float] = None,
        offset_guess: Optional[float] = None,
        amplitude_bounds: Optional[List[float]] = None,
        sigma_bounds: Optional[List[float]] = None,
        freq_bounds: Optional[List[float]] = None,
        offset_bounds: Optional[List[float]] = None,
        plot: bool = True,
        ax: Optional["AxesSubplot"] = None,
    ) -> Tuple[AnalysisResult, None]:
        """
        Analyse a spectroscopy experiment by fitting the data to a Gaussian function.
        The fit function is:

        .. math::

            a * exp(-(x-x0)**2/(2*sigma**2)) + b

        Here, :math:`x` is the frequency. The analysis loops over the initial guesses
        of the width parameter :math:`sigma`.

        Args:
            experiment_data: The experiment data to analyze.
            data_processor: The data processor with which to process the data.
            meas_level: The measurement level of the experiment data.
            amp_guess: The amplitude of the Gaussian function, i.e. :math:`a`. If not
                provided, this will default to the maximum absolute value of the ydata.
            sigma_guesses: The guesses for the standard deviation of the Gaussian distribution.
                If it is not given this will default to an array of ten
                points linearly spaced between zero and the full width of the data.
            freq_guess: A guess for the frequency of the peak :math:`x0`. If not provided
                this guess will default to the location of the highest absolute data point.
            offset_guess: A guess for the magnitude :math:`b` offset of the fit function.
                If not provided, the initial guess defaults to the average of the ydata.
            amplitude_bounds: Bounds on the amplitude of the Gaussian function as a list of
                two floats. The default bounds are [0, 1.1*max(ydata)]
            sigma_bounds: Bounds on the standard deviation of the Gaussian function as a list
                of two floats. The default values are [0, frequency range].
            freq_bounds: Bounds on the center frequency as a list of two floats. The default
                values are 90% of the lower end of the frequency and 110% of the upper end of
                the frequency.
            offset_bounds: Bounds on the offset of the Gaussian function as a list of two floats.
                The default values are the minimum and maximum of the ydata.
            plot: If True generate a plot of fitted data.
            ax: Optional, matplotlib axis to add plot to.

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

        # Fitting will not work if any sigmas are exactly 0.
        if any(sigmas == 0.0):
            sigmas = None

        if not offset_guess:
            offset_guess = np.average(ydata)
        if not amp_guess:
            amp_guess = np.max(ydata)
        if not freq_guess:
            peak_idx = np.argmax(ydata)
            freq_guess = xdata[peak_idx]
        if not sigma_guesses:
            sigma_guesses = np.linspace(1e-6, abs(xdata[-1] - xdata[0]), 20)
        if amplitude_bounds is None:
            amplitude_bounds = [0.0, 1.1 * max(ydata)]
        if sigma_bounds is None:
            sigma_bounds = [0, abs(xdata[-1] - xdata[0])]
        if freq_bounds is None:
            freq_bounds = [0.9 * xdata[0], 1.1 * xdata[-1]]
        if offset_bounds is None:
            offset_bounds = [np.min(ydata), np.max(ydata)]

        best_fit = None

        lower = np.array([amplitude_bounds[0], sigma_bounds[0], freq_bounds[0], offset_bounds[0]])
        upper = np.array([amplitude_bounds[1], sigma_bounds[1], freq_bounds[1], offset_bounds[1]])

        # Perform fit
        def fit_fun(x, a, sigma, x0_, b):
            return a * np.exp(-((x - x0_) ** 2) / (2 * sigma ** 2)) + b

        for sigma_guess in sigma_guesses:

            fit_result = curve_fit(
                fit_fun,
                xdata,
                ydata,
                np.array([amp_guess, sigma_guess, freq_guess, offset_guess]),
                sigmas,
                (lower, upper),
            )

            if not best_fit:
                best_fit = fit_result
            else:
                if fit_result["reduced_chisq"] < best_fit["reduced_chisq"]:
                    best_fit = fit_result

        best_fit["value"] = best_fit["popt"][2]
        best_fit["stderr"] = (best_fit["popt_err"][2],)
        best_fit["unit"] = (experiment_data.data[0]["metadata"].get("unit", "Hz"),)
        best_fit["label"] = "Spectroscopy"
        best_fit["xdata"] = xdata
        best_fit["ydata"] = ydata
        best_fit["ydata_err"] = sigmas
        best_fit["quality"] = self._fit_quality(
            best_fit["popt"],
            best_fit["popt_err"],
            best_fit["reduced_chisq"],
            xdata[0],
            xdata[-1],
        )

        if plot:
            ax = plot_curve_fit(fit_fun, best_fit, ax=ax)
            ax = plot_scatter(xdata, ydata, ax=ax)
            self._format_plot(ax, best_fit)
            best_fit.plt = plt

        return best_fit, None

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

    @classmethod
    def _format_plot(cls, ax, analysis_result):
        """Format curve fit plot."""
        ax.tick_params(labelsize=14)
        ax.set_xlabel(f"Frequency shift {analysis_result['unit']}", fontsize=16)
        ax.set_ylabel("Signal [arb. unit.]", fontsize=16)
        ax.grid(True)


class Spectroscopy(BaseExperiment):
    """Class the runs spectroscopy by sweeping the qubit frequency."""

    __analysis_class__ = SpectroscopyAnalysis

    # Supported units for spectroscopy.
    __units__ = {"Hz": 1.0, "kHz": 1.0e3, "MHz": 1.0e6, "GHz": 1.0e9}

    # default run options
    __run_defaults__ = {"meas_level": MeasLevel.CLASSIFIED}

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
