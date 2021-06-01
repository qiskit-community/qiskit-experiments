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
from qiskit.circuit import Gate, Parameter
from qiskit.exceptions import QiskitError
from qiskit.providers import Backend
import qiskit.pulse as pulse
from qiskit.qobj.utils import MeasLevel
from qiskit.providers.options import Options

from qiskit_experiments.analysis.curve_fitting import curve_fit
from qiskit_experiments.base_analysis import BaseAnalysis
from qiskit_experiments.base_experiment import BaseExperiment
from qiskit_experiments import AnalysisResult
from qiskit_experiments import ExperimentData
from qiskit_experiments.data_processing.processor_library import get_to_signal_processor
from qiskit_experiments.analysis import plotting


class SpectroscopyAnalysis(BaseAnalysis):
    """A class to analyze a spectroscopy experiment.

    Analyze a spectroscopy experiment by fitting the data to a Gaussian function.
    The fit function is:

    .. math::

        a * exp(-(x-x0)**2/(2*sigma**2)) + b

    Here, :math:`x` is the frequency. The analysis loops over the initial guesses
    of the width parameter :math:`sigma`. The measured y-data will be rescaled to
    the interval (0,1).

    Analysis options:

        * amp_guess (float): The amplitude of the Gaussian function, i.e. :math:`a`. If not
            provided, this will default to -1 or 1 depending on the measured values.
        * sigma_guesses (list of float): The guesses for the standard deviation of the Gaussian
            distribution. If it is not given this will default to an array of ten  points linearly
            spaced between zero and width of the x-data.
        * freq_guess (float): A guess for the frequency of the peak :math:`x0`. If not provided
            this guess will default to the location of the highest or lowest point of the y-data
            depending on the y-data.
        * offset_guess (float): A guess for the magnitude :math:`b` offset of the fit function. If
            not provided, the initial guess defaults to the median of the y-data.
        * amp_bounds (tuple of two floats): Bounds on the amplitude of the Gaussian function as a
            tuple of two floats. The default bounds are (-1, 1).
        * sigma_bounds (tuple of two floats): Bounds on the standard deviation of the Gaussian
            function as a tuple of two floats. The default values are (0, frequency range).
        * freq_bounds (tuple of two floats): Bounds on the center frequency as a tuple of two
            floats. The default values are (min(frequencies) - df, max(frequencies) - df).
        * offset_bounds (tuple of two floats): Bounds on the offset of the Gaussian function as a
            tuple of two floats. The default values are (-2, 2).
    """

    @classmethod
    def _default_options(cls):
        return Options(
            meas_level=MeasLevel.KERNELED,
            meas_return="single",
            amp_guess=None,
            sigma_guesses=None,
            freq_guess=None,
            offset_guess=None,
            amp_bounds=(-1, 1),
            sigma_bounds=None,
            freq_bounds=None,
            offset_bounds=(-2, 2),
        )

    # pylint: disable=arguments-differ, unused-argument
    def _run_analysis(
        self,
        experiment_data: ExperimentData,
        data_processor: Optional[callable] = None,
        amp_guess: Optional[float] = None,
        sigma_guesses: Optional[List[float]] = None,
        freq_guess: Optional[float] = None,
        offset_guess: Optional[float] = None,
        amp_bounds: Tuple[float, float] = (-1, 1),
        sigma_bounds: Optional[Tuple[float, float]] = None,
        freq_bounds: Optional[Tuple[float, float]] = None,
        offset_bounds: Tuple[float, float] = (-2, 2),
        plot: bool = True,
        ax: Optional["AxesSubplot"] = None,
        **kwargs,
    ) -> Tuple[AnalysisResult, None]:
        """Analyze the given data by fitting it to a Gaussian.

        Args:
            experiment_data: The experiment data to analyze.
            data_processor: The data processor with which to process the data. If no data
                processor is given a singular value decomposition of the IQ data will be
                used for Kerneled data and a conversion from counts to probabilities will
                be done if Discriminated data was measured.
            amp_guess: The amplitude of the Gaussian function, i.e. :math:`a`. If not
                provided, this will default to -1 or 1 depending on the measured values.
            sigma_guesses: The guesses for the standard deviation of the Gaussian distribution.
                If it is not given this will default to an array of ten
                points linearly spaced between zero and width of the x-data.
            freq_guess: A guess for the frequency of the peak :math:`x0`. If not provided
                this guess will default to the location of the highest or lowest point of
                the y-data depending on the y-data.
            offset_guess: A guess for the magnitude :math:`b` offset of the fit function.
                If not provided, the initial guess defaults to the median of the y-data.
            amp_bounds: Bounds on the amplitude of the Gaussian function as a tuple of
                two floats. The default bounds are (-1, 1).
            sigma_bounds: Bounds on the standard deviation of the Gaussian function as a tuple
                of two floats. The default values are (0, frequency range).
            freq_bounds: Bounds on the center frequency as a tuple of two floats. The default
                values are (min(frequencies) - df, max(frequencies) - df).
            offset_bounds: Bounds on the offset of the Gaussian function as a tuple of two floats.
                The default values are (-2, 2).
            plot: If True generate a plot of fitted data.
            ax: Optional, matplotlib axis to add the plot to.
            kwargs: Trailing unused function parameters.

        Returns:
            The analysis result with the estimated peak frequency and the plots if a plot was
            generated.

        Raises:
            QiskitError:
                - If the measurement level is not supported.
                - If the fit fails.
        """

        meas_level = experiment_data.data(0)["metadata"]["meas_level"]
        meas_return = experiment_data.data(0)["metadata"]["meas_return"]

        # Pick a data processor.
        if data_processor is None:
            data_processor = get_to_signal_processor(meas_level=meas_level, meas_return=meas_return)
            data_processor.train(experiment_data.data())

        y_sigmas = np.array([data_processor(datum) for datum in experiment_data.data()])
        min_y, max_y = min(y_sigmas[:, 0]), max(y_sigmas[:, 0])
        ydata = (y_sigmas[:, 0] - min_y) / (max_y - min_y)

        # Sigmas may be None and fitting will not work if any sigmas are exactly 0.
        try:
            sigmas = y_sigmas[:, 1] / (max_y - min_y)
            if any(sigmas == 0.0):
                sigmas = None

        except TypeError:
            sigmas = None

        xdata = np.array([datum["metadata"]["xval"] for datum in experiment_data.data()])

        # Set the default options that depend on the y-data.
        if not offset_guess:
            offset_guess = np.median(ydata)
        if not amp_guess:
            amp_guess = -1 if offset_guess > 0.5 else 1
        if not freq_guess:
            peak_idx = np.argmin(ydata) if offset_guess > 0.5 else np.argmax(ydata)
            freq_guess = xdata[peak_idx]
        if not sigma_guesses:
            sigma_guesses = np.linspace(1e-6, abs(xdata[-1] - xdata[0]), 20)
        if sigma_bounds is None:
            sigma_bounds = (0, abs(xdata[-1] - xdata[0]))
        if freq_bounds is None:
            dx = xdata[1] - xdata[0]
            freq_bounds = (xdata[0] - dx, xdata[-1] + dx)

        # Perform fit
        best_fit = None
        bounds = {"a": amp_bounds, "sigma": sigma_bounds, "freq": freq_bounds, "b": offset_bounds}

        def fit_fun(x, a, sigma, freq, b):
            return a * np.exp(-((x - freq) ** 2) / (2 * sigma ** 2)) + b

        for sigma_guess in sigma_guesses:
            init = {"a": amp_guess, "sigma": sigma_guess, "freq": freq_guess, "b": offset_guess}
            try:
                fit_result = curve_fit(fit_fun, xdata, ydata, init, sigmas, bounds)

                if not best_fit:
                    best_fit = fit_result
                else:
                    if fit_result["reduced_chisq"] < best_fit["reduced_chisq"]:
                        best_fit = fit_result

            except RuntimeError:
                pass

        if best_fit is None:
            raise QiskitError("Could not find a fit to the spectroscopy data.")

        best_fit["value"] = best_fit["popt"][2]
        best_fit["stderr"] = (best_fit["popt_err"][2],)
        best_fit["unit"] = experiment_data.data(0)["metadata"].get("unit", "Hz")
        best_fit["label"] = "Spectroscopy"
        best_fit["xdata"] = xdata
        best_fit["ydata"] = ydata
        best_fit["ydata_err"] = sigmas
        best_fit["quality"] = self._fit_quality(
            best_fit["popt"][0],
            best_fit["popt"][1],
            best_fit["popt"][2],
            best_fit["popt"][3],
            best_fit["reduced_chisq"],
            xdata,
            ydata,
            best_fit["popt_err"][1],
        )

        if plot and plotting.HAS_MATPLOTLIB:
            ax = plotting.plot_curve_fit(fit_fun, best_fit, ax=ax)
            ax = plotting.plot_scatter(xdata, ydata, ax=ax)
            self._format_plot(ax, best_fit)
            figures = [ax.get_figure()]
        else:
            figures = None

        return best_fit, figures

    @staticmethod
    def _fit_quality(
        fit_amp: float,
        fit_sigma: float,
        fit_freq: float,
        fit_offset: float,
        reduced_chisq: float,
        xdata: np.array,
        ydata: np.array,
        fit_sigma_err: Optional[float] = None,
    ) -> str:
        """Algorithmic criteria for whether the fit is good or bad.

        A good fit has:
            - a reduced chi-squared less than 3,
            - a peak within the scanned frequency range,
            - a standard deviation that is not larger than the scanned frequency range,
            - a standard deviation that is wider than the smallest frequency increment,
            - a signal-to-noise ratio, defined as the amplitude of the peak divided by the
              square root of the median y-value less the fit offset, greater than a
              threshold of two, and
            - a standard error on the sigma of the Gaussian that is smaller than the sigma.

        Args:
            fit_amp: Amplitude of the fitted peak.
            fit_sigma: Standard deviation of the fitted Gaussian.
            fit_freq: Frequency of the fitted peak.
            fit_offset: Offset of the fit.
            reduced_chisq: Reduced chi-squared of the fit.
            xdata: x-values, i.e. the frequencies.
            ydata: y-values, i.e. the measured signal.
            fit_sigma_err: Errors on the standard deviation of the fit.

        Returns:
            computer_bad or computer_good if the fit passes or fails the criteria, respectively.
        """
        min_freq = xdata[0]
        max_freq = xdata[-1]
        freq_increment = xdata[1] - xdata[0]

        snr = abs(fit_amp) / np.sqrt(abs(np.median(ydata) - fit_offset))
        fit_width_ratio = fit_sigma / (max_freq - min_freq)

        # pylint: disable=too-many-boolean-expressions
        if (
            min_freq <= fit_freq <= max_freq
            and 1.5 * freq_increment < fit_sigma
            and fit_width_ratio < 0.25
            and reduced_chisq < 3
            and (fit_sigma_err is None or (fit_sigma_err < fit_sigma))
            and snr > 2
        ):
            return "computer_good"
        else:
            return "computer_bad"

    @classmethod
    def _format_plot(cls, ax, analysis_result):
        """Format curve fit plot."""
        ax.tick_params(labelsize=14)
        ax.set_xlabel(f"Frequency ({analysis_result['unit']})", fontsize=16)
        ax.set_ylabel("Signal [arb. unit.]", fontsize=16)
        ax.grid(True)


class QubitSpectroscopy(BaseExperiment):
    """Class that runs spectroscopy by sweeping the qubit frequency.

    The circuits produced by spectroscopy, i.e.

    .. parsed-literal::

                   ┌────────────┐ ░ ┌─┐
              q_0: ┤ Spec(freq) ├─░─┤M├
                   └────────────┘ ░ └╥┘
        measure: 1/══════════════════╩═
                                     0

    have a spectroscopy pulse-schedule embedded in a spectroscopy gate. The
    pulse-schedule consists of a set frequency instruction followed by a GaussianSquare
    pulse. A list of circuits is generated, each with a different frequency "freq".
    """

    __analysis_class__ = SpectroscopyAnalysis

    # Supported units for spectroscopy.
    __units__ = {"Hz": 1.0, "kHz": 1.0e3, "MHz": 1.0e6, "GHz": 1.0e9}

    @classmethod
    def _default_run_options(cls) -> Options:
        """Default options values for the experiment :meth:`run` method."""
        return Options(
            meas_level=MeasLevel.KERNELED,
            meas_return="single",
        )

    @classmethod
    def _default_experiment_options(cls) -> Options:
        """Default option values used for the spectroscopy pulse."""
        return Options(
            amp=0.1,
            duration=1024,
            sigma=256,
            width=0,
        )

    def __init__(
        self,
        qubit: int,
        frequencies: Union[List[float], np.array],
        unit: Optional[str] = "Hz",
        absolute: bool = True,
    ):
        """
        A spectroscopy experiment run by setting the frequency of the qubit drive.
        The parameters of the GaussianSquare spectroscopy pulse can be specified at run-time.
        The spectroscopy pulse has the following parameters:
        - amp: The amplitude of the pulse must be between 0 and 1, the default is 0.1.
        - duration: The duration of the spectroscopy pulse in samples, the default is 1000 samples.
        - sigma: The standard deviation of the pulse, the default is duration / 4.
        - width: The width of the flat-top in the pulse, the default is 0, i.e. a Gaussian.

        Args:
            qubit: The qubit on which to run spectroscopy.
            frequencies: The frequencies to scan in the experiment.
            unit: The unit in which the user specifies the frequencies. Can be one
                of 'Hz', 'kHz', 'MHz', 'GHz'. Internally, all frequencies will be converted
                to 'Hz'.
            absolute: Boolean to specify if the frequencies are absolute or relative to the
                qubit frequency in the backend.

        Raises:
            QiskitError: if there are less than three frequency shifts or if the unit is not known.

        """
        if len(frequencies) < 3:
            raise QiskitError("Spectroscopy requires at least three frequencies.")

        if unit not in self.__units__:
            raise QiskitError(f"Unsupported unit: {unit}.")

        self._frequencies = [freq * self.__units__[unit] for freq in frequencies]
        self._absolute = absolute

        super().__init__([qubit])

    def circuits(self, backend: Optional[Backend] = None):
        """Create the circuit for the spectroscopy experiment.

        The circuits are based on a GaussianSquare pulse and a frequency_shift instruction
        encapsulated in a gate.

        Args:
            backend: A backend object.

        Returns:
            circuits: The circuits that will run the spectroscopy experiment.

        Raises:
            QiskitError:
                - If relative frequencies are used but no backend was given.
                - If the backend configuration does not define dt.
        """
        if not backend and not self._absolute:
            raise QiskitError("Cannot run spectroscopy relative to qubit without a backend.")

        # Create a template circuit
        freq_param = Parameter("frequency")
        with pulse.build(backend=backend, name="spectroscopy") as sched:
            pulse.set_frequency(freq_param, pulse.DriveChannel(self.physical_qubits[0]))
            pulse.play(
                pulse.GaussianSquare(
                    duration=self.experiment_options.duration,
                    amp=self.experiment_options.amp,
                    sigma=self.experiment_options.sigma,
                    width=self.experiment_options.width,
                ),
                pulse.DriveChannel(self.physical_qubits[0]),
            )

        gate = Gate(name="Spec", num_qubits=1, params=[freq_param])

        circuit = QuantumCircuit(1)
        circuit.append(gate, (0,))
        circuit.add_calibration(gate, (self.physical_qubits[0],), sched, params=[freq_param])
        circuit.measure_active()

        if not self._absolute:
            center_freq = backend.defaults().qubit_freq_est[self.physical_qubits[0]]

        # Create the circuits to run
        circs = []
        for freq in self._frequencies:
            if not self._absolute:
                freq += center_freq

            assigned_circ = circuit.assign_parameters({freq_param: freq}, inplace=False)
            assigned_circ.metadata = {
                "experiment_type": self._type,
                "qubit": self.physical_qubits[0],
                "xval": freq,
                "unit": "Hz",
                "amplitude": self.experiment_options.amp,
                "duration": self.experiment_options.duration,
                "sigma": self.experiment_options.sigma,
                "width": self.experiment_options.width,
                "schedule": str(sched),
                "meas_level": self.run_options.meas_level,
                "meas_return": self.run_options.meas_return,
            }

            if not self._absolute:
                assigned_circ.metadata["center frequency"] = center_freq

            try:
                assigned_circ.metadata["dt"] = getattr(backend.configuration(), "dt")
            except AttributeError as no_dt:
                raise QiskitError("Dt parameter is missing in backend configuration") from no_dt

            circs.append(assigned_circ)

        return circs
