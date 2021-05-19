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

"""Rabi amplitude Experiment class."""

from typing import Callable, List, Optional, Tuple, Union
import numpy as np

from qiskit import QiskitError
from qiskit.circuit import Parameter
from qiskit.qobj.utils import MeasLevel
from qiskit.providers import Backend
from qiskit.pulse import ScheduleBlock
import qiskit.pulse as pulse

from qiskit import QuantumCircuit
from qiskit_experiments import BaseAnalysis, BaseExperiment, ExperimentData, AnalysisResult
from qiskit_experiments.analysis.curve_fitting import curve_fit
from qiskit_experiments.data_processing.nodes import Probability#, SVD
from qiskit_experiments.data_processing.data_processor import DataProcessor
from qiskit_experiments.analysis import plotting

try:
    from matplotlib import pyplot as plt

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


class RabiAnalysis(BaseAnalysis):
    """Rabi analysis class based on a fit to a cosine function.

    Analyse a Rabi experiment by fitting it to a cosine function

    .. math::
        a * \cos(b*x) + c

    The y-values will be normalized to the range 0-1.
    """

    def _run_analysis(
        self,
        experiment_data: ExperimentData,
        data_processor: Optional[Callable] = None,
        meas_level: Optional[int] = MeasLevel.KERNELED,
        amp_guess: float = 0.5,
        freq_guess: float = np.pi,
        offset_guess: float = 0.5,
        amp_bounds: Tuple[float, float] = (-1, 1),
        freq_bounds: Tuple[float, float] = (0, np.inf),
        offset_bounds: Tuple[float, float] = (0, 1),
        plot: bool = True,
        ax: Optional["AxesSubplot"] = None,
        **options
    ) -> Tuple[AnalysisResult, List["plotting.pyplot.Figure"]]:
        """Fit the data to an oscillating function.

        Args:
            experiment_data: The experiment data to fit.
            data_processor: A data processor with which to analyse the data. If None is given
                a SVD-based data processor will be used for kerneled data while a conversion
                from counts to probabilities will be used for discriminated data.
            meas_level: The measurement level used.
            amp_guess: The amplitude guess for the fit which will default to 0.5.
            freq_guess: The frequency guess for the fit which defaults to pi.
            offset_guess: The y-axis offset which defaults to 0.5.
            amp_bounds: Bounds on the amplitude which default to (-1, 1).
            freq_bounds: Bounds on the frequency which default to (0, inf).
            offset_bounds: Bounds on the offset which default to (0,1).
            plot: If True generate a plot of fitted data.
            ax: Optional, matplotlib axis to add plot to.
            kwargs: Trailing unused function parameters.

        Returns:
            The analysis result with the fit and optional plots.
        """

        # Pick a data processor.
        if data_processor is None:
            if meas_level == MeasLevel.CLASSIFIED:
                data_processor = DataProcessor("counts", [Probability("1")])
            elif meas_level == MeasLevel.KERNELED:
                raise NotImplementedError
                #data_processor = DataProcessor("memory", [SVD()])
                data_processor.train(experiment_data.data())
            else:
                raise ValueError("Unsupported measurement level.")

        y_sigmas = np.array([data_processor(datum) for datum in experiment_data.data()])
        y_max, y_min = max(y_sigmas[:, 0]), min(y_sigmas[:, 0])
        sigmas = np.sqrt(y_sigmas[:, 1]) / (y_max - y_min)
        ydata = (y_sigmas[:, 0] - y_min) / (y_max - y_min)
        xdata = np.array([datum["metadata"]["xval"] for datum in experiment_data.data()])

        lower = np.array([amp_bounds[0], freq_bounds[0], offset_bounds[0]])
        upper = np.array([amp_bounds[1], freq_bounds[1], offset_bounds[1]])

        # Perform fit
        def fit_fun(x, a, b, c):
            return a * np.cos(b*x) + c

        fit_result = curve_fit(
            fit_fun,
            xdata,
            ydata,
            np.array([amp_guess, freq_guess, offset_guess]),
            sigmas,
            (lower, upper),
        )

        fit_result["value"] = fit_result["popt"][2]
        fit_result["stderr"] = (fit_result["popt_err"][2],)
        fit_result["label"] = "Spectroscopy"
        fit_result["xdata"] = xdata
        fit_result["ydata"] = ydata
        fit_result["ydata_err"] = sigmas
        fit_result["quality"] = self._fit_quality(
            fit_result["popt"][1],
            fit_result["reduced_chisq"]
        )

        if plot and HAS_MATPLOTLIB:
            ax = plotting.plot_curve_fit(fit_fun, fit_result, ax=ax)
            ax = plotting.plot_scatter(xdata, ydata, ax=ax)
            self._format_plot(ax)
            figures = [ax.get_figure()]
        else:
            figures = None

        return fit_result, figures

    @staticmethod
    def _fit_quality(fit_freq: float, reduced_chisq: float):
        """Method to check the quality of the fit."""

        if reduced_chisq < 5 and fit_freq > np.pi:
            return "computer_good"

        return "computer_bad"

    @classmethod
    def _format_plot(cls, ax):
        """Format curve fit plot."""
        ax.tick_params(labelsize=14)
        ax.set_xlabel(f"Amplitude [arb. unit]", fontsize=16)
        ax.set_ylabel("Signal [arb. unit.]", fontsize=16)
        ax.grid(True)

class Rabi(BaseExperiment):
    """A calibration experiment that scans the amplitude of a pulse."""

    __analysis_class__ = RabiAnalysis

    __run_defaults__ = {"meas_level": MeasLevel.KERNELED}

    def __init__(self, qubit: int, amplitudes: Optional[Union[List[float], np.array]] = None):
        """Setup a Rabi experiment on the given qubit.

        Args:
            qubit: The qubit on which to run the Rabi experiment.
            amplitudes: The amplitudes that will be scanned.
        """
        super().__init__([qubit])

        if amplitudes:
            self._amplitudes = amplitudes
        else:
            self._amplitudes = np.linspace(-0.95, 0.95, 51)

    def circuits(
        self,
        backend: Optional[Backend] = None,
        schedule: Optional[ScheduleBlock] = None,
        **circuit_options
    ) -> List[QuantumCircuit]:
        """Create the circuits for the Rabi experiment.

        Args:
            backend: A backend object.
            schedule: The schedule for which to scan the amplitude. This schedule must have
                one parameter that will be scanned.
            circuit_options: Circuit options that may include "amplitudes", i.e. the run-time
                given amplitudes that will override those given be the init method.

        Returns:
            A ist of circuits with a rx rotation with a calibration whose amplitude is scanned.
        """
        if schedule is None:
            amp = Parameter("amp")
            with pulse.build() as schedule:
                pulse.play(
                    pulse.Gaussian(duration=160, amp=amp, sigma=40),
                    pulse.DriveChannel(self.physical_qubits[0])
                )

        if len(schedule.parameters) != 1:
            raise QiskitError("Schedule in Rabi must have exactly one free parameter.")

        param = next(iter(schedule.parameters))

        circuit = QuantumCircuit(1)
        circuit.rx(param, 0)
        circuit.measure_active()

        circs = []
        for amp in circuit_options.get("amplitudes", self._amplitudes):
            assigned_circ = circuit.assign_parameters({param: amp}, inplace=False)
            assigned_circ.metadata = {
                "experiment_type": self._type,
                "qubit": self.physical_qubits[0],
                "xval": amp,
                "unit": "Hz",
                "amplitude": amp,
                "schedule": str(schedule),
            }

            if backend:
                assigned_circ.metadata["dt"] = getattr(backend.configuration(), "dt", "n.a.")

            circs.append(assigned_circ)

        return circs
