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
    get_opt_value,
    get_opt_error,
)
from qiskit_experiments.base_experiment import BaseExperiment
from qiskit_experiments.data_processing.processor_library import get_to_signal_processor
from qiskit_experiments.calibration.exceptions import CalibrationError


class RoughDragAnalysis(CurveAnalysis):
    r"""Rough Drag calibration analysis class based on a fit to a cosine function.

    Analyse a Drag clibration experiment by fitting it to a cosine function

    .. math::
        y = a \cos\left(2 \pi {\rm freq} x + {\rm phase}\right) + b

    Fit Parameters TODO
        - :math:`a`: Amplitude of the oscillation.
        - :math:`b`: Base line.
        - :math:`{\rm freq}`: Frequency of the oscillation. This is the fit parameter of interest.
        - :math:`{\rm phase}`: Phase of the oscillation.

    Initial Guesses
        - :math:`a`: The maximum y value less the minimum y value.
        - :math:`b`: The average of the data.
        - :math:`{\rm freq}`: The frequency with the highest power spectral density.
        - :math:`{\rm phase}`: Zero.

    Bounds
        - :math:`a`: [-2, 2] scaled to the maximum signal value.
        - :math:`b`: [-1, 1] scaled to the maximum signal value.
        - :math:`{\rm freq}`: [0, inf].
        - :math:`{\rm phase}`: [-pi, pi].

    """

    # TODO this depends on reps.
    __series__ = [
        SeriesDef(
            fit_func=lambda x, a, freq, phase, b: fit_function.cos(
                x, amp=a, freq=freq, phase=phase, baseline=b
            ),
            plot_color="blue",
        ),
        SeriesDef(
            fit_func=lambda x, a, freq, phase, b: fit_function.cos(
                x, amp=a, freq=freq, phase=phase, baseline=b
            ),
            plot_color="green",
        )
        ,
        SeriesDef(
            fit_func=lambda x, a, freq, phase, b: fit_function.cos(
                x, amp=a, freq=freq, phase=phase, baseline=b
            ),
            plot_color="red",
        )
    ]

    __colors__ = ["blue", "green", "red", "purple", "black"]

    @classmethod
    def _default_options(cls):
        """Return the default analysis options.
        See :meth:`~qiskit_experiment.analysis.CurveAnalysis._default_options` for
        descriptions of analysis options.
        TODO
        """
        default_options = super()._default_options()
        default_options.p0 = {"a": None, "freq": None, "phase": None, "b": None}
        default_options.bounds = {"a": None, "freq": None, "phase": None, "b": None}
        default_options.fit_reports = {"freq": "rate"}
        default_options.xlabel = "Beta"
        default_options.ylabel = "Signal (arb. units)"

        return default_options

    def _setup_fitting(self, reps: list, **options) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """

        :param options:
        :return:
        """
        self.__series__ = []
        for index, rep in enumerate(reps):
            self.__series__.append(
                SeriesDef(
                    fit_func=lambda x, a, freq, phase, b: fit_function.cos(
                        x, amp=a, freq=freq, phase=phase, baseline=b
                    ),
                    plot_color=self.__colors__[index % len(self.__colors__)],
                )
            )

        user_p0 = self._get_option("p0")
        user_bounds = self._get_option("bounds")

        max_abs_y = np.max(np.abs(self._data().y))

        # Use a fast Fourier transform to guess the frequency.
        fft = np.abs(np.fft.fft(self._data().y - np.average(self._data().y)))
        damp = self._data().x[1] - self._data().x[0]
        freqs = np.linspace(0.0, 1.0 / (2.0 * damp), len(fft))

        b_guess = np.average(self._data().y)
        a_guess = np.max(self._data().y) - np.min(self._data().y) - b_guess
        f_guess = freqs[np.argmax(fft[0 : len(fft) // 2])]

        if user_p0["phase"] is not None:
            p_guesses = [user_p0["phase"]]
        else:
            p_guesses = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4, np.pi]

        fit_options = []
        for p_guess in p_guesses:
            fit_option = {
                "p0": {
                    "a": user_p0["a"] or a_guess,
                    "freq": user_p0["freq"] or f_guess,
                    "phase": p_guess,
                    "b": user_p0["b"] or b_guess,
                },
                "bounds": {
                    "a": user_bounds["a"] or (-2 * max_abs_y, 2 * max_abs_y),
                    "freq": user_bounds["freq"] or (0, np.inf),
                    "phase": user_bounds["phase"] or (-np.pi, np.pi),
                    "b": user_bounds["b"] or (-1 * max_abs_y, 1 * max_abs_y),
                },
            }
            fit_option.update(options)
            fit_options.append(fit_option)

        return fit_options

    def _post_analysis(self, analysis_result: CurveAnalysisResult) -> CurveAnalysisResult:
        """Algorithmic criteria for whether the fit is good or bad.

        A good fit has:
            - TODO
        """
        analysis_result["quality"] = "computer_bad"

        return analysis_result

class RoughDrag(BaseExperiment):
    """An experiment that scans the drag parameter to find the optimal value.

    The rough Drag calibration will run several series of circuits. In a given circuit
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
    the leakage to the second excited state.
    """

    __analysis_class__ = RoughDragAnalysis

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
        options = super()._default_experiment_options()

        options.xp = None
        options.xm = None
        options.amp = 0.2
        options.duration = 160
        options.sigma = 40
        options.reps = [1, 3, 5]
        options.betas = np.linspace(0, 10, 51)

        return options

    def __init__(self, qubit: int):
        """
        Args:
            qubit: The qubit for which to run the rough Drag calibration.
        """

        super().__init__([qubit])

    def circuits(self, backend: Optional[Backend] = None) -> List[QuantumCircuit]:
        """Create the circuits for the rough Drag calibration.

        Args:
            backend: A backend object.

        Returns:
            circuits: The circuits that will run the rough Drag calibration.

        Raises:
            CalibrationError:
                - If the beta parameters in the xp and xm pulses are not the same.
                - If either the xp or xm pulse do not have at least one Drag pulse.
        """
        # TODO this is temporarily logic. Need update of circuit data and processor logic.
        self.set_analysis_options(
            data_processor=get_to_signal_processor(
                meas_level=self.run_options.meas_level,
                meas_return=self.run_options.meas_return,
            ),
            reps=self.experiment_options.reps
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
                    pulse.drive_channel(self._physical_qubits[0])
                )

            with pulse.build(backend=backend, name="xm") as xm:
                pulse.play(
                    pulse.Drag(
                        duration=self.experiment_options.duration,
                        amp=-self.experiment_options.amp,
                        sigma=self.experiment_options.sigma,
                        beta=beta,
                    ),
                    pulse.drive_channel(self._physical_qubits[0])
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

        circuits = []
        for beta in self.experiment_options.betas:

            beta = np.round(beta, decimals=6)

            for rep in self.experiment_options.reps:
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
                "qubit": self.physical_qubits[0],
                "xval": beta,
                "series": str(rep)
                }

                circuits.append(circuit)

        return circuits
