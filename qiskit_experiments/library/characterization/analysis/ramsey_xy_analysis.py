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

"""The analysis class for the Ramsey XY experiment."""

from typing import List, Union

import lmfit
import numpy as np
from uncertainties import unumpy as unp

import qiskit_experiments.curve_analysis as curve
import qiskit_experiments.visualization as vis
from qiskit_experiments.framework import ExperimentData


class RamseyXYAnalysis(curve.CurveAnalysis):
    r"""Ramsey XY analysis based on a fit to a cosine function and a sine function.

    # section: fit_model

        Analyze a Ramsey XY experiment by fitting the X and Y series to a cosine and sine
        function, respectively. The two functions share the frequency and amplitude parameters.

        .. math::

            y_X = {\rm amp}e^{-x/\tau}\cos\left(2\pi\cdot{\rm freq}_i\cdot x\right) + {\rm base} \\
            y_Y = {\rm amp}e^{-x/\tau}\sin\left(2\pi\cdot{\rm freq}_i\cdot x\right) + {\rm base}

    # section: fit_parameters
        defpar \rm amp:
            desc: Amplitude of both series.
            init_guess: Half of the maximum y value less the minimum y value. When the
                oscillation frequency is low, it uses an averaged difference of
                Ramsey X data - Ramsey Y data.
            bounds: [0, 2 * average y peak-to-peak]

        defpar \tau:
            desc: The exponential decay of the curve.
            init_guess: The initial guess is obtained by fitting an exponential to the
                square root of (X data)**2 + (Y data)**2.
            bounds: [0, inf]

        defpar \rm base:
            desc: Base line of both series.
            init_guess: Roughly the average of the data. When the oscillation frequency is low,
                it uses an averaged data of Ramsey Y experiment.
            bounds: [min y - average y peak-to-peak, max y + average y peak-to-peak]

        defpar \rm freq:
            desc: Frequency of both series. This is the parameter of interest.
            init_guess: The frequency with the highest power spectral density.
            bounds: [-inf, inf]

        defpar \rm phase:
            desc: Common phase offset.
            init_guess: 0
            bounds: [-pi, pi]
    """

    def __init__(self):
        super().__init__(
            models=[
                lmfit.models.ExpressionModel(
                    expr="amp * exp(-x / tau) * cos(2 * pi * freq * x + phase) + base",
                    name="X",
                ),
                lmfit.models.ExpressionModel(
                    expr="amp * exp(-x / tau) * sin(2 * pi * freq * x + phase) + base",
                    name="Y",
                ),
            ]
        )

    @classmethod
    def _default_options(cls):
        """Return the default analysis options.

        See :meth:`~qiskit_experiment.curve_analysis.CurveAnalysis._default_options` for
        descriptions of analysis options.
        """
        default_options = super()._default_options()
        default_options.data_subfit_map = {
            "X": {"series": "X"},
            "Y": {"series": "Y"},
        }
        default_options.plotter.set_figure_options(
            xlabel="Delay",
            ylabel="Signal (arb. units)",
            xval_unit="s",
        )
        default_options.result_parameters = ["freq"]

        return default_options

    def _generate_fit_guesses(
        self,
        user_opt: curve.FitOptions,
        curve_data: curve.ScatterTable,
    ) -> Union[curve.FitOptions, List[curve.FitOptions]]:
        """Create algorithmic initial fit guess from analysis options and curve data.

        Args:
            user_opt: Fit options filled with user provided guess and bounds.
            curve_data: Formatted data collection to fit.

        Returns:
            List of fit options that are passed to the fitter function.
        """
        ramx_data = curve_data.get_subset_of("X")
        ramy_data = curve_data.get_subset_of("Y")

        # At very low frequency, y value of X (Y) curve stay at P=1.0 (0.5) for all x values.
        # Computing y peak-to-peak with combined data gives fake amplitude of 0.25.
        # Same for base, i.e. P=0.75 is often estimated in this case.
        full_y_ptp = np.ptp(curve_data.y)
        avg_y_ptp = 0.5 * (np.ptp(ramx_data.y) + np.ptp(ramy_data.y))
        max_y = np.max(curve_data.y)
        min_y = np.min(curve_data.y)

        user_opt.bounds.set_if_empty(
            amp=(0, full_y_ptp * 2),
            tau=(0, np.inf),
            base=(min_y - avg_y_ptp, max_y + avg_y_ptp),
            phase=(-np.pi, np.pi),
        )

        if avg_y_ptp < 0.5 * full_y_ptp:
            # When X and Y curve don't oscillate, X (Y) usually stays at P(1) = 1.0 (0.5).
            # So peak-to-peak of full data is something around P(1) = 0.75, while
            # single curve peak-to-peak is almost zero.
            avg_x = np.average(ramx_data.y)
            avg_y = np.average(ramy_data.y)

            user_opt.p0.set_if_empty(
                amp=np.abs(avg_x - avg_y),
                tau=100 * np.max(curve_data.x),
                base=avg_y,
                phase=0.0,
                freq=0.0,
            )
            return user_opt

        base_guess_x = curve.guess.constant_sinusoidal_offset(ramx_data.y)
        base_guess_y = curve.guess.constant_sinusoidal_offset(ramy_data.y)
        base_guess = 0.5 * (base_guess_x + base_guess_y)
        user_opt.p0.set_if_empty(
            amp=0.5 * full_y_ptp,
            base=base_guess,
            phase=0.0,
        )

        # Guess the exponential decay by combining both curves
        ramx_unbiased = ramx_data.y - user_opt.p0["base"]
        ramy_unbiased = ramy_data.y - user_opt.p0["base"]
        decay_data = ramx_unbiased**2 + ramy_unbiased**2
        if np.ptp(decay_data) < 0.95 * 0.5 * full_y_ptp:
            # When decay is less than 95 % of peak-to-peak value, ignore decay and
            # set large enough tau value compared with the measured x range.
            user_opt.p0.set_if_empty(tau=1000 * np.max(curve_data.x))
        else:
            user_opt.p0.set_if_empty(tau=-1 / curve.guess.exp_decay(ramx_data.x, decay_data))

        # Guess the oscillation frequency, remove offset to eliminate DC peak
        freq_guess_x = curve.guess.frequency(ramx_data.x, ramx_unbiased)
        freq_guess_y = curve.guess.frequency(ramy_data.x, ramy_unbiased)
        freq_val = 0.5 * (freq_guess_x + freq_guess_y)

        # FFT might be up to 1/2 bin off
        df = 2 * np.pi / (np.min(np.diff(ramx_data.x)) * ramx_data.x.size)
        freq_guesses = [freq_val - df, freq_val + df, freq_val]

        # Ramsey XY is frequency sign sensitive.
        # Since experimental data is noisy, correct sign is hardly estimated with phase velocity.
        # Try both positive and negative frequency to find the best fit.
        opts = []
        for sign in (1, -1):
            for freq_guess in freq_guesses:
                opt = user_opt.copy()
                opt.p0.set_if_empty(freq=sign * freq_guess)
                opts.append(opt)

        return opts

    def _evaluate_quality(self, fit_data: curve.CurveFitResult) -> Union[str, None]:
        """Algorithmic criteria for whether the fit is good or bad.

        A good fit has:
            - a reduced chi-squared lower than three and greater than zero,
            - an error on the frequency smaller than the frequency.
        """
        fit_freq = fit_data.ufloat_params["freq"]

        criteria = [
            0 < fit_data.reduced_chisq < 3,
            curve.utils.is_error_not_significant(fit_freq),
        ]

        if all(criteria):
            return "good"

        return "bad"


class StarkRamseyXYAmpScanAnalysis(curve.CurveAnalysis):
    r"""Ramsey XY analysis for the Stark shifted phase sweep.

    # section: overview

        This analysis is a variant of :class:`RamseyXYAnalysis`. In both cases, the X and Y
        data are treated as the real and imaginary parts of a complex oscillating signal.
        In :class:`RamseyXYAnalysis`, the data are fit assuming a phase varying linearly with
        the x-data corresponding to a constant frequency and assuming an exponentially
        decaying amplitude. By contrast, in this model, the phase is assumed to be
        a third order polynomial :math:`\theta(x)` of the x-data.
        Additionally, the amplitude is not assumed to follow a specific form.
        Techniques to compute a good initial guess for the polynomial coefficients inside
        a trigonometric function like this are not trivial. Instead, this analysis extracts the
        raw phase and runs fits the extracted data to a polynomial :math:`\theta(x)` directly.

        The measured P1 values for a Ramsey X and Y experiment can be written in the form of
        a trignometric function taking the phase polynomial :math:`\theta(x)`:

        .. math::

            P_X =  \text{amp}(x) \cdot \cos \theta(x) + \text{offset},\\
            P_Y =  \text{amp}(x) \cdot \sin \theta(x) + \text{offset}.

        Hence the phase polynomial can be extracted as follows

        .. math::

            \theta(x) = \tan^{-1} \frac{P_Y}{P_X}.

        Because the arctangent is implemented by the ``atan2`` function
        defined in :math:`[-\pi, \pi]`, the computed :math:`\theta(x)` is unwrapped to
        ensure continuous phase evolution.

        We call attention to the fact that :math:`\text{amp}(x)` is also Stark tone amplitude
        dependent because of the qubit frequency dependence of the dephasing rate.
        In general :math:`\text{amp}(x)` is unpredictable due to dephasing from
        two-level systems distributed randomly in frequency
        or potentially due to qubit heating. This prevents us from precisely fitting
        the raw :math:`P_X`, :math:`P_Y` data. Fitting only the phase data makes the
        analysis robust to amplitude dependent dephasing.

        In this analysis, the phase polynomial is defined as

        .. math::

            \theta(x) = 2 \pi t_S f_S(x)

        where

        .. math::

            f_S(x) = c_1 x + c_2 x^2 + c_3 x^3 + f_{\rm err},

        denotes the Stark shift. For the lowest order perturbative expansion of a single driven qubit,
        the Stark shift is a quadratic function of :math:`x`, but linear and cubic terms
        and a constant offset are also considered to account for
        other effects, e.g. strong drive, collisions, TLS, and so forth,
        and frequency mis-calibration, respectively.

    # section: fit_model

        .. math::

            \theta^\nu(x) = c_1^\nu x + c_2^\nu x^2 + c_3^\nu x^3 + f_{\rm err},

        where :math:`\nu \in \{+, -\}`.
        The Stark shift is asymmetric with respect to :math:`x=0`, because of the
        anti-crossings of higher energy levels. In a typical transmon qubit,
        these levels appear only in :math:`f_S < 0` because of the negative anharmonicity.
        To precisely fit the results, this analysis uses different model parameters
        for positive (:math:`x > 0`) and negative (:math:`x < 0`) shift domains.

    # section: fit_parameters

        defpar c_1^+:
            desc: The linear term coefficient of the positive Stark shift
                (fit parameter: ``stark_pos_coef_o1``).
            init_guess: 0.
            bounds: None

        defpar c_2^+:
            desc: The quadratic term coefficient of the positive Stark shift.
                This parameter must be positive because Stark amplitude is chosen to
                induce blue shift when its sign is positive.
                Note that the quadratic term is the primary term
                (fit parameter: ``stark_pos_coef_o2``).
            init_guess: 1e6.
            bounds: [0, inf]

        defpar c_3^+:
            desc: The cubic term coefficient of the positive Stark shift
                (fit parameter: ``stark_pos_coef_o3``).
            init_guess: 0.
            bounds: None

        defpar c_1^-:
            desc: The linear term coefficient of the negative Stark shift.
                (fit parameter: ``stark_neg_coef_o1``).
            init_guess: 0.
            bounds: None

        defpar c_2^-:
            desc: The quadratic term coefficient of the negative Stark shift.
                This parameter must be negative because Stark amplitude is chosen to
                induce red shift when its sign is negative.
                Note that the quadratic term is the primary term
                (fit parameter: ``stark_neg_coef_o2``).
            init_guess: -1e6.
            bounds: [-inf, 0]

        defpar c_3^-:
            desc: The cubic term coefficient of the negative Stark shift
                (fit parameter: ``stark_neg_coef_o3``).
            init_guess: 0.
            bounds: None

        defpar f_{\rm err}:
            desc: Constant phase accumulation which is independent of the Stark tone amplitude.
                (fit parameter: ``stark_ferr``).
            init_guess: Averaege of y values at minimum absolute x values on
                positive and negative shift data.
            bounds: None

    # section: see_also

        :class:`qiskit_experiments.library.characterization.analysis.ramsey_xy_analysis.RamseyXYAnalysis`

    """

    def __init__(self):

        models = [
            lmfit.models.ExpressionModel(
                expr="c1_pos * x + c2_pos * x**2 + c3_pos * x**3 + f_err",
                name="FREQpos",
            ),
            lmfit.models.ExpressionModel(
                expr="c1_neg * x + c2_neg * x**2 + c3_neg * x**3 + f_err",
                name="FREQneg",
            ),
        ]
        super().__init__(models=models)

    @classmethod
    def _default_options(cls):
        """Default analysis options."""
        ramsey_plotter = vis.CurvePlotter(vis.MplDrawer())
        ramsey_plotter.set_figure_options(
            xlabel="Stark tone amplitude",
            ylabel=["Stark shift", "P1"],
            yval_unit=["Hz", None],
            series_params={
                "Fpos": {
                    "color": "#123FE8",
                    "symbol": "^",
                    "label": "",
                    "canvas": 0,
                },
                "Fneg": {
                    "color": "#123FE8",
                    "symbol": "v",
                    "label": "",
                    "canvas": 0,
                },
                "Xpos": {
                    "color": "#123FE8",
                    "symbol": "o",
                    "label": "Ramsey X",
                    "canvas": 1,
                },
                "Ypos": {
                    "color": "#6312E8",
                    "symbol": "^",
                    "label": "Ramsey Y",
                    "canvas": 1,
                },
                "Xneg": {
                    "color": "#E83812",
                    "symbol": "o",
                    "label": "Ramsey X",
                    "canvas": 1,
                },
                "Yneg": {
                    "color": "#E89012",
                    "symbol": "^",
                    "label": "Ramsey Y",
                    "canvas": 1,
                },
            },
            sharey=False,
        )
        ramsey_plotter.set_options(subplots=(2, 1), style=vis.PlotStyle({"figsize": (10, 8)}))

        options = super()._default_options()
        options.update_options(
            data_subfit_map={
                "Xpos": {"series": "X", "direction": "pos"},
                "Ypos": {"series": "Y", "direction": "pos"},
                "Xneg": {"series": "X", "direction": "neg"},
                "Yneg": {"series": "Y", "direction": "neg"},
            },
            result_parameters=[
                curve.ParameterRepr("c1_pos", "stark_pos_coef_o1", "Hz"),
                curve.ParameterRepr("c2_pos", "stark_pos_coef_o2", "Hz"),
                curve.ParameterRepr("c3_pos", "stark_pos_coef_o3", "Hz"),
                curve.ParameterRepr("c1_neg", "stark_neg_coef_o1", "Hz"),
                curve.ParameterRepr("c2_neg", "stark_neg_coef_o2", "Hz"),
                curve.ParameterRepr("c3_neg", "stark_neg_coef_o3", "Hz"),
                curve.ParameterRepr("f_err", "stark_ferr", "Hz"),
            ],
            plotter=ramsey_plotter,
            fit_category="freq",
            pulse_len=None,
        )

        return options

    def _freq_phase_coef(self) -> float:
        """Return a coefficient to convert frequency into phase value."""
        try:
            return 2 * np.pi * self.options.pulse_len
        except TypeError as ex:
            raise TypeError(
                "A float-value duration in units of sec of the Stark pulse must be provided. "
                f"The pulse_len option value {self.options.pulse_len} is not valid."
            ) from ex

    def _format_data(
        self,
        curve_data: curve.ScatterTable,
        category: str = "freq",
    ) -> curve.ScatterTable:

        curve_data = super()._format_data(curve_data, category="ramsey_xy")
        ramsey_xy = curve_data[curve_data.category == "ramsey_xy"]

        # Create phase data by arctan(Y/X)
        columns = list(curve_data.columns)
        phase_data = np.empty((0, len(columns)))
        y_mean = ramsey_xy.yval.mean()

        grouped = ramsey_xy.groupby("name")
        for m_id, direction in enumerate(("pos", "neg")):
            x_quadrature = grouped.get_group(f"X{direction}")
            y_quadrature = grouped.get_group(f"Y{direction}")
            if not np.array_equal(x_quadrature.xval, y_quadrature.xval):
                raise ValueError(
                    "Amplitude values of X and Y quadrature are different. "
                    "Same values must be used."
                )
            x_uarray = unp.uarray(x_quadrature.yval, x_quadrature.yerr)
            y_uarray = unp.uarray(y_quadrature.yval, y_quadrature.yerr)

            amplitudes = x_quadrature.xval.to_numpy()

            # pylint: disable=no-member
            phase = unp.arctan2(y_uarray - y_mean, x_uarray - y_mean)
            phase_n = unp.nominal_values(phase)
            phase_s = unp.std_devs(phase)

            # Unwrap phase
            # We assume a smooth slope and correct 2pi phase jump to minimize the change of the slope.
            unwrapped_phase = np.unwrap(phase_n)
            if amplitudes[0] < 0:
                # Preserve phase value closest to 0 amplitude
                unwrapped_phase = unwrapped_phase + (phase_n[-1] - unwrapped_phase[-1])

            # Store new data
            tmp = np.empty((len(amplitudes), len(columns)), dtype=object)
            tmp[:, columns.index("xval")] = amplitudes
            tmp[:, columns.index("yval")] = unwrapped_phase / self._freq_phase_coef()
            tmp[:, columns.index("yerr")] = phase_s / self._freq_phase_coef()
            tmp[:, columns.index("name")] = f"FREQ{direction}"
            tmp[:, columns.index("class_id")] = m_id
            tmp[:, columns.index("shots")] = x_quadrature.shots + y_quadrature.shots
            tmp[:, columns.index("category")] = category
            phase_data = np.r_[phase_data, tmp]

        return curve_data.append_list_values(other=phase_data)

    def _generate_fit_guesses(
        self,
        user_opt: curve.FitOptions,
        curve_data: curve.ScatterTable,
    ) -> Union[curve.FitOptions, List[curve.FitOptions]]:
        """Create algorithmic initial fit guess from analysis options and curve data.

        Args:
            user_opt: Fit options filled with user provided guess and bounds.
            curve_data: Formatted data collection to fit.

        Returns:
            List of fit options that are passed to the fitter function.
        """
        user_opt.bounds.set_if_empty(c2_pos=(0, np.inf), c2_neg=(-np.inf, 0))
        user_opt.p0.set_if_empty(
            c1_pos=0, c2_pos=1e6, c3_pos=0, c1_neg=0, c2_neg=-1e6, c3_neg=0, f_err=0
        )
        return user_opt

    def _create_figures(
        self,
        curve_data: curve.ScatterTable,
    ) -> List["matplotlib.figure.Figure"]:

        # plot unwrapped phase on first axis
        for d in ("pos", "neg"):
            sub_data = curve_data[(curve_data.name == f"FREQ{d}") & (curve_data.category == "freq")]
            self.plotter.set_series_data(
                series_name=f"F{d}",
                x_formatted=sub_data.xval.to_numpy(),
                y_formatted=sub_data.yval.to_numpy(),
                y_formatted_err=sub_data.yerr.to_numpy(),
            )

        # plot raw RamseyXY plot on second axis
        for name in ("Xpos", "Ypos", "Xneg", "Yneg"):
            sub_data = curve_data[(curve_data.name == name) & (curve_data.category == "ramsey_xy")]
            self.plotter.set_series_data(
                series_name=name,
                x_formatted=sub_data.xval.to_numpy(),
                y_formatted=sub_data.yval.to_numpy(),
                y_formatted_err=sub_data.yerr.to_numpy(),
            )

        # find base and amplitude guess
        ramsey_xy = curve_data[curve_data.category == "ramsey_xy"]
        offset_guess = 0.5 * (ramsey_xy.yval.min() + ramsey_xy.yval.max())
        amp_guess = 0.5 * np.ptp(ramsey_xy.yval)

        # plot frequency and Ramsey fit lines
        line_data = curve_data[curve_data.category == "fitted"]
        for direction in ("pos", "neg"):
            sub_data = line_data[line_data.name == f"FREQ{direction}"]
            if len(sub_data) == 0:
                continue
            xval = sub_data.xval.to_numpy()
            yn = sub_data.yval.to_numpy()
            ys = sub_data.yerr.to_numpy()
            yval = unp.uarray(yn, ys) * self._freq_phase_coef()

            # Ramsey fit lines are predicted from the phase fit line.
            # Note that this line doesn't need to match with the expeirment data
            # because Ramsey P1 data may fluctuate due to phase damping.

            # pylint: disable=no-member
            ramsey_cos = amp_guess * unp.cos(yval) + offset_guess
            ramsey_sin = amp_guess * unp.sin(yval) + offset_guess

            self.plotter.set_series_data(
                series_name=f"F{direction}",
                x_interp=xval,
                y_interp=yn,
            )
            self.plotter.set_series_data(
                series_name=f"X{direction}",
                x_interp=xval,
                y_interp=unp.nominal_values(ramsey_cos),
            )
            self.plotter.set_series_data(
                series_name=f"Y{direction}",
                x_interp=xval,
                y_interp=unp.nominal_values(ramsey_sin),
            )

            if np.isfinite(ys).all():
                self.plotter.set_series_data(
                    series_name=f"F{direction}",
                    y_interp_err=ys,
                )
                self.plotter.set_series_data(
                    series_name=f"X{direction}",
                    y_interp_err=unp.std_devs(ramsey_cos),
                )
                self.plotter.set_series_data(
                    series_name=f"Y{direction}",
                    y_interp_err=unp.std_devs(ramsey_sin),
                )
        return [self.plotter.figure()]

    def _initialize(
        self,
        experiment_data: ExperimentData,
    ):
        super()._initialize(experiment_data)

        # Set scaling factor to convert phase to frequency
        if "stark_length" in experiment_data.metadata:
            self.set_options(pulse_len=experiment_data.metadata["stark_length"])
