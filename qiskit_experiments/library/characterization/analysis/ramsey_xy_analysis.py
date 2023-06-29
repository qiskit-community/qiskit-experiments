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
from itertools import product

import lmfit
import numpy as np

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
        curve_data: curve.CurveData,
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
            - a reduced chi-squared lower than three,
            - an error on the frequency smaller than the frequency.
        """
        fit_freq = fit_data.ufloat_params["freq"]

        criteria = [
            fit_data.reduced_chisq < 3,
            curve.utils.is_error_not_significant(fit_freq),
        ]

        if all(criteria):
            return "good"

        return "bad"


class StarkRamseyXYAmpScanAnalysis(curve.CurveAnalysis):
    r"""Ramsey XY analysis for the Stark shifted phase sweep.

    # section: overview

        This analysis is a variant of :class:`RamseyXYAnalysis` in which
        the data is fit for a trigonometric function model with a linear phase.
        By contrast, in this model, the phase is assumed to be a polynominal of the x-data,
        and techniques to compute a good initial guess for these polynominal coefficients
        are not trivial. For example, when the phase is a linear function of the x-data,
        one may apply a Fourier transform to the data to estimate the coefficient,
        but this technique can not be used for a higher order polynomial.

        This analysis assumes the following polynominal for the phase imparted by the Stark shift.

        .. math::

            \theta_{\text Stark}(x) = 2 \pi t_S f_S(x),

        where

        .. math::

            f_S(x) = c_1 x + c_2 x^2 + c_3 x^3 + f_\epsilon,

        denotes the Stark shift. For the lowest order perturbative expansion of a single driven qubit,
        the Stark shift is a quadratic function of :math:`x`, but linear and cubic terms
        and a constant offset are also considered to account for
        other effects, e.g. strong drive, collisions, TLS, and so forth,
        and frequency mis-calibration, respectively.

    # section: fit_model

        .. math::

            F_{X+} = \text{amp} \cdot \cos \left( dt f_S^+(x) \right) + \text{offset}, \\
            F_{Y+} = \text{amp} \cdot \sin \left( dt f_S^+(x) \right) + \text{offset}, \\
            F_{X-} = \text{amp} \cdot \cos \left( dt f_S^-(x) \right) + \text{offset}, \\
            F_{Y-} = \text{amp} \cdot \sin \left( dt f_S^-(x) \right) + \text{offset},

        where

        .. math ::

            f_S^\nu(x) = c_1^\nu x + c_2^\nu x^2 + c_3^\nu x^3 + f_\epsilon.

        The Stark shift is asymmetric with respect to :math:`x=0`, because of the
        anti-crossings of higher energy levels. In a typical transmon qubit,
        these levels appear only in :math:`f_S < 0` because of the negative anharmonicity.
        To precisely fit the results, this analysis uses different model parameters
        for positive (:math:`x > 0`) and negative (:math:`x < 0`) shift domains.

        To obtain the initial guess, the following calculation is employed in this analysis.
        First, oscillations in each quadrature are normalized and differentiated.

        .. math ::

            \dot{F}_X = \frac{\partial}{\partial x} \bar{F}_X = dt \frac{d}{dx} f_S \bar{F}_Y, \\
            \dot{F}_Y = \frac{\partial}{\partial x} \bar{F}_Y = - dt \frac{d}{dx} f_S \bar{F}_X. \\

        The square root of the sum of the squares of the above quantities yields

        .. math ::

            \sqrt{\dot{F}_X + \dot{F}_Y} = dt \frac{d}{dx} f_S = dt (c_1 + 2 c_2 x + 3 c_3 x^2).

        By using this synthesized data, one can estimate the initial guess of the
        polynomial coefficients by quadratic regression.
        This fit protocol is independently conducted for the experiment data on the
        positive and negative shift domain.

    # section: fit_parameters

        defpar \rm amp:
            desc: Amplitude of both series.
            init_guess: Median of root sum square of Ramsey X and Y oscillation.
            bounds: [0, 1]

        defpar \rm offset:
            desc: Base line of all series.
            init_guess: The average of the data.
            bounds: [-1, 1]

        defpar dt:
            desc: Fixed parameter of :math:`2 \pi t_S`, where :math:`t_S` is
                the ``stark_length`` experiment option.
            init_guess: Automatically set from metadata when this analysis is run.
            bounds: None

        defpar c_1^+:
            desc: The linear term coefficient of the positive Stark shift.
            init_guess: See the fit model description.
            bounds: None

        defpar c_2^+:
            desc: The quadratic term coefficient of the positive Stark shift.
            init_guess: See the fit model description.
            bounds: None

        defpar c_3^+:
            desc: The cubic term coefficient of the positive Stark shift.
            init_guess: See the fit model description.
            bounds: None

        defpar c_1^-:
            desc: The linear term coefficient of the negative Stark shift.
            init_guess: See the fit model description.
            bounds: None

        defpar c_2^-:
            desc: The quadratic term coefficient of the negative Stark shift.
            init_guess: See the fit model description.
            bounds: None

        defpar c_3^-:
            desc: The cubic term coefficient of the negative Stark shift.
            init_guess: See the fit model description.
            bounds: None

        defpar f_\epsilon:
            desc: Constant phase accumulation which is independent of the Stark tone amplitude.
            init_guess: 0
            bounds: None

    # section: see_also

        :class:`qiskit_experiments.library.characterization.analysis.ramsey_xy_analysis.RamseyXYAnalysis`

    """

    def __init__(self):

        models = []
        for direction in ("pos", "neg"):
            # Ramsey phase := 2π Δf(x) Δt; Δf(x) = c1 x + c2 x^2 + c3 x^3 + f_err
            # dt := 2π Δt (const.)
            fs = f"(c1_{direction} * x + c2_{direction} * x**2 + c3_{direction} * x**3 + f_err)"
            models.extend(
                [
                    lmfit.models.ExpressionModel(
                        expr=f"amp * cos(dt * {fs}) + offset",
                        name=f"X{direction}",
                    ),
                    lmfit.models.ExpressionModel(
                        expr=f"amp * sin(dt * {fs}) + offset",
                        name=f"Y{direction}",
                    ),
                ]
            )

        super().__init__(models=models)

    @classmethod
    def _default_options(cls):
        """Default analysis options."""
        ramsey_plotter = vis.CurvePlotter(vis.MplDrawer())
        ramsey_plotter.set_figure_options(
            xlabel="Stark tone amplitude",
            ylabel="P(1)",
            ylim=(0, 1),
            series_params={
                "Xpos": {"color": "#123FE8", "symbol": "o", "label": "Ramsey X(+)"},
                "Ypos": {"color": "#6312E8", "symbol": "^", "label": "Ramsey Y(+)"},
                "Xneg": {"color": "#E83812", "symbol": "o", "label": "Ramsey X(-)"},
                "Yneg": {"color": "#E89012", "symbol": "^", "label": "Ramsey Y(-)"},
            },
        )
        ramsey_plotter.set_options(style=vis.PlotStyle({"figsize": (12, 5)}))

        options = super()._default_options()
        options.update_options(
            result_parameters=[
                curve.ParameterRepr("c1_pos", "stark_pos_coef_o1", "Hz"),
                curve.ParameterRepr("c2_pos", "stark_pos_coef_o2", "Hz"),
                curve.ParameterRepr("c3_pos", "stark_pos_coef_o3", "Hz"),
                curve.ParameterRepr("c1_neg", "stark_neg_coef_o1", "Hz"),
                curve.ParameterRepr("c2_neg", "stark_neg_coef_o2", "Hz"),
                curve.ParameterRepr("c3_neg", "stark_neg_coef_o3", "Hz"),
                curve.ParameterRepr("f_err", "stark_ferr", "Hz"),
            ],
            data_subfit_map={
                "Xpos": {"series": "X", "direction": "pos"},
                "Ypos": {"series": "Y", "direction": "pos"},
                "Xneg": {"series": "X", "direction": "neg"},
                "Yneg": {"series": "Y", "direction": "neg"},
            },
            plotter=ramsey_plotter,
        )

        return options

    def _generate_fit_guesses(
        self,
        user_opt: curve.FitOptions,
        curve_data: curve.CurveData,
    ) -> Union[curve.FitOptions, List[curve.FitOptions]]:
        """Create algorithmic initial fit guess from analysis options and curve data.

        Args:
            user_opt: Fit options filled with user provided guess and bounds.
            curve_data: Formatted data collection to fit.

        Returns:
            List of fit options that are passed to the fitter function.
        """
        # Compute offset guess
        user_opt.p0.set_if_empty(
            offset=np.mean(curve_data.y),
            f_err=0.0,
        )
        user_opt.bounds.set_if_empty(
            offset=(-1, 1),
            amp=(0, 1),
        )
        est_offs = user_opt.p0["offset"]

        # Compute amplitude guess
        amps = np.zeros(0)
        for direction in ("pos", "neg"):
            ram_x_off = curve_data.get_subset_of(f"X{direction}").y - est_offs
            ram_y_off = curve_data.get_subset_of(f"Y{direction}").y - est_offs
            amps = np.concatenate([amps, np.sqrt(ram_x_off**2 + ram_y_off**2)])
        user_opt.p0.set_if_empty(amp=np.median(amps))
        est_a = user_opt.p0["amp"]
        d_const = user_opt.p0["dt"]

        # Compute polynominal coefficients
        guesses = []
        for direction in ("pos", "neg"):
            ram_x_data = curve_data.get_subset_of(f"X{direction}")
            ram_y_data = curve_data.get_subset_of(f"Y{direction}")
            xvals = ram_x_data.x

            # Get normalized sinusoidals
            xnorm = (ram_x_data.y - est_offs) / est_a
            ynorm = (ram_y_data.y - est_offs) / est_a

            # Compute derivative to extract polynominals from sinusoidal
            dx = np.diff(xnorm) / np.diff(xvals)
            dy = np.diff(ynorm) / np.diff(xvals)

            # Eliminate sinusoidal
            phase_poly = np.sqrt(dx**2 + dy**2)

            tmp = []
            # Consider both signs to expand square root.
            for sign in (1, -1):
                # Do polyfit up to 2rd order.
                # This must correspond to the 3rd order in the original function.
                vmat_xpoly = np.vstack((xvals[1:] ** 2, xvals[1:], np.ones(xvals.size - 1))).T
                coeffs = np.linalg.lstsq(vmat_xpoly, sign * phase_poly, rcond=-1)[0]

                poly_guess = {
                    f"c1_{direction}": coeffs[2] / 1 / d_const,
                    f"c2_{direction}": coeffs[1] / 2 / d_const,
                    f"c3_{direction}": coeffs[0] / 3 / d_const,
                }
                tmp.append(poly_guess)
            guesses.append(tmp)

        fit_opts = []
        for guess_pos, guess_neg in product(*guesses):
            new_opt = user_opt.copy()
            new_opt.p0.set_if_empty(**guess_pos, **guess_neg)
            fit_opts.append(new_opt)

        return fit_opts

    def _initialize(
        self,
        experiment_data: ExperimentData,
    ):
        super()._initialize(experiment_data)

        # Set scaling factor to convert phase to frequency
        fixed_params = self.options.fixed_parameters.copy()
        fixed_params["dt"] = 2 * np.pi * experiment_data.metadata["stark_length"]
        self.set_options(fixed_parameters=fixed_params)
