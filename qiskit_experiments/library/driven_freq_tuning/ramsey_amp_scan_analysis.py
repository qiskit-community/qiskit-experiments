# This code is part of Qiskit.
#
# (C) Copyright IBM 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
"""Ramsey amplitude scan analysis."""

from __future__ import annotations

from typing import List, Union

import lmfit
import numpy as np
from uncertainties import unumpy as unp

from qiskit.utils.deprecation import deprecate_func

import qiskit_experiments.curve_analysis as curve
import qiskit_experiments.visualization as vis
from qiskit_experiments.framework import ExperimentData, AnalysisResultData
from .coefficients import StarkCoefficients


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
        raw phase and runs fits on the extracted data to a polynomial :math:`\theta(x)` directly.

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
            init_guess: 0
            bounds: None

    # section: see_also

        :class:`qiskit_experiments.library.characterization.analysis.ramsey_xy_analysis.RamseyXYAnalysis`

    """

    @deprecate_func(
        since="0.8",
        package_name="qiskit-experiments",
        additional_msg=(
            "Due to the deprecation of Qiskit Pulse, experiments and related classses "
            "involving pulse gate calibrations like this one have been deprecated."
        ),
    )
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
        """Default analysis options.

        Analysis Options:
            pulse_len (float): Duration of effective Stark pulse in units of sec.
        """
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
        ramsey_xy = curve_data.filter(category="ramsey_xy")
        y_mean = ramsey_xy.y.mean()

        # Create phase data by arctan(Y/X)
        for data_id, direction in enumerate(("pos", "neg")):
            x_quadrature = ramsey_xy.filter(series=f"X{direction}")
            y_quadrature = ramsey_xy.filter(series=f"Y{direction}")
            if not np.array_equal(x_quadrature.x, y_quadrature.x):
                raise ValueError(
                    "Amplitude values of X and Y quadrature are different. "
                    "Same values must be used."
                )
            x_uarray = unp.uarray(x_quadrature.y, x_quadrature.y_err)
            y_uarray = unp.uarray(y_quadrature.y, y_quadrature.y_err)
            amplitudes = x_quadrature.x

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
            unwrapped_phase /= self._freq_phase_coef()
            phase_s /= self._freq_phase_coef()
            shot_sums = x_quadrature.shots + y_quadrature.shots
            for new_x, new_y, new_y_err, shot in zip(
                amplitudes, unwrapped_phase, phase_s, shot_sums
            ):
                curve_data.add_row(
                    xval=new_x,
                    yval=new_y,
                    yerr=new_y_err,
                    series_name=f"FREQ{direction}",
                    series_id=data_id,
                    shots=shot,
                    category=category,
                    analysis=self.name,
                )

        return curve_data

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

    def _create_analysis_results(
        self,
        fit_data: curve.CurveFitResult,
        quality: str,
        **metadata,
    ) -> List[AnalysisResultData]:
        outcomes = super()._create_analysis_results(fit_data, quality, **metadata)

        # Combine fit coefficients
        coeffs = StarkCoefficients(
            pos_coef_o1=fit_data.ufloat_params["c1_pos"].nominal_value,
            pos_coef_o2=fit_data.ufloat_params["c2_pos"].nominal_value,
            pos_coef_o3=fit_data.ufloat_params["c3_pos"].nominal_value,
            neg_coef_o1=fit_data.ufloat_params["c1_neg"].nominal_value,
            neg_coef_o2=fit_data.ufloat_params["c2_neg"].nominal_value,
            neg_coef_o3=fit_data.ufloat_params["c3_neg"].nominal_value,
            offset=fit_data.ufloat_params["f_err"].nominal_value,
        )
        outcomes.append(
            AnalysisResultData(
                name="stark_coefficients",
                value=coeffs,
                chisq=fit_data.reduced_chisq,
                quality=quality,
                extra=metadata,
            )
        )
        return outcomes

    def _create_figures(
        self,
        curve_data: curve.ScatterTable,
    ) -> List["matplotlib.figure.Figure"]:

        # plot unwrapped phase on first axis
        for direction in ("pos", "neg"):
            sub_data = curve_data.filter(series=f"FREQ{direction}", category="freq")
            self.plotter.set_series_data(
                series_name=f"F{direction}",
                x_formatted=sub_data.x,
                y_formatted=sub_data.y,
                y_formatted_err=sub_data.y_err,
            )

        # plot raw RamseyXY plot on second axis
        for name in ("Xpos", "Ypos", "Xneg", "Yneg"):
            sub_data = curve_data.filter(series=name, category="ramsey_xy")
            self.plotter.set_series_data(
                series_name=name,
                x_formatted=sub_data.x,
                y_formatted=sub_data.y,
                y_formatted_err=sub_data.y_err,
            )

        # find base and amplitude guess
        ramsey_xy = curve_data.filter(category="ramsey_xy")
        offset_guess = 0.5 * (np.min(ramsey_xy.y) + np.max(ramsey_xy.y))
        amp_guess = 0.5 * np.ptp(ramsey_xy.y)

        # plot frequency and Ramsey fit lines
        line_data = curve_data.filter(category="fitted")
        for direction in ("pos", "neg"):
            sub_data = line_data.filter(series=f"FREQ{direction}")
            if len(sub_data) == 0:
                continue
            xval = sub_data.x
            yn = sub_data.y
            ys = sub_data.y_err
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
