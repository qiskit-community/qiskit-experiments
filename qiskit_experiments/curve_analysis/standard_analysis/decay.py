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
"""Decay analysis class."""

from typing import List, Union, Callable

import numpy as np

from qiskit_experiments.framework import ExperimentData
import qiskit_experiments.curve_analysis as curve
from qiskit_experiments.data_processing import DataProcessor
from qiskit_experiments.exceptions import AnalysisError


class DecayAnalysis(curve.CurveAnalysis):
    r"""A class to analyze general exponential decay curve.

    # section: fit_model

    The fit is based on the following decay function.

    .. math::
        F(x) = {\rm amp} \cdot e^{-x/\tau} + {\rm base}

    # section: fit_parameters

        defpar \rm amp:
           desc: Height of the decay curve.
           init_guess: Determined by :py:func:`~qiskit_experiments.curve_analysis.guess.min_height`.
           bounds: None

        defpar \rm base:
           desc: Base line of the decay curve.
           init_guess: Determined by the difference of minimum and maximum points.
           bounds: None

        defpar \tau:
           desc: This is the fit parameter of main interest.
           init_guess: Determined by :py:func:`~qiskit_experiments.curve_analysis.guess.exp_decay`.
           bounds: None

    """

    __series__ = [
        curve.SeriesDef(
            fit_func=lambda x, amp, base, tau: curve.fit_function.exponential_decay(
                x,
                amp=amp,
                lamb=1 / tau,
                baseline=base,
            ),
            plot_color="blue",
            model_description=r"amp \exp(-x/tau) + base",
            plot_fit_uncertainty=True,
        )
    ]

    def _generate_fit_guesses(
        self, user_opt: curve.FitOptions
    ) -> Union[curve.FitOptions, List[curve.FitOptions]]:
        """Compute the initial guesses.

        Args:
            user_opt: Fit options filled with user provided guess and bounds.

        Returns:
            List of fit options that are passed to the fitter function.

        Raises:
            AnalysisError: When the y data is likely constant.
        """
        curve_data = self._data()

        user_opt.p0.set_if_empty(base=curve.guess.min_height(curve_data.y)[0])

        alpha = curve.guess.exp_decay(curve_data.x, curve_data.y)

        if alpha != 0.0:
            user_opt.p0.set_if_empty(
                tau=-1 / alpha,
                amp=curve.guess.max_height(curve_data.y)[0] - user_opt.p0["base"],
            )
        else:
            # Likely there is no slope. Cannot fit constant line with this model.
            # Set some large enough number against to the scan range.
            user_opt.p0.set_if_empty(
                tau=100 * np.max(curve_data.x),
                amp=curve.guess.max_height(curve_data.y)[0] - user_opt.p0["base"],
            )
        return user_opt

    def _evaluate_quality(self, fit_data: curve.FitData) -> Union[str, None]:
        """Algorithmic criteria for whether the fit is good or bad.

        A good fit has:
            - a reduced chi-squared lower than three
            - tau error is less than its value
        """
        tau = fit_data.fitval("tau")

        criteria = [
            fit_data.reduced_chisq < 3,
            tau.stderr is None or tau.stderr < tau.value,
        ]

        if all(criteria):
            return "good"

        return "bad"

    def _extract_curves(
        self, experiment_data: ExperimentData, data_processor: Union[Callable, DataProcessor]
    ):
        """Extract curve data from experiment data.

        This method internally populates two types of curve data.

        - raw_data:

            This is the data directly obtained from the experiment data.
            You can access this data with ``self._data(label="raw_data")``.

        - fit_ready:

            This is the formatted data created by pre-processing defined by
            `self._format_data()` method. This method is implemented by subclasses.
            You can access to this data with ``self._data(label="fit_ready")``.

        If multiple series exist, you can optionally specify ``series_name`` in
        ``self._data`` method to filter data in the target series.

        .. notes::
            The target metadata properties to define each curve entry is described by
            the class attribute __series__ (see `filter_kwargs`).

        Args:
            experiment_data: ExperimentData object to fit parameters.
            data_processor: A callable or DataProcessor instance to format data into numpy array.
                This should take a list of dictionaries and return two tuple of float values,
                that represent a y value and an error of it.
        Raises:
            DataProcessorError: When `x_key` specified in the analysis option is not
                defined in the circuit metadata.
            AnalysisError:
                - When formatted data has label other than fit_ready.
                - When less the data contain less than three points
        """
        super()._extract_curves(experiment_data, data_processor)
        if len(self._data().x) < 3:
            raise AnalysisError("Number of points must be at least 3")
