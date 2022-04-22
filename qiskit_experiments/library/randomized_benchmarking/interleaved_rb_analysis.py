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
"""
Interleaved RB analysis class.
"""
from typing import List, Union

import numpy as np
import qiskit_experiments.curve_analysis as curve
from qiskit_experiments.framework import AnalysisResultData


class InterleavedRBAnalysis(curve.CurveAnalysis):
    r"""A class to analyze interleaved randomized benchmarking experiment.

    # section: overview
        This analysis takes only two series for standard and interleaved RB curve fitting.
        From the fit :math:`\alpha` and :math:`\alpha_c` value this analysis estimates
        the error per Clifford (EPC) of the interleaved gate.

        The EPC estimate is obtained using the equation

        .. math::

            r_{\mathcal{C}}^{\text{est}} =
                \frac{\left(d-1\right)\left(1-\alpha_{\overline{\mathcal{C}}}/\alpha\right)}{d}

        The systematic error bounds are given by

        .. math::

            E = \min\left\{
                \begin{array}{c}
                    \frac{\left(d-1\right)\left[\left|\alpha-\alpha_{\overline{\mathcal{C}}}\right|
                    +\left(1-\alpha\right)\right]}{d} \\
                    \frac{2\left(d^{2}-1\right)\left(1-\alpha\right)}
                    {\alpha d^{2}}+\frac{4\sqrt{1-\alpha}\sqrt{d^{2}-1}}{\alpha}
                \end{array}
            \right.

        See Ref. [1] for more details.

    # section: fit_model
        The fit is based on the following decay functions:

        Fit model for standard RB

        .. math::

            F(x) = a \alpha^{x} + b

        Fit model for interleaved RB

        .. math::

            F(x) = a (\alpha_c \alpha)^{x_2} + b

    # section: fit_parameters
        defpar a:
            desc: Height of decay curve.
            init_guess: Determined by :math:`1 - b`.
            bounds: [0, 1]
        defpar b:
            desc: Base line.
            init_guess: Determined by :math:`(1/2)^n` where :math:`n` is number of qubit.
            bounds: [0, 1]
        defpar \alpha:
            desc: Depolarizing parameter.
            init_guess: Determined by :func:`~rb_decay` with standard RB curve.
            bounds: [0, 1]
        defpar \alpha_c:
            desc: Ratio of the depolarizing parameter of interleaved RB to standard RB curve.
            init_guess: Determined by alpha of interleaved RB curve devided by one of
                standard RB curve. Both alpha values are estimated by :func:`~rb_decay`.
            bounds: [0, 1]

    # section: reference
        .. ref_arxiv:: 1 1203.4550

    """

    __series__ = [
        curve.SeriesDef(
            name="Standard",
            fit_func=lambda x, a, alpha, alpha_c, b: curve.fit_function.exponential_decay(
                x, amp=a, lamb=-1.0, base=alpha, baseline=b
            ),
            filter_kwargs={"interleaved": False},
            plot_color="red",
            plot_symbol=".",
            model_description=r"a \alpha^{x} + b",
        ),
        curve.SeriesDef(
            name="Interleaved",
            fit_func=lambda x, a, alpha, alpha_c, b: curve.fit_function.exponential_decay(
                x, amp=a, lamb=-1.0, base=alpha * alpha_c, baseline=b
            ),
            filter_kwargs={"interleaved": True},
            plot_color="orange",
            plot_symbol="^",
            model_description=r"a (\alpha_c\alpha)^{x} + b",
        ),
    ]

    @classmethod
    def _default_options(cls):
        """Default analysis options."""
        default_options = super()._default_options()
        default_options.result_parameters = ["alpha", "alpha_c"]
        return default_options

    def _generate_fit_guesses(
        self, user_opt: curve.FitOptions
    ) -> Union[curve.FitOptions, List[curve.FitOptions]]:
        """Compute the initial guesses.

        Args:
            user_opt: Fit options filled with user provided guess and bounds.

        Returns:
            List of fit options that are passed to the fitter function.
        """
        user_opt.bounds.set_if_empty(
            a=(0, 1),
            alpha=(0, 1),
            alpha_c=(0, 1),
            b=(0, 1),
        )

        b_guess = 1 / 2**self._num_qubits
        a_guess = 1 - b_guess

        # for standard RB curve
        std_curve = self._data(series_name="Standard")
        alpha_std = curve.guess.rb_decay(std_curve.x, std_curve.y, a=a_guess, b=b_guess)

        # for interleaved RB curve
        int_curve = self._data(series_name="Interleaved")
        alpha_int = curve.guess.rb_decay(int_curve.x, int_curve.y, a=a_guess, b=b_guess)

        alpha_c = min(alpha_int / alpha_std, 1.0)

        user_opt.p0.set_if_empty(
            b=b_guess,
            a=a_guess,
            alpha=alpha_std,
            alpha_c=alpha_c,
        )

        return user_opt

    def _format_data(
        self,
        curve_data: curve.CurveData,
    ) -> curve.CurveData:
        # TODO Eventually move this to data processor, then create RB data processor.

        # take average over the same x value by keeping sigma
        data_allocation, xdata, ydata, sigma, shots = curve.data_processing.multi_mean_xy_data(
            series=curve_data.data_allocation,
            xdata=curve_data.x,
            ydata=curve_data.y,
            sigma=curve_data.y_err,
            shots=curve_data.shots,
            method="sample",
        )

        # sort by x value in ascending order
        data_allocation, xdata, ydata, sigma, shots = curve.data_processing.data_sort(
            series=data_allocation,
            xdata=xdata,
            ydata=ydata,
            sigma=sigma,
            shots=shots,
        )

        return curve.CurveData(
            x=xdata,
            y=ydata,
            y_err=sigma,
            shots=shots,
            data_allocation=data_allocation,
            labels=curve_data.labels,
        )

    def _extra_database_entry(self, fit_data: curve.FitData) -> List[AnalysisResultData]:
        """Calculate EPC."""
        nrb = 2**self._num_qubits
        scale = (nrb - 1) / nrb

        alpha = fit_data.fitval("alpha")
        alpha_c = fit_data.fitval("alpha_c")

        # Calculate epc_est (=r_c^est) - Eq. (4):
        epc = scale * (1 - alpha_c)

        # Calculate the systematic error bounds - Eq. (5):
        systematic_err_1 = scale * (abs(alpha.n - alpha_c.n) + (1 - alpha.n))
        systematic_err_2 = (
            2 * (nrb * nrb - 1) * (1 - alpha.n) / (alpha.n * nrb * nrb)
            + 4 * (np.sqrt(1 - alpha.n)) * (np.sqrt(nrb * nrb - 1)) / alpha.n
        )

        systematic_err = min(systematic_err_1, systematic_err_2)
        systematic_err_l = epc.n - systematic_err
        systematic_err_r = epc.n + systematic_err

        extra_data = AnalysisResultData(
            name="EPC",
            value=epc,
            chisq=fit_data.reduced_chisq,
            quality=self._evaluate_quality(fit_data),
            extra={
                "EPC_systematic_err": systematic_err,
                "EPC_systematic_bounds": [max(systematic_err_l, 0), systematic_err_r],
            },
        )

        return [extra_data]
