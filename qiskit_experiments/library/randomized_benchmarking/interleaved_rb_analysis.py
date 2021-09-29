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
from qiskit_experiments.framework import AnalysisResultData, FitVal
from .rb_analysis import RBAnalysis


class InterleavedRBAnalysis(RBAnalysis):
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
            init_guess: Determined by the average :math:`a` of the standard and interleaved RB.
            bounds: [0, 1]
        defpar b:
            desc: Base line.
            init_guess: Determined by the average :math:`b` of the standard and interleaved RB.
                Usually equivalent to :math:`(1/2)^n` where :math:`n` is number of qubit.
            bounds: [0, 1]
        defpar \alpha:
            desc: Depolarizing parameter.
            init_guess: Determined by the slope of :math:`(y - b)^{-x}` of the first and the
                second data point of the standard RB.
            bounds: [0, 1]
        defpar \alpha_c:
            desc: Ratio of the depolarizing parameter of interleaved RB to standard RB curve.
            init_guess: Estimate :math:`\alpha' = \alpha_c \alpha` from the
                interleaved RB curve, then divide this by the initial guess of :math:`\alpha`.
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
            plot_fit_uncertainty=True,
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
            plot_fit_uncertainty=True,
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

        # for standard RB curve
        std_curve = self._data(series_name="Standard")
        opt_std = user_opt.copy()
        opt_std = self._initial_guess(opt_std, std_curve.x, std_curve.y, self._num_qubits)

        # for interleaved RB curve
        int_curve = self._data(series_name="Interleaved")
        opt_int = user_opt.copy()
        if opt_int.p0["alpha_c"] is not None:
            opt_int.p0["alpha"] = opt_std.p0["alpha"] * opt_int.p0["alpha_c"]
        opt_int = self._initial_guess(opt_int, int_curve.x, int_curve.y, self._num_qubits)

        user_opt.p0.set_if_empty(
            a=np.mean([opt_std.p0["a"], opt_int.p0["a"]]),
            alpha=opt_std.p0["alpha"],
            alpha_c=min(opt_int.p0["alpha"] / opt_std.p0["alpha"], 1),
            b=np.mean([opt_std.p0["b"], opt_int.p0["b"]]),
        )

        return user_opt

    def _extra_database_entry(self, fit_data: curve.FitData) -> List[AnalysisResultData]:
        """Calculate EPC."""
        nrb = 2 ** self._num_qubits
        scale = (nrb - 1) / nrb

        alpha = fit_data.fitval("alpha")
        alpha_c = fit_data.fitval("alpha_c")

        # Calculate epc_est (=r_c^est) - Eq. (4):
        epc = FitVal(value=scale * (1 - alpha_c.value), stderr=scale * alpha_c.stderr)

        # Calculate the systematic error bounds - Eq. (5):
        systematic_err_1 = scale * (abs(alpha.value - alpha_c.value) + (1 - alpha.value))
        systematic_err_2 = (
            2 * (nrb * nrb - 1) * (1 - alpha.value) / (alpha.value * nrb * nrb)
            + 4 * (np.sqrt(1 - alpha.value)) * (np.sqrt(nrb * nrb - 1)) / alpha.value
        )
        systematic_err = min(systematic_err_1, systematic_err_2)
        systematic_err_l = epc.value - systematic_err
        systematic_err_r = epc.value + systematic_err

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
