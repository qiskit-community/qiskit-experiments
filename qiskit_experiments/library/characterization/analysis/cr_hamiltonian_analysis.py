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

"""Cross resonance Hamiltonian tomography experiment analysis."""

from typing import List, Dict
import numpy as np

import qiskit_experiments.curve_analysis as curve
from qiskit_experiments.framework import AnalysisResultData


class CrossResonanceHamiltonianAnalysis(curve.CompositeCurveAnalysis):
    r"""A class to analyze cross resonance Hamiltonian tomography experiment.

    # section: fit_model

        This analysis performs :class:`.BlochTrajectoryAnalysis` on the target qubit
        with the control qubit states in :math:`\in \{ |0\rangle, |1\rangle \}`.

        Based on the fit result, cross resonance Hamiltonian coefficients can be determined by

        .. math::

            ZX &= \frac{p_{x, |0\rangle} - p_{x, |1\rangle}}{2}, \\
            ZY &= \frac{p_{y, |0\rangle} - p_{y, |1\rangle}}{2}, \\
            ZZ &= \frac{p_{z, |0\rangle} - p_{z, |1\rangle}}{2}, \\
            IX &= \frac{p_{x, |0\rangle} + p_{x, |1\rangle}}{2}, \\
            IY &= \frac{p_{y, |0\rangle} + p_{y, |1\rangle}}{2}, \\
            IZ &= \frac{p_{z, |0\rangle} + p_{z, |1\rangle}}{2},

        where :math:`p_{\beta, |j\rangle}` is a fit parameter of :class:`.BlochTrajectoryAnalysis`
        for the projection axis :math:`\beta` with the control qubit state :math:`|j\rangle`.

    # section: see_also

        qiskit_experiments.curve_analysis.standard_analysis.BlochTrajectoryAnalysis

    """

    def __init__(self):
        analyses = []
        for control_state in (0, 1):
            analysis = curve.BlochTrajectoryAnalysis(name=f"ctrl{control_state}")
            analysis.set_options(filter_data={"control_state": control_state})
            analyses.append(analysis)

        super().__init__(analyses=analyses)

    @classmethod
    def _default_options(cls):
        """Return the default analysis options."""
        default_options = super()._default_options()
        default_options.curve_drawer.set_options(
            subplots=(3, 1),
            xlabel="Flat top width",
            ylabel=[
                r"$\langle$X(t)$\rangle$",
                r"$\langle$Y(t)$\rangle$",
                r"$\langle$Z(t)$\rangle$",
            ],
            xval_unit="s",
            figsize=(8, 10),
            legend_loc="lower right",
            fit_report_rpos=(0.28, -0.10),
            ylim=(-1, 1),
            plot_options={
                "x_ctrl0": {"color": "blue", "symbol": "o", "canvas": 0},
                "y_ctrl0": {"color": "blue", "symbol": "o", "canvas": 1},
                "z_ctrl0": {"color": "blue", "symbol": "o", "canvas": 2},
                "x_ctrl1": {"color": "red", "symbol": "^", "canvas": 0},
                "y_ctrl1": {"color": "red", "symbol": "^", "canvas": 1},
                "z_ctrl1": {"color": "red", "symbol": "^", "canvas": 2},
            },
        )

        return default_options

    def _create_analysis_results(
        self,
        fit_data: Dict[str, curve.CurveFitResult],
        quality: str,
        **metadata,
    ) -> List[AnalysisResultData]:
        outcomes = []

        for control in ("z", "i"):
            for target in ("x", "y", "z"):
                p0_val = fit_data["ctrl0"].ufloat_params[f"p{target}"]
                p1_val = fit_data["ctrl1"].ufloat_params[f"p{target}"]

                if control == "z":
                    coef_val = 0.5 * (p0_val - p1_val) / (2 * np.pi)
                else:
                    coef_val = 0.5 * (p0_val + p1_val) / (2 * np.pi)

                outcomes.append(
                    AnalysisResultData(
                        name=f"omega_{control}{target}",
                        value=coef_val,
                        quality=quality,
                        extra={
                            "unit": "Hz",
                            **metadata,
                        },
                    )
                )

        return outcomes
