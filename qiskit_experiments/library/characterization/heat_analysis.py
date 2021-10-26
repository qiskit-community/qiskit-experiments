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
Analysis for HEAT experiments.
"""

from abc import ABC

import numpy as np

from qiskit_experiments.curve_analysis import ErrorAmplificationAnalysis
from qiskit_experiments.exceptions import AnalysisError
from qiskit_experiments.framework import (
    CompositeAnalysis,
    CompositeExperimentData,
    AnalysisResultData,
    Options,
    FitVal,
)


class HeatAnalysis(ErrorAmplificationAnalysis):
    """An analysis class for HEAT experiment to define the fixed parameters."""

    __fixed_parameters__ = ["angle_per_gate", "phase_offset", "amp"]

    @classmethod
    def _default_options(cls) -> Options:
        """Default analysis options."""
        options = super()._default_options()
        options.angle_per_gate = np.pi
        options.phase_offset = np.pi / 2
        options.amp = 1.0

        return options


class CompositeHeatAnalysis(CompositeAnalysis, ABC):
    r"""A composite error amplification analysis to get unitary error coefficients.

    # section: fit_model

        This analysis takes a set of `d_theta` parameters from two error amplification
        analysis results. Each parameter is extracted from the HEAT experiment with
        different control qubit states. Namely,

        .. math::

            A_{I\beta} = \frac{{d\theta_{\beta 0}} + d\theta_{\beta 1}}}{2}

            A_{Z\beta} = \frac{{d\theta_{\beta 0}} - d\theta_{\beta 1}}}{2}

        where, :math:`\beta \in [X, Y, Z]` is one of single qubit Pauli terms,
        :math:`d\theta_{\beta k}` is `d_theta` parameter extracted from the HEAT experiment
        with the control qubit state :math:`|k\rangle \in [|0\rangle, |1\rangle]`.

    # section: see_also
        qiskit_experiments.curve_analysis.standard_analysis.error_amplification_analysis.\
        ErrorAmplificationAnalysis

    """
    __fit_params__ = []
    __out_params__ = []

    def _run_analysis(self, experiment_data: CompositeExperimentData, **options):

        # Validate setup
        if len(self.__fit_params__) != 2:
            raise AnalysisError(
                f"{self.__class__.__name__} assumes two fit parameters extracted from "
                "a set of experiments with different control qubit state input. "
                f"{len(self.__fit_params__)} input parameter names are specified."
            )

        if len(self.__out_params__) != 2:
            raise AnalysisError(
                f"{self.__class__.__name__} assumes two output parameters computed with "
                "a set of experiment results with different control qubit state input. "
                f"{len(self.__out_params__)} output parameter names are specified."
            )

        # Create analysis data of nested experiment and discard redundant entry.
        # Note that experiment_data is mutable.
        super()._run_analysis(experiment_data, **options)

        sub_analysis_results = [
            experiment_data.component_experiment_data(i).analysis_results(pname)
            for i, pname in enumerate(self.__fit_params__)
        ]

        # Check data quality
        is_good_quality = all(r.quality == "good" for r in sub_analysis_results)

        # Compute unitary terms
        ib = (sub_analysis_results[0].value.value + sub_analysis_results[1].value.value) / 2
        zb = (sub_analysis_results[0].value.value - sub_analysis_results[1].value.value) / 2

        # Compute new variance
        sigma = np.sqrt(
            sub_analysis_results[0].value.stderr ** 2 + sub_analysis_results[1].value.stderr ** 2
        )

        estimate_ib = AnalysisResultData(
            name=self.__out_params__[0],
            value=FitVal(value=ib, stderr=sigma, unit="rad"),
            quality="good" if is_good_quality else "bad",
        )

        estimate_zb = AnalysisResultData(
            name=self.__out_params__[1],
            value=FitVal(value=zb, stderr=sigma, unit="rad"),
            quality="good" if is_good_quality else "bad",
        )

        composite_analysis_results = [estimate_ib, estimate_zb]

        return composite_analysis_results, None


class HeatYAnalysis(CompositeHeatAnalysis):
    """"""
    __fit_params__ = ["d_heat_y0", "d_heat_y1"]
    __out_params__ = ["A_iy", "A_zy"]


class HeatZAnalysis(CompositeAnalysis):
    """"""
    __fit_params__ = ["d_heat_z0", "d_heat_z1"]
    __out_params__ = ["A_iz", "A_zz"]
