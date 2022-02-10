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

from typing import Tuple

import numpy as np

from qiskit_experiments.curve_analysis import ErrorAmplificationAnalysis, ParameterRepr
from qiskit_experiments.exceptions import AnalysisError
from qiskit_experiments.framework import (
    CompositeAnalysis,
    ExperimentData,
    AnalysisResultData,
    Options,
)
from qiskit_experiments.data_processing import DataProcessor, Probability


class HeatElementAnalysis(ErrorAmplificationAnalysis):
    """An analysis class for HEAT experiment to define the fixed parameters.

    # section: note

        This analysis assumes the experiment measures only single qubit
        regardless of the number of physical qubits used in the experiment.

    # section: overview

        This is standard error amplification analysis.

    # section: see_also
        qiskit_experiments.curve_analysis.ErrorAmplificationAnalysis
    """

    __fixed_parameters__ = ["angle_per_gate", "phase_offset", "amp"]

    @classmethod
    def _default_options(cls) -> Options:
        """Default analysis options."""
        options = super()._default_options()
        options.angle_per_gate = np.pi
        options.phase_offset = np.pi / 2
        options.amp = 1.0
        options.data_processor = DataProcessor(input_key="counts", data_actions=[Probability("1")])

        return options


class HeatAnalysis(CompositeAnalysis):
    r"""A composite error amplification analysis to get unitary error coefficients.

    # section: fit_model

        Heat experiment amplifies the dynamics of the entangling gate along the
        experiment-specific error axis on the target qubit Bloch sphere.
        This analysis takes two error amplification experiment results performed with
        different states of the control qubit to distinguish the contribution
        of local term (such as IX) from non-local term (such as ZX).

        This analysis takes a set of `d_theta` parameters from child error amplification results
        which might be represented by a unique name in the child experiment data.
        With these fit parameters, two Hamiltonian coefficients will be computed as

        .. math::

            A_{I\beta} = \frac{1}{2}\left( d\theta_{\beta 0} + d\theta_{\beta 1} \right) \\

            A_{Z\beta} = \frac{1}{2}\left( d\theta_{\beta 0} - d\theta_{\beta 1} \right)

        where, :math:`\beta \in [X, Y, Z]` is a single-qubit Pauli term, and
        :math:`d\theta_{\beta k}` is an angle error ``d_theta`` extracted from the HEAT experiment
        with the control qubit in state :math:`|k\rangle \in [|0\rangle, |1\rangle]`.

    # section: see_also
        qiskit_experiments.library.hamiltonian.HeatElementAnalysis
        qiskit_experiments.curve_analysis.ErrorAmplificationAnalysis

    """

    def __init__(
        self,
        fit_params: Tuple[str, str],
        out_params: Tuple[str, str],
    ):
        """Create new HEAT analysis.

        Args:
            fit_params: Name of error parameters for each amplification sequence.
            out_params: Name of Hamiltonian coefficients.

        Raises:
            AnalysisError: When size of ``fit_params`` or ``out_params`` are not 2.
        """
        if len(fit_params) != 2:
            raise AnalysisError(
                f"{self.__class__.__name__} assumes two fit parameters extracted from "
                "a set of experiments with different control qubit state input. "
                f"{len(fit_params)} input parameter names are specified."
            )

        if len(out_params) != 2:
            raise AnalysisError(
                f"{self.__class__.__name__} assumes two output parameters computed with "
                "a set of experiment results with different control qubit state input. "
                f"{len(out_params)} output parameter names are specified."
            )

        analyses = []
        for fit_parm in fit_params:
            sub_analysis = HeatElementAnalysis()
            sub_analysis.set_options(result_parameters=[ParameterRepr("d_theta", fit_parm, "rad")])
            analyses.append(sub_analysis)

        super().__init__(analyses=analyses)

        self._fit_params = fit_params
        self._out_params = out_params

    def _run_analysis(self, experiment_data: ExperimentData):

        # wait for child experiments to complete
        super()._run_analysis(experiment_data)

        # extract d_theta parameters
        fit_results = []
        for i, pname in enumerate(self._fit_params):
            fit_results.append(experiment_data.child_data(i).analysis_results(pname))

        # Check data quality
        is_good_quality = all(r.quality == "good" for r in fit_results)

        estimate_ib = AnalysisResultData(
            name=self._out_params[0],
            value=(fit_results[0].value + fit_results[1].value) / 2,
            quality="good" if is_good_quality else "bad",
            extra={"unit": "rad"},
        )

        estimate_zb = AnalysisResultData(
            name=self._out_params[1],
            value=(fit_results[0].value - fit_results[1].value) / 2,
            quality="good" if is_good_quality else "bad",
            extra={"unit": "rad"},
        )

        return [estimate_ib, estimate_zb], []
