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
T2Ramsey Experiment class.
"""
from typing import Union, List

from qiskit_experiments.data_processing import DataProcessor, Probability
import qiskit_experiments.curve_analysis as curve


from qiskit_experiments.framework import Options


class T2RamseyAnalysis(curve.DumpedOscillationAnalysis):
    """T2 Ramsey result analysis class.

    # section: see_also
        qiskit_experiments.curve_analysis.standard_analysis.oscillation.DumpedOscillationAnalysis

    """

    @classmethod
    def _default_options(cls) -> Options:
        """Default analysis options."""
        options = super()._default_options()
        options.data_processor = DataProcessor(
            input_key="counts", data_actions=[Probability(outcome="0")]
        )
        options.xlabel = "Delay"
        options.ylabel = "P(0)"
        options.xval_unit = "s"
        options.result_parameters = [
            curve.ParameterRepr("freq", "Frequency", "Hz"),
            curve.ParameterRepr("tau", "T2star", "s"),
        ]

        return options

    def _generate_fit_guesses(
        self, user_opt: curve.FitOptions
    ) -> Union[curve.FitOptions, List[curve.FitOptions]]:
        """Apply conversion factor to tau."""
        extra = self._get_option("extra")

        conversion_factor = extra.get("conversion_factor", 1)
        user_opt.p0["tau"] *= conversion_factor

        return super()._generate_fit_guesses(user_opt)
