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
T1 Analysis class.
"""
from typing import Union, List

import qiskit_experiments.curve_analysis as curve

from qiskit_experiments.framework import Options


class T1Analysis(curve.DecayAnalysis):
    r"""A class to analyze T1 experiments.

    # section: see_also
        qiskit_experiments.curve_analysis.standard_analysis.decay.DecayAnalysis

    """

    @classmethod
    def _default_options(cls) -> Options:
        """Default analysis options."""
        options = super()._default_options()
        options.xlabel = "Delay"
        options.ylabel = "P(1)"
        options.xval_unit = "s"
        options.result_parameters = [curve.ParameterRepr("tau", "T1", "s")]

        return options

    def _generate_fit_guesses(
        self, user_opt: curve.FitOptions
    ) -> Union[curve.FitOptions, List[curve.FitOptions]]:
        """Apply conversion factor to tau."""

        extra = self._get_option("extra")

        conversion_factor = extra.get("conversion_factor", 1)
        user_opt.p0["tau"] *= conversion_factor

        return super()._generate_fit_guesses(user_opt)
