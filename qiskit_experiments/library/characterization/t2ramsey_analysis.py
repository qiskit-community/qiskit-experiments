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

from qiskit_experiments.curve_analysis import DumpedOscillationAnalysis, ParameterRepr
from qiskit_experiments.data_processing import DataProcessor, Probability

from qiskit_experiments.framework import Options


class T2RamseyAnalysis(DumpedOscillationAnalysis):
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
            ParameterRepr("freq", "Frequency", "Hz"),
            ParameterRepr("tau", "T2star", "s"),
        ]

        return options
