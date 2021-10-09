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
Purity Estimation analysis.
"""

import numpy as np
from qiskit.result import marginal_counts
from qiskit_experiments.framework import BaseAnalysis, AnalysisResultData


class PurityEstimationAnalysis(BaseAnalysis):
    """Purity estimation analysis class."""

    @staticmethod
    def _hamming_dist(outcome1: str, outcome2: str) -> int:
        """Return the Hamming-distance between two bitstrings"""
        return bin(int(outcome1, 2) ^ int(outcome2, 2)).count("1")

    def _run_analysis(self, experiment_data):
        data = experiment_data.data()
        num_samples = len(data)
        purity_samples = np.zeros(num_samples, dtype=float)

        for idx, datum in enumerate(data):
            meas_clbits = datum["metadata"]["clbits"]
            counts = marginal_counts(datum["counts"], meas_clbits)
            n_sub = len(next(iter(counts)))
            shots = 0
            purity_k = 0

            # Compute purity component for given counts dict
            for i, ci in counts.items():
                shots += ci
                for j, cj in counts.items():
                    hwt = self._hamming_dist(i, j)
                    purity_k += (-2) ** (-hwt) * ci * cj
            purity_k *= (2 ** n_sub) / (shots ** 2)

            # Accumualte with average purity estimate
            purity_samples[idx] = purity_k

        # Compute purity estimate
        purity = np.sum(purity_samples) / num_samples

        # Create analysis result data structure
        result = AnalysisResultData(
            "purity",
            value=purity,
            extra={
                "num_samples": num_samples,
                "samples": purity_samples
            })

        return [result], []
