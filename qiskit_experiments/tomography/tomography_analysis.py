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
Quantum process tomography analysis
"""


from typing import List, Dict, Tuple, Union
import numpy as np

from qiskit.result import marginal_counts
from qiskit_experiments.exceptions import AnalysisError
from qiskit_experiments.base_analysis import BaseAnalysis, AnalysisResult, Options
from .basis import TomographyBasis, FitterBasis
from .fitters import (
    scipy_guassian_lstsq,
    linear_inversion,
    cvxpy_guassian_lstsq,
)


class TomographyAnalysis(BaseAnalysis):
    """Quantum state and process tomography experiment analysis."""

    @classmethod
    def _default_options(cls) -> Options:
        return Options(
            measurement_basis=None,
            preparation_basis=None,
            fitter="scipy_lstsq",
        )

    @staticmethod
    def _get_fitter(fitter):
        """Return fitter function for named builtin fitters"""
        if fitter is None:
            raise AnalysisError("No tomography fitter given")
        if not isinstance(fitter, str):
            return fitter

        # Lookup built-in fitters
        if fitter == "scipy_lstsq":
            return scipy_guassian_lstsq
        if fitter == "lininv":
            return linear_inversion
        if fitter == "cvxpy_lstsq":
            return cvxpy_guassian_lstsq
        raise AnalysisError(f"Unrecognized tomography fitter {fitter}")

    @staticmethod
    def _get_matrix_basis(basis: Union[TomographyBasis, FitterBasis]) -> FitterBasis:
        """Validate an input matrix basis."""
        if basis is None or isinstance(basis, FitterBasis):
            return basis
        if isinstance(basis, TomographyBasis):
            return basis.matrix
        raise AnalysisError("Invalid tomography FitterBasis")

    def _run_analysis(self, experiment_data, **options):
        # Extract tomography measurement data
        mbasis_data, pbasis_data, freq_data, shot_data = self._measurement_data(
            experiment_data.data()
        )

        # Get tomography bases
        measurement_basis = self._get_matrix_basis(options.pop("measurement_basis"))
        preparation_basis = self._get_matrix_basis(options.pop("preparation_basis"))

        # Get tomography fitter function
        fitter = self._get_fitter(options.pop("fitter", None))
        try:
            result = fitter(
                mbasis_data,
                pbasis_data,
                freq_data,
                shot_data,
                measurement_basis=measurement_basis,
                preparation_basis=preparation_basis,
                **options,
            )
            result["fitter"] = fitter.__name__

        except AnalysisError as ex:
            raise AnalysisError(f"Tomography fitter failed with error: {str(ex)}") from ex

        return [AnalysisResult(result)], [None]

    @staticmethod
    def _measurement_data(
        data: List[Dict[str, any]]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return list a tuple of basis, frequency, shot data"""
        freq_dict = {}
        shot_dict = {}
        meas_size = None
        prep_size = None

        for datum in data:
            metadata = datum["metadata"]
            counts = marginal_counts(datum["counts"], metadata["clbits"])
            shots = sum(counts.values())
            meas_element = tuple(metadata["m_idx"])
            prep_element = tuple(metadata["p_idx"]) if "p_idx" in metadata else tuple()

            if meas_size is None:
                meas_size = len(meas_element)
            if prep_size is None:
                prep_size = len(prep_element)

            # Get count data
            for key, freq in counts.items():
                outcome_element = list(meas_element)
                for i, outcome in enumerate(reversed(key)):
                    outcome_element[i] += int(outcome)
                element = tuple(outcome_element), prep_element
                if element in freq_dict:
                    freq_dict[element] += freq
                    shot_dict[element] += shots
                else:
                    freq_dict[element] = freq
                    shot_dict[element] = shots

        num_elements = len(freq_dict)

        meas_basis_data = np.zeros((num_elements, meas_size), dtype=int)
        prep_basis_data = np.zeros((num_elements, prep_size), dtype=int)
        freq_data = np.zeros(num_elements, dtype=int)
        shot_data = np.zeros(num_elements, dtype=int)

        for i, (key, val) in enumerate(freq_dict.items()):
            meas_basis_data[i] = key[0]
            prep_basis_data[i] = key[1]
            freq_data[i] = val
            shot_data[i] = shot_dict[key]
        return meas_basis_data, prep_basis_data, freq_data, shot_data
