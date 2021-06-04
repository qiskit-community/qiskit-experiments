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
import time
import numpy as np

from qiskit.result import marginal_counts
from qiskit.quantum_info import state_fidelity, process_fidelity
from qiskit.quantum_info.states.quantum_state import QuantumState
from qiskit_experiments.exceptions import AnalysisError
from qiskit_experiments.base_analysis import BaseAnalysis, AnalysisResult, Options
from .basis import TomographyBasis, FitterBasis
from .fitters import (
    linear_inversion,
    scipy_linear_lstsq,
    scipy_gaussian_lstsq,
    cvxpy_linear_lstsq,
    cvxpy_gaussian_lstsq,
)


class TomographyAnalysis(BaseAnalysis):
    """Quantum state and process tomography experiment analysis.

    **Analysis Options**

    The tomography analysis class supports the following options

    * ``measurement_basis``: The measurement
      :class:`~qiskit_experiments.tomography.basis.FitterBasis` to use for
      tomographic reconstruction when running a
      :class:`~qiskit_experiments.tomography.StateTomographyExperiment` or
      :class:`~qiskit_experiments.tomography.ProcessTomographyExperiment`.
    * ``preparation_basis``: the preparation
      :class:`~qiskit_experiments.tomography.basis.FitterBasis` to
      use for tomographic reconstruction for
      :class:`~qiskit_experiments.tomography.ProcessTomographyExperiment`.
    * ``fitter``: The fitter function to use for reconstruction. This can
      be a string to select one of the built-in fitters, or a callable to
      supply a custom fitter function.
    * Additional kwargs will be supplied to the fitter function,
      for documentation of available args refer to the fitter function
      documentation.

    **Fitter Functions**

    Built-in fitter functions may be selected using the following string
    labels, refer to the corresponding functions documentation for additional
    details on the fitters.

    * ``"lininv"``: :func:`~qiskit_experiments.tomography.fitters.linear_inversion` (Default)
    * ``"scipy_lstsq"``: :func:`~qiskit_experiments.tomography.fitters.scipy_linear_lstsq`
    * ``"scipy_glstsq"``: :func:`~qiskit_experiments.tomography.fitters.scipy_gaussian_lstsq`
    * ``"cvxpy_lstsq"``: :func:`~qiskit_experiments.tomography.fitters.cvxpy_linear_lstsq`
    * ``"cvxpy_glstsq"``: :func:`~qiskit_experiments.tomography.fitters.cvxpy_gaussian_lstsq`

    .. note::

        Fitters starting with ``"cvxpy_*"`` require the optional CVXPY Python
        package to be installed.

    A custom fitter function must have signature

    .. code::

        fitter(measurement_data: np.ndarray,
               preparation_data: np.ndarray,
               frequency_data: np.ndarray,
               shot_data: np.ndarray,
               measurement_basis: Optional[FitterBasis] = None,
               preparation_basis: Optional[FitterBasis] = None,
               **kwargs) -> Dict[str, any]
    """

    @classmethod
    def _default_options(cls) -> Options:
        return Options(
            measurement_basis=None,
            preparation_basis=None,
            fitter="lininv"
        )

    @staticmethod
    def _get_fitter(fitter):
        """Return fitter function for named builtin fitters"""
        if fitter is None:
            raise AnalysisError("No tomography fitter given")
        if not isinstance(fitter, str):
            return fitter

        # Lookup built-in fitters
        if fitter == "lininv":
            return linear_inversion
        if fitter == "scipy_lstsq":
            return scipy_linear_lstsq
        if fitter == "scipy_glstsq":
            return scipy_gaussian_lstsq
        if fitter == "cvxpy_lstsq":
            return cvxpy_linear_lstsq
        if fitter == "cvxpy_glstsq":
            return cvxpy_gaussian_lstsq
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
            t_start = time.time()
            result = fitter(
                mbasis_data,
                pbasis_data,
                freq_data,
                shot_data,
                measurement_basis=measurement_basis,
                preparation_basis=preparation_basis,
                **options,
            )
            t_stop = time.time()
            result["fitter"] = fitter.__name__
            result["fitter_time"] = t_stop - t_start

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
