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


from typing import List, Dict, Tuple
import time
import numpy as np
import scipy.linalg as la

from qiskit.result import marginal_counts, Counts
from qiskit.quantum_info import Choi, Operator
from qiskit.quantum_info.operators.base_operator import BaseOperator
from qiskit.quantum_info.operators.channel.quantum_channel import QuantumChannel

from qiskit_experiments.exceptions import AnalysisError
from qiskit_experiments.base_analysis import BaseAnalysis, AnalysisResult, Options
from .fitters import (
    linear_inversion,
    scipy_linear_lstsq,
    scipy_gaussian_lstsq,
    cvxpy_linear_lstsq,
    cvxpy_gaussian_lstsq,
)


class TomographyAnalysis(BaseAnalysis):
    """Quantum state and process tomography experiment analysis.

    Analysis Options
        - **measurement_basis**
          (:class:`~qiskit_experiments.tomography.basis.BaseFitterMeasurementBasis`):
          The measurement
          :class:`~qiskit_experiments.tomography.basis.BaseFitterMeasurementBasis`
          to use for tomographic reconstruction when running a
          :class:`~qiskit_experiments.tomography.StateTomographyExperiment` or
          :class:`~qiskit_experiments.tomography.ProcessTomographyExperiment`.
        - **preparation_basis**
          (:class:`~qiskit_experiments.tomography.basis.BaseFitterPreparationBasis`):
          The preparation
          :class:`~qiskit_experiments.tomography.basis.BaseFitterPreparationBasis`
          to use for tomographic reconstruction for
          :class:`~qiskit_experiments.tomography.ProcessTomographyExperiment`.
        - **fitter** (``str`` or ``Callable``): The fitter function to use for reconstruction.
          This can  be a string to select one of the built-in fitters, or a callable to
          supply a custom fitter function. See the `Fitter Functions` section
          for additional information.
        - **rescale_psd** (``bool``): If True rescale the state returned by the fitter to be
          positive-semidefinite. See the `PSD Rescaling` section for
          additional information (Default: True).
        - **rescale_trace** (``bool``): If True rescale the state returned by the fitter
          have either trace 1 for :class:`~qiskit.quantum_info.DensityMatrix`,
          or trace dim for :class:`~qiskit.quantum_info.Choi`.
          matrices (Default: True).
        - **kwargs**: will be supplied to the fitter function,
          for documentation of available args refer to the fitter function
          documentation.

    Fitter Functions
        Built-in fitter functions may be selected using the following string
        labels, refer to the corresponding functions documentation for additional
        details on the fitters.

        * ``"linear_inversion"``:
          :func:`~qiskit_experiments.tomography.fitters.linear_inversion` (Default)
        * ``"scipy_linear_lstsq"``:
          :func:`~qiskit_experiments.tomography.fitters.scipy_linear_lstsq`
        * ``"cvxpy_linear_lstsq"``:
          :func:`~qiskit_experiments.tomography.fitters.cvxpy_linear_lstsq`
        * ``"scipy_gaussian_lstsq"``:
          :func:`~qiskit_experiments.tomography.fitters.scipy_gaussian_lstsq`
        * ``"cvxpy_gaussian_lstsq"``:
          :func:`~qiskit_experiments.tomography.fitters.cvxpy_gaussian_lstsq`

        .. note::

            Fitters starting with ``"cvxpy_*"`` require the optional CVXPY Python
            package to be installed.

        A custom fitter function must have signature

        .. code:: python

            fitter(outcome_data: List[np.ndarray],
                shot_data: np.ndarray,
                measurement_data: np.ndarray,
                preparation_data: np.ndarray,
                measurement_basis: BaseFitterMeasurementBasis,
                preparation_basis: Optional[BaseFitterPreparationBasis] = None,
                **kwargs) -> Dict[str, any]

    PSD Rescaling
        For fitters that do not constrain the reconstructed state to be
        `positive-semidefinite` (PSD) we construct the maximum-likelihood
        nearest PSD state under the assumption of Gaussian measurement noise
        using the rescaling method in Reference [1]. For fitters that already
        support PSD constraints this option can be disabled by setting
        ``rescale_psd=False``.

    References
        1. J Smolin, JM Gambetta, G Smith, Phys. Rev. Lett. 108, 070502 (2012).
           Open access: https://arxiv.org/abs/arXiv:1106.5458
    """

    _builtin_fitters = {
        "linear_inversion": linear_inversion,
        "scipy_linear_lstsq": scipy_linear_lstsq,
        "scipy_gaussian_lstsq": scipy_gaussian_lstsq,
        "cvxpy_linear_lstsq": cvxpy_linear_lstsq,
        "cvxpy_gaussian_lstsq": cvxpy_gaussian_lstsq,
    }

    @classmethod
    def _default_options(cls) -> Options:
        return Options(
            measurement_basis=None,
            preparation_basis=None,
            fitter="linear_inversion",
            rescale_psd=True,
            rescale_trace=True,
        )

    @classmethod
    def _get_fitter(cls, fitter):
        """Return fitter function for named builtin fitters"""
        if fitter is None:
            raise AnalysisError("No tomography fitter given")
        if not isinstance(fitter, str):
            return fitter
        if fitter in cls._builtin_fitters:
            return cls._builtin_fitters[fitter]
        raise AnalysisError(f"Unrecognized tomography fitter {fitter}")

    def _run_analysis(self, experiment_data, **options):
        # Extract tomography measurement data
        outcome_data, shot_data, measurement_data, preparation_data = self._fitter_data(
            experiment_data.data()
        )

        # Get tomography options
        measurement_basis = options.pop("measurement_basis")
        preparation_basis = options.pop("preparation_basis")
        rescale_psd = options.pop("rescale_psd")
        rescale_trace = options.pop("rescale_trace")

        # Get target state
        metadata = experiment_data.metadata()
        target_state = metadata.get("target_state", None)

        # Get tomography fitter function
        fitter = self._get_fitter(options.pop("fitter", None))
        try:
            t_start = time.time()
            result = fitter(
                outcome_data,
                shot_data,
                measurement_data,
                preparation_data,
                measurement_basis,
                preparation_basis=preparation_basis,
                **options,
            )
            t_stop = time.time()
            result["fitter"] = fitter.__name__
            result["fitter_time"] = t_stop - t_start

            self._postprocess_fit(
                result,
                target_state=target_state,
                rescale_psd=rescale_psd,
                rescale_trace=rescale_trace,
                qpt=bool(preparation_basis),
            )

        except AnalysisError as ex:
            raise AnalysisError(f"Tomography fitter failed with error: {str(ex)}") from ex

        return [AnalysisResult(result)], [None]

    @classmethod
    def _postprocess_fit(
        cls, analysis_result, target_state=None, rescale_psd=False, rescale_trace=False, qpt=False
    ):
        """Post-process fitter data"""
        # Get eigensystem of state
        state = analysis_result["state"]
        evals, evecs = cls._state_eigensystem(state)

        # Rescale eigenvalues to be PSD and specified trace
        rescaled = False
        if rescale_psd and np.any(evals < 0):
            scaled_evals = cls._make_positive(evals)
            rescaled = True
        else:
            scaled_evals = evals
        if rescale_trace:
            trace = np.sqrt(len(scaled_evals)) if qpt else 1
            scaled_evals = trace * scaled_evals / np.sum(scaled_evals)
            rescaled = True

        # Compute state with rescaled eigenvalues
        if rescaled:
            scaled_state = evecs @ (scaled_evals * evecs).T.conj()
            analysis_result["state"] = type(state)(scaled_state)
            analysis_result["state_eigvals"] = scaled_evals
            analysis_result["raw_eigvals"] = evals
        else:
            analysis_result["state_eigvals"] = evals

        # Compute fidelity with target
        if target_state is None:
            analysis_result["value"] = None
            analysis_result["value_label"] = None
        else:
            analysis_result["value"] = cls._state_fidelity(scaled_evals, evecs, target_state)
            if isinstance(target_state, QuantumChannel):
                analysis_result["value_label"] = "process_fidelity"
            else:
                analysis_result["value_label"] = "state_fidelity"

        return analysis_result

    @staticmethod
    def _state_eigensystem(state):
        evals, evecs = la.eigh(state)
        # Truncate eigenvalues to real part
        evals = np.real(evals)
        # Sort eigensystem from largest to smallest eigenvalues
        sort_inds = np.flip(np.argsort(evals))
        return evals[sort_inds], evecs[:, sort_inds]

    @staticmethod
    def _make_positive(evals, epsilon=0):
        if epsilon < 0:
            raise AnalysisError("epsilon must be non-negative.")
        ret = evals.copy()
        dim = len(evals)
        idx = dim - 1
        accum = 0.0
        while idx >= 0:
            shift = accum / (idx + 1)
            if evals[idx] + shift < epsilon:
                ret[idx] = 0
                accum = accum + evals[idx]
                idx -= 1
            else:
                for j in range(idx + 1):
                    ret[j] = evals[j] + shift
                break
        return ret

    @staticmethod
    def _state_fidelity(evals, evecs, target, qpt=False):
        """Faster computation of fidelity from eigen decomposition"""
        # Format target to statevector or densitymatrix array
        trace = np.sqrt(len(evals)) if qpt else 1
        if isinstance(target, QuantumChannel):
            target_state = Choi(target).data / trace
        elif isinstance(target, BaseOperator):
            target_state = np.ravel(Operator(target), order="F") / np.sqrt(trace)
        else:
            target_state = np.array(target)

        if target_state.ndim == 1:
            rho = evecs @ (evals / trace * evecs).T.conj()
            return np.real(target_state.conj() @ rho @ target_state)
        else:
            sqrt_rho = evecs @ (np.sqrt(evals / trace) * evecs).T.conj()
            eig = la.eigvalsh(sqrt_rho @ target_state @ sqrt_rho)
            return np.sum(np.sqrt(np.maximum(eig, 0))) ** 2

    @staticmethod
    def _fitter_data(
        data: List[Dict[str, any]]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[np.ndarray]]:
        """Return list a tuple of basis, frequency, shot data"""
        outcome_dict = {}
        meas_size = None
        prep_size = None

        for datum in data:
            # Get basis data
            metadata = datum["metadata"]
            meas_element = tuple(metadata["m_idx"])
            prep_element = tuple(metadata["p_idx"]) if "p_idx" in metadata else tuple()
            if meas_size is None:
                meas_size = len(meas_element)
            if prep_size is None:
                prep_size = len(prep_element)

            # Add outcomes
            counts = Counts(marginal_counts(datum["counts"], metadata["clbits"])).int_outcomes()
            basis_key = (meas_element, prep_element)
            if basis_key in outcome_dict:
                TomographyAnalysis._append_counts(outcome_dict[basis_key], counts)
            else:
                outcome_dict[basis_key] = counts

        num_basis = len(outcome_dict)
        measurement_data = np.zeros((num_basis, meas_size), dtype=int)
        preparation_data = np.zeros((num_basis, prep_size), dtype=int)
        shot_data = np.zeros(num_basis, dtype=int)
        outcome_data = []

        for i, (basis_key, counts) in enumerate(outcome_dict.items()):
            measurement_data[i] = basis_key[0]
            preparation_data[i] = basis_key[1]
            outcome_arr = np.zeros((len(counts), 2), dtype=int)
            for j, (outcome, freq) in enumerate(counts.items()):
                outcome_arr[j] = [outcome, freq]
                shot_data[i] += freq
            outcome_data.append(outcome_arr)
        return outcome_data, shot_data, measurement_data, preparation_data

    @staticmethod
    def _append_counts(counts1, counts2):
        for key, val in counts2:
            if key in counts1:
                counts1[key] += val
            else:
                counts1[key] = val
        return counts1
