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
from qiskit.quantum_info import DensityMatrix, Choi, Operator
from qiskit.quantum_info.operators.base_operator import BaseOperator
from qiskit.quantum_info.operators.channel.quantum_channel import QuantumChannel

from qiskit_experiments.exceptions import AnalysisError
from qiskit_experiments.framework import BaseAnalysis, AnalysisResultData, Options
from .fitters import (
    linear_inversion,
    scipy_linear_lstsq,
    scipy_gaussian_lstsq,
    cvxpy_linear_lstsq,
    cvxpy_gaussian_lstsq,
)


class TomographyAnalysis(BaseAnalysis):
    """Base analysis for state and process tomography experiments."""

    _builtin_fitters = {
        "linear_inversion": linear_inversion,
        "scipy_linear_lstsq": scipy_linear_lstsq,
        "scipy_gaussian_lstsq": scipy_gaussian_lstsq,
        "cvxpy_linear_lstsq": cvxpy_linear_lstsq,
        "cvxpy_gaussian_lstsq": cvxpy_gaussian_lstsq,
    }

    @classmethod
    def _default_options(cls) -> Options:
        """Default analysis options

        Analysis Options:
            measurement_basis
                (:class:`~qiskit_experiments.library.tomography.basis.BaseFitterMeasurementBasis`):
                The measurement
                :class:`~qiskit_experiments.library.tomography.basis.BaseFitterMeasurementBasis`
                to use for tomographic reconstruction when running a
                :class:`~qiskit_experiments.library.tomography.StateTomography` or
                :class:`~qiskit_experiments.library.tomography.ProcessTomography`.
            preparation_basis
                (:class:`~qiskit_experiments.library.tomography.basis.BaseFitterPreparationBasis`):
                The preparation
                :class:`~qiskit_experiments.library.tomography.basis.BaseFitterPreparationBasis`
                to use for tomographic reconstruction for
                :class:`~qiskit_experiments.library.tomography.ProcessTomography`.
            fitter (str or Callable): The fitter function to use for reconstruction.
                This can  be a string to select one of the built-in fitters, or a callable to
                supply a custom fitter function. See the `Fitter Functions` section for
                additional information.
            fitter_options (dict): Any addition kwarg options to be supplied to the fitter
                function. For documentation of available kargs refer to the fitter function
                documentation.
            rescale_positive (bool): If True rescale the state returned by the fitter
                to be positive-semidefinite. See the `PSD Rescaling` section for
                additional information (Default: True).
            rescale_trace (bool): If True rescale the state returned by the fitter
                have either trace 1 for :class:`~qiskit.quantum_info.DensityMatrix`,
                or trace dim for :class:`~qiskit.quantum_info.Choi` matrices (Default: True).
            target (Any): depends on subclass.
        """
        options = super()._default_options()

        options.measurement_basis = None
        options.preparation_basis = None
        options.fitter = "linear_inversion"
        options.fitter_options = {}
        options.rescale_positive = True
        options.rescale_trace = True
        options.target = "default"
        return options

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

    def _run_analysis(self, experiment_data):
        # Extract tomography measurement data
        outcome_data, shot_data, measurement_data, preparation_data = self._fitter_data(
            experiment_data.data()
        )

        # Get target state
        target_state = self.options.target
        if target_state == "default":
            metadata = experiment_data.metadata
            target_state = metadata.get("target", None)

        # Get tomography fitter function
        fitter = self._get_fitter(self.options.fitter)
        try:
            t_fitter_start = time.time()
            state, fitter_metadata = fitter(
                outcome_data,
                shot_data,
                measurement_data,
                preparation_data,
                self.options.measurement_basis,
                self.options.preparation_basis,
                **self.options.fitter_options,
            )
            t_fitter_stop = time.time()
            if fitter_metadata is None:
                fitter_metadata = {}
            state = Choi(state) if self.options.preparation_basis else DensityMatrix(state)
            fitter_metadata["fitter"] = fitter.__name__
            fitter_metadata["fitter_time"] = t_fitter_stop - t_fitter_start

            analysis_results = self._postprocess_fit(
                state,
                metadata=fitter_metadata,
                target_state=target_state,
                rescale_positive=self.options.rescale_positive,
                rescale_trace=self.options.rescale_trace,
                qpt=bool(self.options.preparation_basis),
            )

        except AnalysisError as ex:
            raise AnalysisError(f"Tomography fitter failed with error: {str(ex)}") from ex

        return analysis_results, []

    @classmethod
    def _postprocess_fit(
        cls,
        state,
        metadata=None,
        target_state=None,
        rescale_positive=False,
        rescale_trace=False,
        qpt=False,
    ):
        """Post-process fitter data"""
        # Get eigensystem of state
        state_cls = type(state)
        evals, evecs = cls._state_eigensystem(state)

        # Rescale eigenvalues to be PSD
        rescaled_psd = False
        if rescale_positive and np.any(evals < 0):
            scaled_evals = cls._make_positive(evals)
            rescaled_psd = True
        else:
            scaled_evals = evals

        # Rescale trace
        trace = np.sqrt(len(scaled_evals)) if qpt else 1
        sum_evals = np.sum(scaled_evals)
        rescaled_trace = False
        if rescale_trace and not np.isclose(sum_evals - trace, 0, atol=1e-12):
            scaled_evals = trace * scaled_evals / sum_evals
            rescaled_trace = True

        # Compute state with rescaled eigenvalues
        state_result = AnalysisResultData("state", state, extra=metadata)
        state_result.extra["eigvals"] = scaled_evals
        if rescaled_psd or rescaled_trace:
            state = state_cls(evecs @ (scaled_evals * evecs).T.conj())
            state_result.value = state
            state_result.extra["raw_eigvals"] = evals

        if rescaled_trace:
            state_result.extra["trace"] = np.sum(scaled_evals)
            state_result.extra["raw_trace"] = sum_evals
        else:
            state_result.extra["trace"] = sum_evals

        # Results list
        analysis_results = [state_result]

        # Compute fidelity with target
        if target_state is not None:
            analysis_results.append(
                cls._fidelity_result(scaled_evals, evecs, target_state, qpt=qpt)
            )

        # Check positive
        analysis_results.append(cls._positivity_result(scaled_evals, qpt=qpt))

        # Check trace preserving
        if qpt:
            analysis_results.append(cls._tp_result(scaled_evals, evecs))

        return analysis_results

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
    def _positivity_result(evals, qpt=False):
        """Check if eigenvalues are positive"""
        cond = np.sum(np.abs(evals[evals < 0]))
        is_pos = bool(np.isclose(cond, 0))
        name = "completely_positive" if qpt else "positive"
        result = AnalysisResultData(name, is_pos)
        if not is_pos:
            result.extra = {"delta": cond}
        return result

    @staticmethod
    def _tp_result(evals, evecs):
        """Check if QPT channel is trace preserving"""
        size = len(evals)
        dim = int(np.sqrt(size))
        mats = np.reshape(evecs.T, (size, dim, dim), order="F")
        kraus_cond = np.einsum("i,ija,ijb->ab", evals, mats.conj(), mats)
        cond = np.sum(np.abs(la.eigvalsh(kraus_cond - np.eye(dim))))
        is_tp = bool(np.isclose(cond, 0))
        result = AnalysisResultData("trace_preserving", is_tp)
        if not is_tp:
            result.extra = {"delta": cond}
        return result

    @staticmethod
    def _fidelity_result(evals, evecs, target, qpt=False):
        """Faster computation of fidelity from eigen decomposition"""
        # Format target to statevector or densitymatrix array
        trace = np.sqrt(len(evals)) if qpt else 1
        name = "process_fidelity" if qpt else "state_fidelity"
        if target is None:
            raise AnalysisError("No target state provided")
        if isinstance(target, QuantumChannel):
            target_state = Choi(target).data / trace
        elif isinstance(target, BaseOperator):
            target_state = np.ravel(Operator(target), order="F") / np.sqrt(trace)
        else:
            target_state = np.array(target)

        if target_state.ndim == 1:
            rho = evecs @ (evals / trace * evecs).T.conj()
            fidelity = np.real(target_state.conj() @ rho @ target_state)
        else:
            sqrt_rho = evecs @ (np.sqrt(evals / trace) * evecs).T.conj()
            eig = la.eigvalsh(sqrt_rho @ target_state @ sqrt_rho)
            fidelity = np.sum(np.sqrt(np.maximum(eig, 0))) ** 2
        return AnalysisResultData(name, fidelity)

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
