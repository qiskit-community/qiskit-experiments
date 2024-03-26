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


from typing import List, Union, Callable
from collections import defaultdict
import numpy as np
import scipy.linalg as la
from uncertainties import ufloat

from qiskit.quantum_info import DensityMatrix, Choi, Operator
from qiskit.quantum_info.operators.base_operator import BaseOperator
from qiskit.quantum_info.operators.channel.quantum_channel import QuantumChannel

from qiskit_experiments.exceptions import AnalysisError
from qiskit_experiments.framework import BaseAnalysis, AnalysisResultData, Options
from qiskit_experiments.framework.package_deps import version_is_at_least
from .fitters import (
    tomography_fitter_data,
    postprocess_fitter,
    linear_inversion,
    cvxpy_linear_lstsq,
    cvxpy_gaussian_lstsq,
)


class TomographyAnalysis(BaseAnalysis):
    """Base analysis for state and process tomography experiments."""

    _builtin_fitters = {
        "linear_inversion": linear_inversion,
        "cvxpy_linear_lstsq": cvxpy_linear_lstsq,
        "cvxpy_gaussian_lstsq": cvxpy_gaussian_lstsq,
    }
    _cvxpy_fitters = (
        cvxpy_linear_lstsq,
        cvxpy_gaussian_lstsq,
    )

    @classmethod
    def _default_options(cls) -> Options:
        """Default analysis options

        Analysis Options:
            measurement_basis
                (:class:`~qiskit_experiments.library.tomography.basis.MeasurementBasis`):
                The measurement
                :class:`~qiskit_experiments.library.tomography.basis.MeasurementBasis`
                to use for tomographic reconstruction when running a
                :class:`~qiskit_experiments.library.tomography.StateTomography` or
                :class:`~qiskit_experiments.library.tomography.ProcessTomography`.
            preparation_basis
                (:class:`~qiskit_experiments.library.tomography.basis.PreparationBasis`):
                The preparation
                :class:`~qiskit_experiments.library.tomography.basis.PreparationBasis`
                to use for tomographic reconstruction for
                :class:`~qiskit_experiments.library.tomography.ProcessTomography`.
            fitter (str or Callable): The fitter function to use for reconstruction.
                This can  be a string to select one of the built-in fitters, or a callable to
                supply a custom fitter function. See the `Fitter Functions` section for
                additional information.
            fitter_options (dict): Any addition kwarg options to be supplied to the fitter
                function. For documentation of available kwargs refer to the fitter function
                documentation.
            rescale_positive (bool): If True rescale the state returned by the fitter
                to be positive-semidefinite. See the `PSD Rescaling` section for
                additional information (Default: True).
            rescale_trace (bool): If True rescale the state returned by the fitter
                have either trace 1 for :class:`~qiskit.quantum_info.DensityMatrix`,
                or trace dim for :class:`~qiskit.quantum_info.Choi` matrices (Default: True).
            measurement_qubits (Sequence[int]): Optional, the physical qubits with tomographic
                measurements. If not specified will be set to ``[0, ..., N-1]`` for N-qubit
                tomographic measurements.
            preparation_qubits (Sequence[int]): Optional, the physical qubits with tomographic
                preparations. If not specified will be set to ``[0, ..., N-1]`` for N-qubit
                tomographic preparations.
            target (Any): Optional, target object for fidelity comparison of the fit
                (Default: None).
            target_bootstrap_samples (int): Optional, number of outcome re-samples to draw
                from measurement data for each basis for computing a bootstrapped standard
                error of fidelity with the target state. If 0 no bootstrapping will be
                performed and the target fidelity will not include a standard error
                (Default: 0).
            target_bootstrap_seed (int | None | Generator): Optional, RNG seed or
                Generator to use for bootstrapping data for boostrapped fidelity
                standard error calculation (Default: None).
            conditional_circuit_clbits (list[int]): Optional, the clbit indices in the
                source circuit to be conditioned on when reconstructing the state.
                Enabling this will return a list of reconstructed state components
                conditional on the values of these clbit values. The integer value of the
                conditioning clbits is stored in state analysis result extra field
                `"conditional_circuit_outcome"`.
            conditional_measurement_indices (list[int]): Optional, indices of tomography
                measurement qubits to used for conditional state reconstruction. Enabling
                this will return a list of reconstructed state components conditioned on
                the remaining tomographic bases conditional on the basis index, and outcome
                value for these measurements. The conditional measurement basis index and
                integer value of the measurement outcome is stored in state analysis result
                extra fields `"conditional_measurement_index"` and
                `"conditional_measurement_outcome"` respectively.
            conditional_preparation_indices (list[int]): Optional, indices of tomography
                preparation qubits to used for conditional state reconstruction. Enabling
                this will return a list of reconstructed channel components conditioned on
                the remaining tomographic bases conditional on the basis index. The
                conditional preparation basis index is stored in state analysis result
                extra fields `"conditional_preparation_index"`.
            extra (Dict[str, Any]): Extra metadata dictionary attached to analysis results.
        """
        options = super()._default_options()

        options.measurement_basis = None
        options.preparation_basis = None
        options.fitter = "linear_inversion"
        options.fitter_options = {}
        options.rescale_positive = True
        options.rescale_trace = True
        options.measurement_qubits = None
        options.preparation_qubits = None
        options.target = None
        options.target_bootstrap_samples = 0
        options.target_bootstrap_seed = None
        options.conditional_circuit_clbits = None
        options.conditional_measurement_indices = None
        options.conditional_preparation_indices = None
        options.extra = {}
        return options

    @classmethod
    def _get_fitter(cls, fitter: Union[str, Callable]) -> Callable:
        """Return fitter function for named builtin fitters"""
        if fitter is None:
            raise AnalysisError("No tomography fitter given")
        if not isinstance(fitter, str):
            return fitter
        if fitter in cls._builtin_fitters:
            return cls._builtin_fitters[fitter]
        raise AnalysisError(f"Unrecognized tomography fitter {fitter}")

    def _run_analysis(self, experiment_data):

        # Get option values
        meas_basis = self.options.measurement_basis
        meas_qubits = self.options.measurement_qubits
        if meas_basis and meas_qubits is None:
            meas_qubits = experiment_data.metadata.get("m_qubits")
        prep_basis = self.options.preparation_basis
        prep_qubits = self.options.preparation_qubits
        if prep_basis and prep_qubits is None:
            prep_qubits = experiment_data.metadata.get("p_qubits")
        cond_meas_indices = self.options.conditional_measurement_indices
        if cond_meas_indices is True:
            cond_meas_indices = list(range(len(meas_qubits)))
        cond_prep_indices = self.options.conditional_preparation_indices
        if cond_prep_indices is True:
            cond_prep_indices = list(range(len(prep_qubits)))

        # Generate tomography fitter data
        outcome_shape = None
        if meas_basis and meas_qubits:
            outcome_shape = meas_basis.outcome_shape(meas_qubits)

        outcome_data, shot_data, meas_data, prep_data = tomography_fitter_data(
            experiment_data.data(),
            outcome_shape=outcome_shape,
        )
        qpt = prep_data.size > 0

        # Get fitter kwargs
        fitter_kwargs = {}
        if meas_basis:
            fitter_kwargs["measurement_basis"] = meas_basis
        if meas_qubits:
            fitter_kwargs["measurement_qubits"] = meas_qubits
        if cond_meas_indices:
            fitter_kwargs["conditional_measurement_indices"] = cond_meas_indices
        if prep_basis:
            fitter_kwargs["preparation_basis"] = prep_basis
        if prep_qubits:
            fitter_kwargs["preparation_qubits"] = prep_qubits
        if cond_prep_indices:
            fitter_kwargs["conditional_preparation_indices"] = cond_prep_indices
        fitter_kwargs.update(**self.options.fitter_options)
        fitter = self._get_fitter(self.options.fitter)

        # Fit state results
        state_results = self._fit_state_results(
            fitter,
            outcome_data,
            shot_data,
            meas_data,
            prep_data,
            qpt=qpt,
            **fitter_kwargs,
        )

        other_results = []

        # Compute fidelity with target
        if len(state_results) == 1:
            other_results += self._fidelity_result(
                state_results[0],
                fitter,
                outcome_data,
                shot_data,
                meas_data,
                prep_data,
                qpt=qpt,
                **fitter_kwargs,
            )

        # Check positive
        other_results += self._positivity_result(state_results, qpt=qpt)

        # Check trace preserving
        if qpt:
            output_dim = np.prod(state_results[0].value.output_dims())
            other_results += self._tp_result(state_results, output_dim)

        # Finally format state result metadata to remove eigenvectors
        # which are no longer needed to reduce size
        for state_result in state_results:
            state_result.extra.pop("eigvecs")

        analysis_results = state_results + other_results

        if self.options.extra:
            for res in analysis_results:
                res.extra.update(self.options.extra)
        return analysis_results, []

    def _fit_state_results(
        self,
        fitter: Callable,
        outcome_data: np.ndarray,
        shot_data: np.ndarray,
        measurement_data: np.ndarray,
        preparation_data: np.ndarray,
        qpt: Union[bool, str, None] = "auto",
        **fitter_kwargs,
    ):
        """Fit state results from tomography data,"""
        try:
            fits, fitter_metadata = fitter(
                outcome_data,
                shot_data,
                measurement_data,
                preparation_data,
                **fitter_kwargs,
            )
        except AnalysisError as ex:
            raise AnalysisError(f"Tomography fitter failed with error: {str(ex)}") from ex

        # Post process fit
        states, states_metadata = postprocess_fitter(
            fits,
            fitter_metadata,
            make_positive=self.options.rescale_positive,
            trace="auto" if self.options.rescale_trace else None,
            qpt=qpt,
        )

        # Convert to results
        state_results = [
            AnalysisResultData("state", state, extra=extra)
            for state, extra in zip(states, states_metadata)
        ]
        return state_results

    def _fidelity_result(
        self,
        state_result: AnalysisResultData,
        fitter: Callable,
        outcome_data: np.ndarray,
        shot_data: np.ndarray,
        measurement_data: np.ndarray,
        preparation_data: np.ndarray,
        qpt: bool = False,
        **fitter_kwargs,
    ) -> List[AnalysisResultData]:
        """Calculate fidelity result if a target has been set"""
        target = self.options.target
        if target is None:
            return []

        # Compute fidelity
        name = "process_fidelity" if qpt else "state_fidelity"
        fidelity = self._compute_fidelity(state_result, target, qpt=qpt)

        if not self.options.target_bootstrap_samples:
            # No bootstrapping
            return [AnalysisResultData(name, fidelity)]

        # Optionally, Estimate std error of fidelity via boostrapping
        seed = self.options.target_bootstrap_seed
        if isinstance(seed, np.random.Generator):
            rng = seed
        else:
            rng = np.random.default_rng(seed)
        prob_data = outcome_data / shot_data[None, :, None]
        bs_fidelities = []
        for _ in range(self.options.target_bootstrap_samples):
            # TODO: remove conditional once numpy is pinned at 1.22 and above
            if version_is_at_least("numpy", "1.22"):
                sampled_data = rng.multinomial(shot_data, prob_data)
            else:
                sampled_data = np.zeros_like(outcome_data)
                for i in range(prob_data.shape[0]):
                    for j in range(prob_data.shape[1]):
                        sampled_data[i, j] = rng.multinomial(shot_data[j], prob_data[i, j])

            try:
                state_results = self._fit_state_results(
                    fitter,
                    sampled_data,
                    shot_data,
                    measurement_data,
                    preparation_data,
                    qpt=qpt,
                    **fitter_kwargs,
                )
                bs_fidelities.append(self._compute_fidelity(state_results[0], target, qpt=qpt))
            except AnalysisError:
                pass

        bs_stderr = np.std(bs_fidelities)
        return [
            AnalysisResultData(
                name,
                ufloat(fidelity, bs_stderr),
                extra={"bootstrap_samples": bs_fidelities},
            )
        ]

    @staticmethod
    def _positivity_result(
        state_results: List[AnalysisResultData], qpt: bool = False
    ) -> List[AnalysisResultData]:
        """Check if eigenvalues are positive"""
        total_cond = defaultdict(float)
        comps_cond = defaultdict(list)
        comps_pos = defaultdict(list)
        name = "completely_positive" if qpt else "positive"
        for result in state_results:
            cond_idx = result.extra.get("conditional_measurement_index", None)
            evals = result.extra["eigvals"]

            # Check if component is positive and add to extra if so
            cond = np.sum(np.abs(evals[evals < 0]))
            pos = bool(np.isclose(cond, 0))
            result.extra[name] = pos

            # Add component to combined result
            comps_cond[cond_idx].append(cond)
            comps_pos[cond_idx].append(pos)
            total_cond[cond_idx] += cond * result.extra["conditional_probability"]

        # Check if combined conditional state is positive
        results = []
        for key, delta in total_cond.items():
            is_pos = bool(np.isclose(delta, 0))
            result = AnalysisResultData(name, is_pos)
            if not is_pos:
                result.extra = {
                    "delta": delta,
                    "components": comps_pos[key],
                    "components_delta": comps_cond[key],
                }
            if key:
                result.extra["conditional_measurement_index"] = key
            results.append(result)
        return results

    @staticmethod
    def _tp_result(
        state_results: List[AnalysisResultData],
        output_dim: int = 1,
    ) -> List[AnalysisResultData]:
        """Check if QPT channel is trace preserving"""
        # Construct the Kraus TP condition matrix sum_i K_i^dag K_i
        # summed over all components k
        kraus_cond = {}
        for result in state_results:
            evals = result.extra["eigvals"]
            evecs = result.extra["eigvecs"]
            prob = result.extra["conditional_probability"]
            cond_meas_idx = result.extra.get("conditional_measurement_index", None)
            cond_prep_idx = result.extra.get("conditional_preparation_index", None)
            cond_idx = (cond_prep_idx, cond_meas_idx)
            size = len(evals)
            input_dim = size // output_dim
            mats = np.reshape(evecs.T, (size, input_dim, output_dim), order="F")
            comp_cond = np.einsum("i,iaj,ibj->ab", evals, mats.conj(), mats)
            if cond_idx in kraus_cond:
                kraus_cond[cond_idx] += prob * comp_cond
            else:
                kraus_cond[cond_idx] = prob * comp_cond

        results = []
        for key, val in kraus_cond.items():
            tp_cond = np.sum(np.abs(la.eigvalsh(val - np.eye(input_dim))))
            is_tp = bool(np.isclose(tp_cond, 0, atol=1e-5))
            result = AnalysisResultData("trace_preserving", is_tp, extra={})
            if not is_tp:
                result.extra["delta"] = tp_cond
            if key:
                result.extra["conditional_measurement_index"] = key
            results.append(result)
        return results

    @staticmethod
    def _compute_fidelity(
        state_result: AnalysisResultData,
        target: Union[Choi, DensityMatrix],
        qpt: bool = False,
    ) -> AnalysisResultData:
        """Faster computation of fidelity from eigen decomposition"""
        if qpt:
            input_dim = np.prod(state_result.value.input_dims())
        else:
            input_dim = 1
        evals = state_result.extra["eigvals"]
        evecs = state_result.extra["eigvecs"]

        # Format target to statevector or densitymatrix array
        if target is None:
            raise AnalysisError("No target state provided")
        if isinstance(target, QuantumChannel):
            target_state = Choi(target).data / input_dim
        elif isinstance(target, BaseOperator):
            target_state = np.ravel(Operator(target), order="F") / np.sqrt(input_dim)
        else:
            # Statevector or density matrix
            target_state = np.array(target)

        if target_state.ndim == 1:
            rho = evecs @ (evals / input_dim * evecs).T.conj()
            fidelity = np.real(target_state.conj() @ rho @ target_state)
        else:
            sqrt_rho = evecs @ (np.sqrt(evals / input_dim) * evecs).T.conj()
            eig = la.eigvalsh(sqrt_rho @ target_state @ sqrt_rho)
            fidelity = np.sum(np.sqrt(np.maximum(eig, 0))) ** 2
        return fidelity
