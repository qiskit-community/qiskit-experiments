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
import warnings
import numpy as np
import scipy.linalg as la

from qiskit.quantum_info import DensityMatrix, Choi, Operator
from qiskit.quantum_info.operators.base_operator import BaseOperator
from qiskit.quantum_info.operators.channel.quantum_channel import QuantumChannel

from qiskit_experiments.exceptions import AnalysisError
from qiskit_experiments.framework import BaseAnalysis, AnalysisResultData, Options
from .fitters import (
    tomography_fitter_data,
    postprocess_fitter,
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
            conditional_measurement_indices (list[int]): Optional, indices of measurement
                qubits to treat as conditional for conditional fragment reconstruction of
                the circuit.
            conditional_circuit_clbits (list[int]): Optional, clbits in the source circuit
                to treat as conditional for conditional fragment reconstruction of the
                circuit.
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
        options.conditional_measurement_indices = None
        options.conditional_circuit_clbits = None
        return options

    def set_options(self, **fields):
        if fields.get("fitter", None) in [
            "scipy_linear_lstsq",
            "scipy_gaussian_lstsq",
            scipy_linear_lstsq,
            scipy_gaussian_lstsq,
        ]:
            warnings.warn(
                "The scipy lstsq tomography fitters are deprecated as of 0.4 and will "
                "be removed after the 0.5 release. Use the `linear_lstsq`, "
                "`cvxpy_linear_lstsq`, or `cvxpy_gaussian_lstsq` fitter instead.",
                DeprecationWarning,
                stacklevel=2,
            )
        super().set_options(**fields)

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
        # Get option values.
        measurement_basis = self.options.measurement_basis
        measurement_qubits = self.options.measurement_qubits
        if measurement_basis and measurement_qubits is None:
            measurement_qubits = experiment_data.metadata.get("m_qubits")
        preparation_basis = self.options.preparation_basis
        preparation_qubits = self.options.preparation_qubits
        if preparation_basis and preparation_qubits is None:
            preparation_qubits = experiment_data.metadata.get("p_qubits")
        conditional_measurement_indices = self.options.conditional_measurement_indices
        if conditional_measurement_indices is None:
            conditional_measurement_indices = experiment_data.metadata.get("c_indices")

        # Generate tomography fitter data
        outcome_shape = None
        if measurement_basis and measurement_qubits:
            outcome_shape = measurement_basis.outcome_shape(measurement_qubits)

        outcome_data, shot_data, meas_data, prep_data = tomography_fitter_data(
            experiment_data.data(),
            outcome_shape=outcome_shape,
        )
        fitter = self._get_fitter(self.options.fitter)

        try:
            fits, fitter_metadata = fitter(
                outcome_data,
                shot_data,
                meas_data,
                prep_data,
                measurement_basis=measurement_basis,
                measurement_qubits=measurement_qubits,
                preparation_basis=preparation_basis,
                preparation_qubits=preparation_qubits,
                conditional_measurement_indices=conditional_measurement_indices,
                **self.options.fitter_options,
            )
        except AnalysisError as ex:
            raise AnalysisError(f"Tomography fitter failed with error: {str(ex)}") from ex

        # Post process fit
        states, states_metadata = postprocess_fitter(
            fits,
            fitter_metadata,
            make_positive=self.options.rescale_positive,
            trace="auto" if self.options.rescale_trace else None,
        )

        if prep_data.shape[1]:
            qpt = True
            input_dim = np.prod(states[0].input_dims())
        else:
            qpt = False
            input_dim = 1

        # Convert to results
        state_results = [
            AnalysisResultData("state", state, extra=extra)
            for state, extra in zip(states, states_metadata)
        ]
        other_results = []

        # Compute fidelity with target
        target_state = self.options.target
        if len(state_results) == 1 and target_state is not None:
            # Note: this currently only works for non-conditional tomography
            other_results.append(
                self._fidelity_result(state_results[0], target_state, input_dim=input_dim)
            )

        # Check positive
        other_results.append(self._positivity_result(state_results, qpt=qpt))

        # Check trace preserving
        if qpt:
            other_results.append(self._tp_result(state_results, input_dim=input_dim))

        # Finally format state result metadata to remove eigenvectors
        # which are no longer needed to reduce size
        for state_result in state_results:
            state_result.extra.pop("eigvecs")

        analysis_results = state_results + other_results

        return analysis_results, []

    @staticmethod
    def _positivity_result(
        state_results: List[AnalysisResultData], qpt: bool = False
    ) -> AnalysisResultData:
        """Check if eigenvalues are positive"""
        total_cond = 0.0
        comps_cond = []
        comps_pos = []
        name = "completely_positive" if qpt else "positive"
        for result in state_results:
            evals = result.extra["eigvals"]

            # Check if component is positive and add to extra if so
            cond = np.sum(np.abs(evals[evals < 0]))
            pos = bool(np.isclose(cond, 0))
            result.extra[name] = pos

            # Add component to combined result
            comps_cond.append(cond)
            comps_pos.append(pos)
            total_cond += cond * result.extra["component_probability"]

        # Check if combined conditional state is positive
        is_pos = bool(np.isclose(total_cond, 0))
        result = AnalysisResultData(name, is_pos)
        if not is_pos:
            result.extra = {
                "delta": total_cond,
                "components": comps_pos,
                "components_delta": comps_cond,
            }
        return result

    @staticmethod
    def _tp_result(
        state_results: List[AnalysisResultData],
        input_dim: int = 1,
    ) -> AnalysisResultData:
        """Check if QPT channel is trace preserving"""
        # Construct the Kraus TP condition matrix sum_i K_i^dag K_i
        # summed over all components k
        kraus_cond = 0.0
        for result in state_results:
            evals = result.extra["eigvals"]
            evecs = result.extra["eigvecs"]
            prob = result.extra["component_probability"]
            size = len(evals)
            output_dim = size // input_dim
            mats = np.reshape(evecs.T, (size, output_dim, input_dim), order="F")
            comp_cond = np.einsum("i,ija,ijb->ab", evals, mats.conj(), mats)
            kraus_cond = kraus_cond + prob * comp_cond

        tp_cond = np.sum(np.abs(la.eigvalsh(kraus_cond - np.eye(input_dim))))
        is_tp = bool(np.isclose(tp_cond, 0))
        result = AnalysisResultData("trace_preserving", is_tp)
        if not is_tp:
            result.extra = {"delta": tp_cond}
        return result

    @staticmethod
    def _fidelity_result(
        state_result: AnalysisResultData,
        target: Union[Choi, DensityMatrix],
        input_dim: int = 1,
    ) -> AnalysisResultData:
        """Faster computation of fidelity from eigen decomposition"""
        evals = state_result.extra["eigvals"]
        evecs = state_result.extra["eigvecs"]

        # Format target to statevector or densitymatrix array
        name = "process_fidelity" if input_dim > 1 else "state_fidelity"
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
        return AnalysisResultData(name, fidelity)
