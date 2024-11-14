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
Constrained convex least-squares tomography fitter.
"""

from typing import Optional, Dict, Tuple, Union
import time
import numpy as np

from qiskit_experiments.library.tomography.basis import (
    MeasurementBasis,
    PreparationBasis,
)
from . import cvxpy_utils
from .cvxpy_utils import cvxpy
from . import lstsq_utils
from .fitter_data import _basis_dimensions


@cvxpy_utils.requires_cvxpy
def cvxpy_linear_lstsq(
    outcome_data: np.ndarray,
    shot_data: np.ndarray,
    measurement_data: np.ndarray,
    preparation_data: np.ndarray,
    measurement_basis: Optional[MeasurementBasis] = None,
    preparation_basis: Optional[PreparationBasis] = None,
    measurement_qubits: Optional[Tuple[int, ...]] = None,
    preparation_qubits: Optional[Tuple[int, ...]] = None,
    conditional_measurement_indices: Optional[Tuple[int, ...]] = None,
    conditional_preparation_indices: Optional[Tuple[int, ...]] = None,
    trace: Union[None, float, str] = "auto",
    psd: bool = True,
    trace_preserving: Union[None, bool, str] = "auto",
    partial_trace: Optional[np.ndarray] = None,
    weights: Optional[np.ndarray] = None,
    **kwargs,
) -> Tuple[np.ndarray, Dict]:
    r"""Constrained weighted linear least-squares tomography fitter.

    Overview
        This fitter reconstructs the maximum-likelihood estimate by using
        ``cvxpy`` to minimize the constrained least-squares negative log
        likelihood function

        .. math::
            \hat{\rho}
                &= -\mbox{argmin }\log\mathcal{L}{\rho} \\
                &= \mbox{argmin }\sum_i w_i^2(\mbox{Tr}[E_j\rho] - \hat{p}_i)^2 \\
                &= \mbox{argmin }\|W(Ax - y) \|_2^2

        subject to

        - *Positive-semidefinite* (``psd=True``): :math:`\rho \gg 0` is constrained
          to be a positive-semidefinite matrix.
        - *Trace* (``trace=t``): :math:`\mbox{Tr}(\rho) = t` is constrained to have
          the specified trace.
        - *Trace preserving* (``trace_preserving=True``): When performing process
          tomography the Choi-state :math:`\rho` represents is constrained to be
          trace preserving.

        where

        - :math:`A` is the matrix of measurement operators
          :math:`A = \sum_i |i\rangle\!\langle\!\langle M_i|`
        - :math:`y` is the vector of expectation value data for each projector
          corresponding to estimates of :math:`b_i = Tr[M_i \cdot x]`.
        - :math:`x` is the vectorized density matrix (or Choi-matrix) to be fitted
          :math:`x = |\rho\rangle\\!\rangle`.

    .. note:

        Various solvers can be called in CVXPY using the `solver` keyword
        argument. When ``psd=True`` the optimization problem is a case of a
        *semidefinite program* (SDP) and requires a SDP compatible solver
        for CVXPY. CVXPY includes an SDP compatible solver `SCS`` but it
        is recommended to install the the open-source ``CVXOPT`` solver
        or one of the supported commercial solvers. See the `CVXPY
        documentation
        <https://www.cvxpy.org/tutorial/advanced/index.html#solve-method-options>`_
        for more information on solvers.

    .. note::

        Linear least-squares constructs the full basis matrix :math:`A` as a dense
        numpy array so should not be used for than 5 or 6 qubits. For larger number
        of qubits try the
        :func:`~qiskit_experiments.library.tomography.fitters.linear_inversion`
        fitter function.

    Args:
        outcome_data: measurement outcome frequency data.
        shot_data: basis measurement total shot data.
        measurement_data: measurement basis index data.
        preparation_data: preparation basis index data.
        measurement_basis: Optional, measurement matrix basis.
        preparation_basis: Optional, preparation matrix basis.
        measurement_qubits: Optional, the physical qubits that were measured.
            If None they are assumed to be ``[0, ..., M-1]`` for M measured qubits.
        preparation_qubits: Optional, the physical qubits that were prepared.
            If None they are assumed to be ``[0, ..., N-1]`` for N prepared qubits.
        conditional_measurement_indices: Optional, conditional measurement data
            indices. If set this will return a list of fitted states conditioned
            on a fixed basis measurement of these qubits.
        conditional_preparation_indices: Optional, conditional preparation data
            indices. If set this will return a list of fitted states conditioned
            on a fixed basis preparation of these qubits.
        trace: trace constraint for the fitted matrix. If "auto" this will be set
               to 1 for QST or the input dimension for QST (default: "auto").
        psd: If True rescale the eigenvalues of fitted matrix to be positive
             semidefinite (default: True)
        trace_preserving: Enforce the fitted matrix to be trace preserving when
                          fitting a Choi-matrix in quantum process
                          tomography. If "auto" this will be set to True for
                          QPT and False for QST (default: "auto").
        partial_trace: Enforce conditional fitted Choi matrices to partial
                       trace to POVM matrices.
        weights: Optional array of weights for least squares objective. Weights
                 should be the same shape as the outcome_data.
        kwargs: kwargs for cvxpy solver.

    Raises:
        QiskitError: If CVXPy is not installed on the current system.
        AnalysisError: If analysis fails.

    Returns:
        The fitted matrix rho that maximizes the least-squares likelihood function.
    """
    t_start = time.time()

    if measurement_basis and measurement_qubits is None:
        measurement_qubits = tuple(range(measurement_data.shape[1]))
    if preparation_basis and preparation_qubits is None:
        preparation_qubits = tuple(range(preparation_data.shape[1]))

    input_dims = _basis_dimensions(
        basis=preparation_basis,
        qubits=preparation_qubits,
        conditional_indices=conditional_preparation_indices,
    )
    output_dims = _basis_dimensions(
        basis=measurement_basis,
        qubits=measurement_qubits,
        conditional_indices=conditional_measurement_indices,
    )

    if trace_preserving == "auto" and preparation_data.shape[1] > 0:
        trace_preserving = True

    if trace == "auto" and output_dims is not None:
        trace = np.prod(output_dims)

    metadata = {
        "fitter": "cvxpy_linear_lstsq",
        "input_dims": input_dims,
        "output_dims": output_dims,
    }

    # Conditional measurement indices
    if conditional_measurement_indices:
        cond_meas_data = measurement_data[:, np.array(conditional_measurement_indices, dtype=int)]
        cond_meas_indices = np.unique(cond_meas_data, axis=0)
        num_meas_cond = len(cond_meas_indices)
        metadata["conditional_measurement_index"] = []
        metadata["conditional_measurement_outcome"] = []
    else:
        num_meas_cond = 0
        cond_meas_data = np.zeros((measurement_data.shape[0], 0), dtype=int)
        cond_meas_indices = np.zeros((1, 0), dtype=int)

    if conditional_preparation_indices:
        cond_prep_data = preparation_data[:, np.array(conditional_preparation_indices, dtype=int)]
        cond_prep_indices = np.unique(cond_prep_data, axis=0)
        num_prep_cond = len(cond_meas_indices)
        metadata["conditional_preparation_index"] = []
    else:
        num_prep_cond = 0
        cond_prep_data = np.zeros((preparation_data.shape[0], 0), dtype=int)
        cond_prep_indices = np.zeros((1, 0), dtype=int)

    if outcome_data.shape[0] > 1:
        metadata["conditional_circuit_outcome"] = []

    fits = []
    for cond_prep_idx in cond_prep_indices:
        for cond_meas_idx in cond_meas_indices:
            # Mask for specified conditional indices
            cond_mask = np.all(cond_meas_data == cond_meas_idx, axis=1) & np.all(
                cond_prep_data == cond_prep_idx, axis=1
            )
            if weights is None:
                cond_weights = None
            else:
                cond_weights = weights[:, cond_mask]

            basis_matrix, probability_data, probability_weights = lstsq_utils.lstsq_data(
                outcome_data[:, cond_mask],
                shot_data[cond_mask],
                measurement_data[cond_mask],
                preparation_data[cond_mask],
                measurement_basis=measurement_basis,
                preparation_basis=preparation_basis,
                measurement_qubits=measurement_qubits,
                preparation_qubits=preparation_qubits,
                weights=cond_weights,
                conditional_measurement_indices=conditional_measurement_indices,
                conditional_preparation_indices=conditional_preparation_indices,
            )

            # Since CVXPY only works with real variables we must specify the real
            # and imaginary parts of matrices separately: rho = rho_r + 1j * rho_i

            num_circ_components, num_tomo_components, _ = probability_data.shape
            dim = int(np.sqrt(basis_matrix.shape[1]))

            # Generate list of conditional components for block diagonal matrix
            # rho = sum_k |k><k| \otimes rho(k)
            rhos_r = []
            rhos_i = []
            cons = []
            for i in range(num_circ_components):
                for j in range(num_tomo_components):
                    rho_r, rho_i, cons_i = cvxpy_utils.complex_matrix_variable(
                        dim, hermitian=True, psd=psd
                    )
                    rhos_r.append(rho_r)
                    rhos_i.append(rho_i)
                    cons.append(cons_i)
                    if num_circ_components > 1:
                        metadata["conditional_circuit_outcome"].append(i)
                    if num_meas_cond:
                        metadata["conditional_measurement_index"].append(tuple(cond_meas_idx))
                        metadata["conditional_measurement_outcome"].append(j)
                    if num_prep_cond:
                        metadata["conditional_preparation_index"].append(tuple(cond_prep_idx))

            # Partial trace when fitting Choi-matrices for quantum process tomography.
            # This applied to the sum of conditional components
            # Note that this adds an implicitly
            # trace preserving is a specific partial trace constraint ptrace(rho) = I
            # Note: partial trace constraints implicitly define a trace constraint,
            # so if a different trace constraint is specified it will be ignored
            joint_cons = None
            if partial_trace is not None:
                for rho_r, rho_i, povm in zip(rhos_r, rhos_i, partial_trace):
                    joint_cons = cvxpy_utils.partial_trace_constaint(rho_r, rho_i, povm)
            elif trace_preserving:
                input_dim = np.prod(input_dims)
                joint_cons = cvxpy_utils.trace_preserving_constaint(
                    rhos_r,
                    rhos_i,
                    input_dim=input_dim,
                    hermitian=True,
                )
            elif trace is not None:
                joint_cons = cvxpy_utils.trace_constraint(
                    rhos_r, rhos_i, trace=trace, hermitian=True
                )

            # OBJECTIVE FUNCTION

            # The function we wish to minimize is || arg ||_2 where
            #   arg =  bm * vec(rho) - data
            # Since we are working with real matrices in CVXPY we expand this as
            #   bm * vec(rho) = (bm_r + 1j * bm_i) * vec(rho_r + 1j * rho_i)
            #                 = bm_r * vec(rho_r) - bm_i * vec(rho_i)
            #                   + 1j * (bm_r * vec(rho_i) + bm_i * vec(rho_r))
            #                 = bm_r * vec(rho_r) - bm_i * vec(rho_i)
            # where we drop the imaginary part since the expectation value is real

            # Construct block diagonal fit variable from conditional components
            # Construct objective function
            if probability_weights is not None:
                probability_data = probability_weights * probability_data
                bms_r = []
                bms_i = []
                for i in range(num_circ_components):
                    for j in range(num_tomo_components):
                        weighted_mat = probability_weights[i, j][:, None] * basis_matrix
                        bms_r.append(np.real(weighted_mat))
                        bms_i.append(np.imag(weighted_mat))
            else:
                bm_r = np.real(basis_matrix)
                bm_i = np.imag(basis_matrix)
                bms_r = [bm_r] * num_circ_components * num_tomo_components
                bms_i = [bm_i] * num_circ_components * num_tomo_components

            # Stack lstsq objective from sum of components
            args = []
            idx = 0
            for i in range(num_circ_components):
                for j in range(num_tomo_components):
                    model = bms_r[idx] @ cvxpy.vec(rhos_r[idx], order="F") - bms_i[idx] @ cvxpy.vec(
                        rhos_i[idx], order="F"
                    )
                    data = probability_data[i, j]
                    args.append(model - data)
                    idx += 1

            # Combine all variables and constraints into a joint optimization problem
            # if there is a joint constraint
            if joint_cons:
                args = [cvxpy.hstack(args)]
                for cons_i in cons:
                    joint_cons += cons_i
                cons = [joint_cons]

            # Solve each component separately
            metadata["cvxpy_solver"] = None
            metadata["cvxpy_status"] = []
            for arg, con in zip(args, cons):
                # Optimization problem
                obj = cvxpy.Minimize(cvxpy.norm(arg, p=2))
                prob = cvxpy.Problem(obj, con)

                # Solve SDP
                cvxpy_utils.solve_iteratively(prob, 5000, **kwargs)

                # Return optimal values and problem metadata
                metadata["cvxpy_solver"] = prob.solver_stats.solver_name
                metadata["cvxpy_status"].append(prob.status)

            fits += [rho_r.value + 1j * rho_i.value for rho_r, rho_i in zip(rhos_r, rhos_i)]

    # Add additional metadata
    if psd:
        metadata["psd_constraint"] = True
    if partial_trace is not None:
        metadata["partial_trace"] = partial_trace
    elif trace_preserving:
        metadata["trace_preserving"] = True
    elif trace is not None:
        metadata["trace"] = trace
    t_stop = time.time()
    metadata["fitter_time"] = t_stop - t_start

    if len(fits) == 1:
        return fits[0], metadata
    return fits, metadata


@cvxpy_utils.requires_cvxpy
def cvxpy_gaussian_lstsq(
    outcome_data: np.ndarray,
    shot_data: np.ndarray,
    measurement_data: np.ndarray,
    preparation_data: np.ndarray,
    measurement_basis: Optional[MeasurementBasis] = None,
    preparation_basis: Optional[PreparationBasis] = None,
    measurement_qubits: Optional[Tuple[int, ...]] = None,
    preparation_qubits: Optional[Tuple[int, ...]] = None,
    conditional_measurement_indices: Optional[Tuple[int, ...]] = None,
    trace: Union[None, float, str] = "auto",
    psd: bool = True,
    trace_preserving: Union[None, bool, str] = "auto",
    partial_trace: Optional[np.ndarray] = None,
    outcome_prior: Union[np.ndarray, int] = 0.5,
    max_weight: float = 1e10,
    **kwargs,
) -> Dict:
    r"""Constrained Gaussian linear least-squares tomography fitter.

    .. note::

        This function calls :func:`cvxpy_linear_lstsq` with a Gaussian weights
        vector. Refer to its documentation for additional details.

    Overview
        This fitter reconstructs the maximum-likelihood estimate by using
        ``cvxpy`` to minimize the constrained least-squares negative log
        likelihood function

        .. math::
            \hat{\rho}
                &= \mbox{argmin} (-\log\mathcal{L}{\rho}) \\
                &= \mbox{argmin }\|W(Ax - y) \|_2^2 \\
            -\log\mathcal{L}(\rho)
                &= |W(Ax -y) \|_2^2 \\
                &= \sum_i \frac{1}{\sigma_i^2}(\mbox{Tr}[E_j\rho] - \hat{p}_i)^2

    Additional Details
        The Gaussian weights are estimated from the observed frequency and shot data
        via a Bayesian update of a Dirichlet distribution with observed outcome data
        frequencies :math:`f_i(s)`, and Dirichlet prior :math:`\alpha_i(s)` for
        tomography basis index `i` and measurement outcome `s`.

        The mean posterior probabilities are computed as

        .. math:
            p_i(s) &= \frac{f_i(s) + \alpha_i(s)}{\bar{\alpha}_i + N_i} \\
            Var[p_i(s)] &= \frac{p_i(s)(1-p_i(s))}{\bar{\alpha}_i + N_i + 1}
            w_i(s) = \sqrt{Var[p_i(s)]}^{-1}

        where :math:`N_i = \sum_s f_i(s)` is the total number of shots, and
        :math:`\bar{\alpha}_i = \sum_s \alpha_i(s)` is the norm of the prior.

    Args:
        outcome_data: measurement outcome frequency data.
        shot_data: basis measurement total shot data.
        measurement_data: measurement basis index data.
        preparation_data: preparation basis index data.
        measurement_basis: Optional, measurement matrix basis.
        preparation_basis: Optional, preparation matrix basis.
        measurement_qubits: Optional, the physical qubits that were measured.
            If None they are assumed to be ``[0, ..., M-1]`` for M measured qubits.
        preparation_qubits: Optional, the physical qubits that were prepared.
            If None they are assumed to be ``[0, ..., N-1]`` for N prepared qubits.
        conditional_measurement_indices: Optional, conditional measurement data
            indices. If set this will return a list of conditional fitted states
            conditioned on a fixed basis measurement of these qubits.
        trace: trace constraint for the fitted matrix. If "auto" this will be set
               to 1 for QST or the input dimension for QST (default: "auto").
        psd: If True rescale the eigenvalues of fitted matrix to be positive
             semidefinite (default: True)
        trace_preserving: Enforce the fitted matrix to be trace preserving when
                          fitting a Choi-matrix in quantum process
                          tomography. If "auto" this will be set to True for
                          QPT and False for QST (default: "auto").
        partial_trace: Enforce conditional fitted Choi matrices to partial
                       trace to POVM matrices.
        outcome_prior: The Bayesian prior :math:`\alpha` to use computing Gaussian
            weights. See additional information.
        max_weight: Set the maximum value allowed for weights vector computed from
            tomography data variance.
        kwargs: kwargs for cvxpy solver.

    Raises:
        QiskitError: If CVXPY is not installed on the current system.
        AnalysisError: If analysis fails.

    Returns:
        The fitted matrix rho that maximizes the least-squares likelihood function.
    """
    t_start = time.time()

    weights = lstsq_utils.binomial_weights(
        outcome_data,
        shot_data=shot_data,
        outcome_prior=outcome_prior,
        max_weight=max_weight,
    )

    fits, metadata = cvxpy_linear_lstsq(
        outcome_data,
        shot_data,
        measurement_data,
        preparation_data,
        measurement_basis=measurement_basis,
        preparation_basis=preparation_basis,
        measurement_qubits=measurement_qubits,
        preparation_qubits=preparation_qubits,
        conditional_measurement_indices=conditional_measurement_indices,
        trace=trace,
        psd=psd,
        trace_preserving=trace_preserving,
        partial_trace=partial_trace,
        weights=weights,
        **kwargs,
    )

    t_stop = time.time()

    # Update metadata
    metadata["fitter"] = "cvxpy_gaussian_lstsq"
    metadata["fitter_time"] = t_stop - t_start

    return fits, metadata
