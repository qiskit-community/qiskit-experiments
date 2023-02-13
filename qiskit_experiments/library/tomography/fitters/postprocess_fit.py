# This code is part of Qiskit.
#
# (C) Copyright IBM 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
"""
Post-process tomography fits
"""

from typing import List, Dict, Tuple, Union, Optional
import numpy as np
import scipy.linalg as la
from qiskit.quantum_info import DensityMatrix, Choi

from qiskit_experiments.exceptions import AnalysisError


def postprocess_fitter(
    fits: Union[np.ndarray, List[np.ndarray]],
    fitter_metadata: Optional[Dict] = None,
    make_positive: bool = False,
    trace: Union[float, str, None] = "auto",
) -> Tuple[List[np.ndarray], List[Dict[str, any]]]:
    """Post-process raw fitter result.

    Args:
        fits: Fitter result, or result components.
        fitter_metadata: Dict of metadata returned from fitter.
        make_positive: If True rescale the fitted state to be PSD if any
            eigenvalues are negative.
        trace: If "auto" or float rescale the fitted state to have the
            specified trace. For "auto" states will be set to trace 1
            and channels to trace = dimension.

    Returns:
        The fitted state components, and metadata.
    """
    if not isinstance(fits, (list, tuple)):
        fits = [fits]

    # Get dimension and trace from fitter metadata
    conditionals = fitter_metadata.pop("component_conditionals", None)
    input_dims = fitter_metadata.pop("input_dims", None)
    output_dims = fitter_metadata.pop("output_dims", None)

    # Convert fitter matrix to state data for post-processing
    input_dim = np.prod(input_dims) if input_dims else 1
    qpt = input_dim > 1
    if trace == "auto":
        trace = input_dim

    states = []
    states_metadata = []
    fit_traces = []
    total_trace = 0.0
    for i, fit in enumerate(fits):
        # Get eigensystem of state fit
        raw_eigvals, eigvecs = _state_eigensystem(fit)

        # Optionally rescale eigenvalues to be non-negative
        if make_positive and np.any(raw_eigvals < 0):
            eigvals = _make_positive(raw_eigvals)
            fit = eigvecs @ (eigvals * eigvecs).T.conj()
            rescaled_psd = True
        else:
            eigvals = raw_eigvals
            rescaled_psd = False

        # Optionally rescale fit trace
        fit_trace = np.sum(eigvals).real
        fit_traces.append(fit_trace)
        if (
            trace is not None
            and not np.isclose(fit_trace, 0, atol=1e-10)
            and not np.isclose(abs(fit_trace - trace), 0, atol=1e-10)
        ):
            scale = trace / fit_trace
            fit = fit * scale
            eigvals = eigvals * scale
        else:
            trace = fit_trace

        # Convert class of value
        if qpt:
            state = Choi(fit, input_dims=input_dims, output_dims=output_dims)
        else:
            state = DensityMatrix(fit, dims=output_dims)

        metadata = {
            "trace": trace,
            "eigvals": eigvals,
            "eigvecs": eigvecs,
            "raw_eigvals": raw_eigvals,
            "rescaled_psd": rescaled_psd,
            "fitter_metadata": fitter_metadata or {},
        }

        states.append(state)
        states_metadata.append(metadata)

    # Compute the conditional probability of each component so that the
    # total probability of all components is 1, and optional rescale trace
    # of each component
    total_trace = sum(fit_traces)
    for i, (fit_trace, meta) in enumerate(zip(fit_traces, states_metadata)):
        # Compute conditional component probability from the the component
        # non-rescaled fit trace
        meta["component_probability"] = fit_trace / total_trace
        meta["component_index"] = i
        if conditionals:
            meta["component_conditional"] = conditionals[i]

    return states, states_metadata


def _state_eigensystem(state: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Compute the eigensystem of a fitted state.

    The eigenvalues are returned as a real array ordered from
    smallest to largest eigenvalues.

    Args:
        state: the fitted state matrix.

    Returns:
        A pair of (eigenvalues, eigenvectors).
    """
    evals, evecs = la.eigh(state)
    # Truncate eigenvalues to real part
    evals = np.real(evals)
    # Sort eigensystem from largest to smallest eigenvalues
    sort_inds = np.flip(np.argsort(evals))
    return evals[sort_inds], evecs[:, sort_inds]


def _make_positive(evals: np.ndarray, epsilon: float = 0) -> np.ndarray:
    """Rescale a real vector to be non-negative.

    This truncates any negative values to zero and rescales
    the remaining eigenvectors such that the sum of the vector
    is preserved.
    """
    if epsilon < 0:
        raise AnalysisError("epsilon must be non-negative.")
    scaled = evals.copy()
    dim = len(evals)
    idx = dim - 1
    accum = 0.0
    while idx >= 0:
        shift = accum / (idx + 1)
        if evals[idx] + shift < epsilon:
            scaled[idx] = 0
            accum = accum + evals[idx]
            idx -= 1
        else:
            for j in range(idx + 1):
                scaled[j] = evals[j] + shift
            break

    return scaled
