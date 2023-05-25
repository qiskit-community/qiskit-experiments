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
Quantum process tomography analysis
"""


from typing import List, Dict, Tuple, Union, Optional, Callable, Sequence
import functools
from collections import defaultdict
import numpy as np

from qiskit.result import marginal_distribution

from qiskit_experiments.exceptions import AnalysisError
from qiskit_experiments.library.tomography.basis import MeasurementBasis, PreparationBasis


def tomography_fitter_data(
    data: List[Dict[str, any]],
    outcome_shape: Optional[Union[np.ndarray, int]] = None,
    m_idx_key: str = "m_idx",
    p_idx_key: str = "p_idx",
    clbits_key: str = "clbits",
    clbits: Optional[List[int]] = None,
    cond_clbits_key: str = "cond_clbits",
    cond_clbits: Optional[List[int]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return a tuple of tomography data arrays.

    Args:
        data: tomography experiment data list.
        outcome_shape: Optional, the shape of measurement outcomes for
                       each measurement index. Default is 2.
        m_idx_key: metadata key for measurement basis index (Default: "m_idx").
        p_idx_key: metadata key for preparation basis index (Default: "p_idx").
        clbits_key: metadata clbit key for marginalizing counts. Used if the
            `clbits` kwarg is not provided (Default: "clbits").
        clbits: Optional, clbit indices for marginalizing counts. If provided
            this overrides `clbits_key` in metadata.
        cond_clbits_key: metadata conditional clbit key for conditioning when
            marginalizing counts. Used if the `cond_clbits` kwarg is not
            provided (Default: "cond_clbits").
        cond_clbits: Optional, clbit indices to condition on when when
            marginalizing counts. If provided this overrides `cond_clbits_key`
            in metadata.

    Returns:
        Tuple of data arrays for tomography fitters of form
        (outcome_data, shot_data, measurement data, preparation data).
    """
    meas_size = None
    prep_size = None

    # Construct marginalized tomography count dicts
    outcome_dict = defaultdict(lambda: defaultdict(lambda: 0))

    for datum in data:
        # Get basis data
        metadata = datum["metadata"]
        meas_element = tuple(metadata[m_idx_key]) if m_idx_key in metadata else tuple()
        prep_element = tuple(metadata[p_idx_key]) if p_idx_key in metadata else tuple()
        if meas_size is None:
            meas_size = len(meas_element)
        if prep_size is None:
            prep_size = len(prep_element)
        basis_key = (meas_element, prep_element)

        # Marginalize counts
        counts = datum["counts"]
        num_cond = 0
        count_clbits = []
        if cond_clbits is None and cond_clbits_key is not None:
            cond_clbits = metadata[cond_clbits_key] or []
        if cond_clbits:
            count_clbits += cond_clbits
            num_cond = len(cond_clbits)
        if clbits is None and clbits_key is not None:
            clbits = metadata[clbits_key] or []
        if clbits:
            count_clbits += clbits
        if count_clbits:
            # The input clbits might come in out of order, sort to ensure we
            # don't permute the output during marginalization
            count_clbits = list(sorted(count_clbits))
            counts = marginal_distribution(counts, count_clbits)

        # Accumulate counts
        combined_counts = outcome_dict[basis_key]
        for key, val in counts.items():
            combined_counts[key.replace(" ", "")] += val

    # Format number of outcomes
    outcome_shape = outcome_shape or 2
    if isinstance(outcome_shape, int):
        outcome_shape = meas_size * (outcome_shape,)
    else:
        outcome_shape = tuple(outcome_shape)
    outcome_size = np.prod(outcome_shape, dtype=int)

    # Construct function for converting count outcome dit-strings into
    # integers based on the specified number of outcomes of the measurement
    # bases on each qubit
    if outcome_size == 1:

        def outcome_func(_):
            return 1

    else:
        outcome_func = _int_outcome_function(outcome_shape)

    # Conditional dimension for conditional measurement in source circuit
    if num_cond:
        cond_shape = 2**num_cond
        cond_mask = sum(1 << i for i in range(num_cond))
    else:
        cond_shape = 1
        cond_mask = 0

    # Initialize and fill data arrays
    num_basis = len(outcome_dict)
    measurement_data = np.zeros((num_basis, meas_size), dtype=int)
    preparation_data = np.zeros((num_basis, prep_size), dtype=int)
    shot_data = np.zeros(num_basis, dtype=int)
    outcome_data = np.zeros((cond_shape, num_basis, outcome_size), dtype=int)
    for i, (basis_key, counts) in enumerate(outcome_dict.items()):
        measurement_data[i] = basis_key[0]
        preparation_data[i] = basis_key[1]
        for outcome, freq in counts.items():
            ioutcome = outcome_func(outcome)
            cond_idx = cond_mask & ioutcome
            meas_outcome = ioutcome >> num_cond
            outcome_data[cond_idx][i][meas_outcome] = freq
            shot_data[i] += freq

    return outcome_data, shot_data, measurement_data, preparation_data


@functools.lru_cache(None)
def _int_outcome_function(outcome_shape: Tuple[int, ...]) -> Callable:
    """Generate function for converting string outcomes to ints"""
    # Recursively extract leading bit(dit)
    if len(set(outcome_shape)) == 1:
        # All outcomes are the same shape, so we can use a constant base
        base = outcome_shape[0]
        return lambda outcome: int(outcome, base)

    # General function where each dit could be a different base
    @functools.lru_cache(2048)
    def _int_outcome_general(outcome: str):
        """Convert a general dit-string outcome to integer"""
        # Recursively extract leading bit(dit)
        value = 0
        for i, base in zip(outcome, outcome_shape):
            value *= base
            value += int(i, base)
        return value

    return _int_outcome_general


def _basis_dimensions(
    basis: Optional[Union[MeasurementBasis, PreparationBasis]] = None,
    qubits: Optional[Sequence[int]] = None,
    conditional_indices: Optional[Sequence[int]] = None,
) -> Tuple[int, ...]:
    """Caculate input and output dimensions for basis and qubits"""
    if not qubits:
        return (1,)

    # Get dimension of the preparation and measurement qubits subsystems
    if conditional_indices is None:
        full_qubits = qubits
    else:
        # Remove conditional qubits from full meas qubits
        full_qubits = [q for i, q in enumerate(qubits) if i not in conditional_indices]

    if full_qubits:
        if not basis:
            raise AnalysisError("No tomography basis provided.")
        return basis.matrix_shape(full_qubits)

    return (1,)
