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
Basic quantum state tomography linear inversion fitter.
"""

import itertools as it
import numpy as np

from qiskit.quantum_info import DensityMatrix
from qiskit.result import Counts

# Temporarily import helper functions from ignis tomography module
from qiskit.ignis.verification.tomography.basis import PauliBasis
from qiskit.ignis.verification.tomography.fitters.lstsq_fit import lstsq_fit
from qiskit.ignis.verification.tomography.data import marginal_counts, count_keys

from qiskit_experiments.base_analysis import BaseAnalysis, AnalysisResult


class QSTAnalysis(BaseAnalysis):
    """Base Experiment result analysis class."""

    def _run_analysis(self, experiment_data, **params):
        # Options
        standard_weights = params.pop("standard_weights", True)
        beta = params.pop("beta", 0.5)

        # Fitter data
        data, basis_matrix, weights = _fitter_data(experiment_data._data, standard_weights, beta)

        rho = lstsq_fit(data, basis_matrix, weights=weights, **params)
        analysis_result = AnalysisResult({"value": DensityMatrix(rho)})
        return analysis_result, None


def _fitter_data(experiment_data, standard_weights, beta):
    """Generate tomography fitter data from a tomography data dictionary."""
    # Get basis matrix functions
    measurement = PauliBasis.measurement_matrix

    data = []
    basis_blocks = []
    if standard_weights:
        weights = []
    else:
        weights = None

    for datum in experiment_data:
        label = datum["metadata"]["meas_basis"]
        ctkeys = count_keys(len(label))
        cts = Counts(datum["counts"])

        # Convert counts dict to numpy array
        if isinstance(cts, dict):
            cts = np.array([cts.get(key, 0) for key in ctkeys])

        # Get probabilities
        shots = np.sum(cts)
        probs = np.array(cts) / shots
        data += list(probs)

        # Compute binomial weights
        if standard_weights is True:
            wts = _binomial_weights(cts, beta)
            weights += list(wts)

        # Get reconstruction basis operators
        block = _basis_operator_matrix([mop for mop in _measurement_ops(label, measurement)])
        basis_blocks.append(block)

    return data, np.vstack(basis_blocks), weights


def _binomial_weights(counts, beta=0.5) -> np.array:
    """
    Compute binomial weights for list or dictionary of counts.

    Args:
        counts (Counts): A set of measurement counts for
            all outcomes of a given measurement configuration.
        beta (float): (default: 0.5) A nonnegative hedging parameter used to bias
        probabilities computed from input counts away from 0 or 1.

    Returns:
        The binomial weights for the input counts and beta parameter.
    Raises:
        ValueError: In case beta is negative.
    Additional Information:

        The weights are determined by
            w[i] = sqrt(shots / p[i] * (1 - p[i]))
            p[i] = (counts[i] + beta) / (shots + K * beta)
        where
            `shots` is the sum of all counts in the input
            `p` is the hedged probability computed for a count
            `K` is the total number of possible measurement outcomes.
    """

    # Sort counts if input is a dictionary
    if isinstance(counts, dict):
        mcts = marginal_counts(counts, pad_zeros=True)
        ordered_keys = sorted(list(mcts))
        counts = np.array([mcts[k] for k in ordered_keys])
    # Assume counts are already sorted if a list
    else:
        counts = np.array(counts)
    shots = np.sum(counts)

    # If beta is 0 check if we would be dividing by zero
    # If so change beta value and log warning.

    if beta < 0:
        raise ValueError("beta = {} must be non-negative.".format(beta))
    if beta == 0 and (shots in counts or 0 in counts):
        beta = 0.5

    outcomes_num = len(counts)
    # Compute hedged frequencies which are shifted to never be 0 or 1.
    freqs_hedged = (counts + beta) / (shots + outcomes_num * beta)

    # Return gaussian weights for 2-outcome measurements.
    return np.sqrt(shots / (freqs_hedged * (1 - freqs_hedged)))


def _basis_operator_matrix(basis) -> np.array:
    """Return a basis measurement matrix of the input basis.

    Args:
        basis (list[np.ndarray]): a list of basis matrices.

    Returns:
        A numpy array of shape (n, col * row) where n is the number
        of operators of shape (row, col) in `basis`.
    """
    # Dimensions
    num_ops = len(basis)
    nrows, ncols = basis[0].shape
    size = nrows * ncols

    ret = np.zeros((num_ops, size), dtype=complex)
    for j, b in enumerate(basis):
        ret[j] = np.array(b).reshape((1, size), order="F").conj()
    return ret


def _measurement_ops(label, meas_matrix_fn):
    """
    Return a list multi-qubit matrices for a measurement label.

    Args:
        label (str): a measurement configuration label for a
            tomography circuit.
        meas_matrix_fn (callable): a function that returns the matrix
            corresponding to a single qubit measurement label
            for a given outcome. The functions should have
            signature meas_matrix_fn(str, int) -> np.array

    Returns:
        list: A list of Numpy array for the multi-qubit measurement operators
              for all measurement outcomes for the measurement basis specified
              by the label. These are ordered in increasing binary order. Eg for
              2-qubits the returned matrices correspond to outcomes
              [00, 01, 10, 11]

    Additional Information:
        See the Pauli measurement function `pauli_measurement_matrix`
        for an example.
    """
    num_qubits = len(label)
    meas_ops = []

    # Construct measurement POVM for all measurement outcomes for a given
    # measurement label. This will be a list of 2 ** n operators.

    for outcomes in sorted(it.product((0, 1), repeat=num_qubits)):
        op = np.eye(1, dtype=complex)
        # Reverse label to correspond to QISKit bit ordering
        for m, outcome in zip(reversed(label), outcomes):
            op = np.kron(op, meas_matrix_fn(m, outcome))
        meas_ops.append(op)
    return meas_ops
