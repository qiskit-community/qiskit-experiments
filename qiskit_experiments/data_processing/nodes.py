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

"""Different data analysis steps."""

from abc import abstractmethod
from numbers import Number
from typing import List, Union, Sequence

import numpy as np
from uncertainties import unumpy as unp, ufloat

from qiskit_experiments.data_processing.data_action import DataAction, TrainableDataAction
from qiskit_experiments.data_processing.exceptions import DataProcessorError


class AverageData(DataAction):
    """A node to average data representable as numpy arrays."""

    def __init__(self, axis: int, validate: bool = True):
        r"""Initialize a data averaging node.

        Args:
            axis: The axis along which to average.
            validate: If set to False the DataAction will not validate its input.

        Notes:
            If the input array has no standard error, then this node will compute the
            standard error of the mean, i.e. the standard deviation of the datum divided by
            :math:`\sqrt{N}` where :math:`N` is the number of data points.
            Otherwise the standard error is given by the square root of :math:`N^{-1}` times
            the sum of the squared errors.
        """
        super().__init__(validate)
        self._axis = axis

    def _format_data(self, data: np.ndarray) -> np.ndarray:
        """Format the data into numpy arrays.

        Args:
            data: A data array to format. This is a single numpy array containing
                all circuit results input to the data processor.

        Returns:
            The data that has been validated and formatted.

        Raises:
            DataProcessorError: When the specified axis does not exist in given array.
        """
        if self._validate:
            if len(data.shape) <= self._axis:
                raise DataProcessorError(
                    f"Cannot average the {len(data.shape)} dimensional "
                    f"array along axis {self._axis}."
                )

        return data

    def _process(self, data: np.ndarray) -> np.ndarray:
        """Average the data.

        Args:
            data: A data array to process. This is a single numpy array containing
                all circuit results input to the data processor.

        Returns:
             Arrays with one less dimension than the given data.
        """
        ax = self._axis

        reduced_array = np.mean(data, axis=ax)
        nominals = unp.nominal_values(reduced_array)
        errors = unp.std_devs(reduced_array)

        if np.any(np.isnan(errors)):
            # replace empty elements with SEM
            sem = np.std(unp.nominal_values(data), axis=ax) / np.sqrt(data.shape[ax])
            errors = np.where(np.isnan(errors), sem, errors)

        return unp.uarray(nominals, errors)


class MinMaxNormalize(DataAction):
    """Normalizes the data."""

    def _process(self, data: np.ndarray) -> np.ndarray:
        """Normalize the data to the interval [0, 1].

        Args:
            data: A data array to process. This is a single numpy array containing
                all circuit results input to the data processor.

        Returns:
            The normalized data.

        Notes:
            This doesn't consider the uncertainties of the minimum or maximum values.
            Input data array is just scaled by the data range.
        """
        nominals = unp.nominal_values(data)
        min_y, max_y = np.min(nominals), np.max(nominals)

        return (data - min_y) / (max_y - min_y)


class SVD(TrainableDataAction):
    """Singular Value Decomposition of averaged IQ data."""

    def __init__(self, validate: bool = True):
        """Create new action.

        Args:
            validate: If set to False the DataAction will not validate its input.
        """
        super().__init__(validate=validate)
        self._main_axes = None
        self._means = None
        self._scales = None
        self._n_circs = 0
        self._n_shots = 0
        self._n_slots = 0
        self._n_iq = 0

    def _format_data(self, data: np.ndarray) -> np.ndarray:
        """Check that the IQ data is 2D and convert it to a numpy array.

        Args:
            data: A data array to format. This is a single numpy array containing
                all circuit results input to the data processor.
                This data has different dimensions depending on whether
                single-shot or averaged data is being processed.
                Single-shot data is four dimensional, i.e., ``[#circuits, #shots, #slots, 2]``,
                while averaged IQ data is three dimensional, i.e., ``[#circuits, #slots, 2]``.
                Here, ``#slots`` is the number of classical registers used in the circuit.

        Returns:
            data and any error estimate as a numpy array.

        Raises:
            DataProcessorError: If the datum does not have the correct format.
        """
        self._n_circs = 0
        self._n_shots = 0
        self._n_slots = 0
        self._n_iq = 0

        # identify shape
        try:
            # level1 single-shot data
            self._n_circs, self._n_shots, self._n_slots, self._n_iq = data.shape
        except ValueError:
            try:
                # level1 data averaged over shots
                self._n_circs, self._n_slots, self._n_iq = data.shape
            except ValueError as ex:
                raise DataProcessorError(
                    f"Data given to {self.__class__.__name__} is not likely level1 data."
                ) from ex

        if self._validate:
            if self._n_iq != 2:
                raise DataProcessorError(
                    f"IQ data given to {self.__class__.__name__} does not have two-dimensions "
                    f"(I and Q). Instead, {self._n_iq} dimensions were found."
                )

        return data

    @property
    def axis(self) -> List[np.array]:
        """Return the axis of the trained SVD"""
        return self._main_axes

    def means(self, qubit: int, iq_index: int) -> float:
        """Return the mean by which to correct the IQ data.

        Before training the SVD the mean of the training data is subtracted from the
        training data to avoid large offsets in the data. These means can be retrieved
        with this function.

        Args:
            qubit: Index of the qubit.
            iq_index: Index of either the in-phase (i.e. 0) or the quadrature (i.e. 1).

        Returns:
            The mean that was determined during training for the given qubit and IQ index.
        """
        return self._means[qubit][iq_index]

    @property
    def scales(self) -> List[float]:
        """Return the scaling of the SVD."""
        return self._scales

    @property
    def is_trained(self) -> bool:
        """Return True is the SVD has been trained.

        Returns:
            True if the SVD has been trained.
        """
        return self._main_axes is not None

    def _process(self, data: np.ndarray) -> np.ndarray:
        """Project the IQ data onto the axis defined by an SVD and scale it.

        Args:
            data: A data array to process. This is a single numpy array containing
                all circuit results input to the data processor.

        Returns:
            A Tuple of 1D arrays of the result of the SVD and the associated error. Each entry
            is the real part of the averaged IQ data of a qubit.

        Raises:
            DataProcessorError: If the SVD has not been previously trained on data.
        """
        if not self.is_trained:
            raise DataProcessorError("SVD must be trained on data before it can be used.")

        # IQ axis is reduced by projection
        if self._n_shots == 0:
            # level1 average mode
            dims = self._n_circs, self._n_slots
        else:
            # level1 single mode
            dims = self._n_circs, self._n_shots, self._n_slots

        projected_data = np.zeros(dims, dtype=object)

        for idx in range(self._n_slots):
            scale = self.scales[idx]
            # error propagation is computed from data if any std error exists
            centered = np.array(
                [data[..., idx, iq] - self.means(qubit=idx, iq_index=iq) for iq in [0, 1]]
            )
            projected_data[..., idx] = (self._main_axes[idx] @ centered) / scale

        return projected_data

    def train(self, data: np.ndarray):
        """Train the SVD on the given data.

        Each element of the given data will be converted to a 2D array of dimension
        n_qubits x 2. The number of qubits is inferred from the shape of the data.
        For each qubit the data is collected into an array of shape 2 x n_data_points.
        The mean of the in-phase a quadratures is subtracted before passing the data
        to numpy's svd function. The dominant axis and the scale is saved for each
        qubit so that future data points can be projected onto the axis.

        Args:
            data: A data array to be trained. This is a single numpy array containing
                all circuit results input to the data processor.
        """
        if data is None:
            return

        # TODO do not remove standard error. Currently svd is not supported.
        data = unp.nominal_values(self._format_data(data))

        self._main_axes = []
        self._scales = []
        self._means = []

        for idx in range(self._n_slots):
            datums = np.vstack([datum[idx] for datum in data]).T

            # Calculate the mean of the data to recenter it in the IQ plane.
            mean_i = np.average(datums[0, :])
            mean_q = np.average(datums[1, :])

            self._means.append((mean_i, mean_q))

            datums[0, :] = datums[0, :] - mean_i
            datums[1, :] = datums[1, :] - mean_q

            mat_u, mat_s, _ = np.linalg.svd(datums)

            self._main_axes.append(mat_u[:, 0])
            self._scales.append(mat_s[0])


class IQPart(DataAction):
    """Abstract class for IQ data post-processing."""

    def __init__(self, scale: float = 1.0, validate: bool = True):
        """
        Args:
            scale: Float with which to multiply the IQ data. Defaults to 1.0.
            validate: If set to False the DataAction will not validate its input.
        """
        self.scale = scale
        super().__init__(validate)

    @abstractmethod
    def _process(self, data: np.ndarray) -> np.ndarray:
        """Defines how the IQ point is processed.

        The last dimension of the array should correspond to [real, imaginary] part of data.

        Args:
            data: A data array to process. This is a single numpy array containing
                all circuit results input to the data processor.

        Returns:
            The data that has been processed.
        """

    def _format_data(self, data: np.ndarray) -> np.ndarray:
        """Format and validate the input.

        Args:
            data: A data array to format. This is a single numpy array containing
                all circuit results input to the data processor.

        Returns:
            The data that has been validated and formatted.

        Raises:
            DataProcessorError: When input data is not likely IQ data.
        """
        if self._validate:
            if data.shape[-1] != 2:
                raise DataProcessorError(
                    f"IQ data given to {self.__class__.__name__} must be a multi-dimensional array"
                    "of dimension [d0, d1, ..., 2] in which the last dimension "
                    "corresponds to IQ elements."
                    f"Input data contains element with length {data.shape[-1]} != 2."
                )

        return data

    def __repr__(self):
        """String representation of the node."""
        return f"{self.__class__.__name__}(validate: {self._validate}, scale: {self.scale})"


class ToReal(IQPart):
    """IQ data post-processing. Isolate the real part of single-shot IQ data."""

    def _process(self, data: np.ndarray) -> np.ndarray:
        """Take the real part of the IQ data.

        Args:
            data: An N-dimensional array of complex IQ point as [real, imaginary].

        Returns:
            A N-1 dimensional array, each entry is the real part of the given IQ data.
        """
        return data[..., 0] * self.scale


class ToImag(IQPart):
    """IQ data post-processing. Isolate the imaginary part of single-shot IQ data."""

    def _process(self, data: np.ndarray) -> np.ndarray:
        """Take the imaginary part of the IQ data.

        Args:
            data: An N-dimensional array of complex IQ point as [real, imaginary].

        Returns:
            A N-1 dimensional array, each entry is the imaginary part of the given IQ data.
        """
        return data[..., 1] * self.scale


class Probability(DataAction):
    r"""Compute the mean probability of a single measurement outcome from counts.

    This node returns the mean and standard deviation of a single measurement
    outcome probability :math:`p` estimated from the observed counts. The mean and
    variance are computed from the posterior Beta distribution
    :math:`B(\alpha_0^\prime,\alpha_1^\prime)` estimated from a Bayesian update
    of a prior Beta distribution :math:`B(\alpha_0, \alpha_1)` given the observed
    counts.

    The mean and variance of the Beta distribution :math:`B(\alpha_0, \alpha_1)` are:

    .. math::

        \text{E}[p] = \frac{\alpha_0}{\alpha_0 + \alpha_1}, \quad
        \text{Var}[p] = \frac{\text{E}[p] (1 - \text{E}[p])}{\alpha_0 + \alpha_1 + 1}

    Given a prior Beta distribution :math:`B(\alpha_0, \alpha_1)`, the posterior
    distribution for the observation of :math:`F` counts of a given
    outcome out of :math:`N` total shots is a
    :math:`B(\alpha_0^\prime,\alpha_1^\prime):math:` with

    .. math::
        \alpha_0^\prime = \alpha_0 + F, \quad
        \alpha_1^\prime = \alpha_1 + N - F.

    .. note::

        The default value for the prior distribution is *Jeffery's Prior*
        :math:`\alpha_0 = \alpha_1 = 0.5` which represents ignorance about the true
        probability value. Note that for this prior the mean probability estimate
        from a finite number of counts can never be exactly 0 or 1. The estimated
        mean and variance are given by

        .. math::

            \text{E}[p] = \frac{F + 0.5}{N + 1}, \quad
            \text{Var}[p] = \frac{\text{E}[p] (1 - \text{E}[p])}{N + 2}

        This node will deprecate standard error provided by the previous node.
    """

    def __init__(
        self,
        outcome: str,
        alpha_prior: Union[float, Sequence[float]] = 0.5,
        validate: bool = True,
    ):
        """Initialize a counts to probability data conversion.

        Args:
            outcome: The bitstring for which to return the probability and variance.
            alpha_prior: A prior Beta distribution parameter ``[`alpha0, alpha1]``.
                         If specified as float this will use the same value for
                         ``alpha0`` and``alpha1`` (Default: 0.5).
            validate: If set to False the DataAction will not validate its input.

        Raises:
            DataProcessorError: When the dimension of the prior and expected parameter vector
                do not match.
        """
        self._outcome = outcome
        if isinstance(alpha_prior, Number):
            self._alpha_prior = [alpha_prior, alpha_prior]
        else:
            if validate and len(alpha_prior) != 2:
                raise DataProcessorError(
                    "Prior for probability node must be a float or pair of floats."
                )
            self._alpha_prior = list(alpha_prior)

        super().__init__(validate)

    def _format_data(self, data: np.ndarray) -> np.ndarray:
        """
        Checks that the given data has a counts format.

        Args:
            data: A data array to format. This is a single numpy array containing
                all circuit results input to the data processor.
                This is usually an object data type containing Python dictionaries of
                count data keyed on the measured bitstring.
                A count value is a discrete quantity representing the frequency of an event.
                Therefore, count values do not have an uncertainty.

        Returns:
            The ``data`` as given.

        Raises:
            DataProcessorError: If the data is not a counts dict or a list of counts dicts.
        """
        valid_count_type = int, np.integer

        if self._validate:
            for datum in data:
                if not isinstance(datum, dict):
                    raise DataProcessorError(
                        f"Data entry must be dictionary of counts, received {type(datum)}."
                    )
                for bit_str, count in datum.items():
                    if not isinstance(bit_str, str):
                        raise DataProcessorError(
                            f"Key {bit_str} is not a valid count key in {self.__class__.__name__}."
                        )
                    if not isinstance(count, valid_count_type):
                        raise DataProcessorError(
                            f"Count {bit_str} is not a valid count for {self.__class__.__name__}. "
                            "The uncertainty of probability is computed based on sampling error, "
                            "thus the count should be an error-free discrete quantity "
                            "representing the frequency of event."
                        )

        return data

    def _process(self, data: np.ndarray) -> np.ndarray:
        """Compute mean and standard error from the beta distribution.

        Args:
            data: A data array to process. This is a single numpy array containing
                all circuit results input to the data processor.
                This is usually an object data type containing Python dictionaries of
                count data keyed on the measured bitstring.

        Returns:
            The data that has been processed.
        """
        probabilities = np.empty(data.size, dtype=object)

        for idx, counts_dict in enumerate(data):
            shots = sum(counts_dict.values())
            freq = counts_dict.get(self._outcome, 0)
            alpha_posterior = [freq + self._alpha_prior[0], shots - freq + self._alpha_prior[1]]
            alpha_sum = sum(alpha_posterior)

            p_mean = alpha_posterior[0] / alpha_sum
            p_var = p_mean * (1 - p_mean) / (alpha_sum + 1)

            probabilities[idx] = ufloat(nominal_value=p_mean, std_dev=np.sqrt(p_var))

        return probabilities


class BasisExpectationValue(DataAction):
    """Compute expectation value of measured basis from probability.

    Note:
        The sign becomes P(0) -> 1, P(1) -> -1.
    """

    def _format_data(self, data: np.ndarray) -> np.ndarray:
        """Format and validate the input.

        Args:
            data: A data array to format. This is a single numpy array containing
                all circuit results input to the data processor.

        Returns:
            The data that has been validated and formatted.

        Raises:
            DataProcessorError: When input value is not in [0, 1]
        """
        if self._validate:
            if not all(0.0 <= p <= 1.0 for p in data):
                raise DataProcessorError(
                    f"Input data for node {self.__class__.__name__} is not likely probability."
                )

        return data

    def _process(self, data: np.ndarray) -> np.ndarray:
        """Compute basis eigenvalue.

        Args:
            data: A data array to process. This is a single numpy array containing
                all circuit results input to the data processor.

        Returns:
            The data that has been processed.
        """
        return 2 * (0.5 - data)
