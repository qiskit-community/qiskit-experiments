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
from abc import ABC
from enum import Enum
from numbers import Number
from typing import List, Union, Sequence, Set
from collections import defaultdict

import numpy as np
from uncertainties import unumpy as unp, ufloat

from qiskit.result.postprocess import format_counts_memory
from qiskit.utils import deprecate_func
from qiskit_experiments.data_processing.data_action import DataAction, TrainableDataAction
from qiskit_experiments.data_processing.exceptions import DataProcessorError
from qiskit_experiments.data_processing.discriminator import BaseDiscriminator
from qiskit_experiments.framework import Options


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
        with np.errstate(invalid="ignore"):
            # Setting std_devs to NaN will trigger floating point exceptions
            # which we can ignore. See https://stackoverflow.com/q/75656026
            errors = unp.std_devs(reduced_array)

        if np.any(np.isnan(errors)):
            # replace empty elements with SEM
            sem = np.std(unp.nominal_values(data), axis=ax) / np.sqrt(data.shape[ax])
            errors = np.where(np.isnan(errors), sem, errors)

        return unp.uarray(nominals, errors)

    def __repr__(self):
        """String representation of the node."""
        return f"{self.__class__.__name__}(validate={self._validate}, axis={self._axis})"


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
        self._n_circs = 0
        self._n_shots = 0
        self._n_slots = 0
        self._n_iq = 0

    @classmethod
    def _default_parameters(cls) -> Options:
        """Default parameters.

        Parameters are defined for each qubit in the data and thus
        represented as an array-like.

        Trainable parameters:
            main_axes: A unit vector representing the main axis in IQ plane.
            i_means: Mean value of training data along I quadrature.
            q_means: Mean value of training data along Q quadrature.
            scales: Scaling of IQ signal.
        """
        params = super()._default_parameters()
        params.main_axes = None
        params.i_means = None
        params.q_means = None
        params.scales = None

        return params

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

    def _process(self, data: np.ndarray) -> np.ndarray:
        """Project the IQ data onto the axis defined by an SVD and scale it.

        Args:
            data: A data array to process. This is a single numpy array containing
                all circuit results input to the data processor.

        Returns:
            A Tuple of 1D arrays of the result of the SVD and the associated error. Each entry
            is the real part of the averaged IQ data of a qubit. The data has the shape
            n_circuits x n_slots for averaged data and n_circuits x n_shots x n_slots for
            single-shot data.

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
            scale = self.parameters.scales[idx]
            axis = self.parameters.main_axes[idx]
            mean_i = self.parameters.i_means[idx]
            mean_q = self.parameters.q_means[idx]

            if self._n_shots != 0:
                # Single shot
                for circ_idx in range(self._n_circs):
                    centered = [
                        data[circ_idx, :, idx, 0] - mean_i,
                        data[circ_idx, :, idx, 1] - mean_q,
                    ]
                    projected_data[circ_idx, :, idx] = axis @ np.array(centered) / scale
            else:
                # Averaged
                centered = [data[:, idx, 0] - mean_i, data[:, idx, 1] - mean_q]
                projected_data[:, idx] = axis @ np.array(centered) / scale

        return projected_data

    def train(self, data: np.ndarray):
        """Train the SVD on the given data.

        Each element of the given data will be converted to a 2D array of dimension
        n_qubits x 2. The number of qubits is inferred from the shape of the data.
        For each qubit the data is collected into an array of shape 2 x n_data_points.
        The mean of the in-phase a quadratures is subtracted before passing the data
        to numpy's svd function. The dominant axis and the scale is saved for each
        qubit so that future data points can be projected onto the axis.

        .. note::

            Before training the SVD the mean of the training data is subtracted from the
            training data to avoid large offsets in the data.
            These means can be retrieved with :attr:`.parameters.i_means` or
            :attr:`parameters.q_means` for I and Q quadrature, respectively.

        Args:
            data: A data array to be trained. This is a single numpy array containing
                all circuit results input to the data processor.
        """
        if data is None:
            return

        # TODO do not remove standard error. Currently svd is not supported.
        data = unp.nominal_values(self._format_data(data))

        main_axes = []
        scales = []
        i_means = []
        q_means = []
        for idx in range(self._n_slots):
            datums = np.vstack([datum[idx] for datum in data]).T

            # Calculate the mean of the data to recenter it in the IQ plane.
            mean_i = np.average(datums[0, :])
            mean_q = np.average(datums[1, :])
            i_means.append(mean_i)
            q_means.append(mean_q)

            datums[0, :] = datums[0, :] - mean_i
            datums[1, :] = datums[1, :] - mean_q

            mat_u, mat_s, _ = np.linalg.svd(datums)

            # There is an arbitrary sign in the direction of the matrix which we fix to
            # positive to make the SVD node more reliable in tests and real settings.
            main_axes.append(np.sign(mat_u[0, 0]) * mat_u[:, 0])
            scales.append(mat_s[0])

        self.set_parameters(
            main_axes=main_axes,
            scales=scales,
            i_means=i_means,
            q_means=q_means,
        )


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
        self._n_circs = 0
        self._n_shots = 0
        self._n_slots = 0
        self._n_iq = 0

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
        self._n_shots = 0

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
        return f"{self.__class__.__name__}(validate={self._validate}, scale={self.scale})"


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


class ToAbs(IQPart):
    """IQ data post-processing. Take the absolute value of the IQ point."""

    def _process(self, data: np.array) -> np.array:
        """Take the absolute value of the IQ data.

        Args:
            data: An N-dimensional array of complex IQ point as [real, imaginary].

        Returns:
            A N-1 dimensional array, each entry is the absolute value of the given IQ data.
        """
        # pylint: disable=no-member
        return unp.sqrt(data[..., 0] ** 2 + data[..., 1] ** 2) * self.scale


class DiscriminatorNode(DataAction):
    """A class to discriminate kerneled data, e.g., IQ data, to produce counts.

    This node integrates into the data processing chain a serializable
    :class:`.BaseDiscriminator` subclass instance which must have a
    :meth:`~.BaseDiscriminator.predict` method that takes as input a list of lists and
    returns a list of labels. Crucially, this node can be initialized with a single
    discriminator which applies to each memory slot or it can be initialized with a list
    of discriminators, i.e., one for each slot.

    .. note::

        Future versions may see this class become a sub-class of :class:`.TrainableDataAction`.

    .. note::

        This node will drop uncertainty from unclassified nodes.
        Returned labels don't have uncertainty.

    """

    def __init__(
        self,
        discriminators: Union[BaseDiscriminator, List[BaseDiscriminator]],
        validate: bool = True,
    ):
        """Initialize the node with an object that can discriminate.

        Args:
            discriminators: The entity that will perform the discrimination. This needs to
                be a :class:`.BaseDiscriminator` or a list thereof that takes
                as input a list of lists and returns a list of labels. If a list of
                discriminators is given then there should be as many discriminators as there
                will be slots in the memory. The discriminator at the i-th index will be applied
                to the i-th memory slot.
            validate: If set to False the DataAction will not validate its input.
        """
        super().__init__(validate)
        self._discriminator = discriminators
        self._n_circs = 0
        self._n_shots = 0
        self._n_slots = 0
        self._n_iq = 0

    def _format_data(self, data: np.ndarray) -> np.ndarray:
        """Check that there are as many discriminators as there are slots."""
        self._n_shots = 0

        # identify shape
        try:
            # level1 single-shot data
            self._n_circs, self._n_shots, self._n_slots, self._n_iq = data.shape
        except ValueError as ex:
            raise DataProcessorError(
                f"The data given to {self.__class__.__name__} does not have the shape of "
                "single-shot IQ data; expecting a 4D array."
            ) from ex

        if self._validate:
            if data.shape[-1] != 2:
                raise DataProcessorError(
                    f"IQ data given to {self.__class__.__name__} must be a multi-dimensional array"
                    "of dimension [d0, d1, ..., 2] in which the last dimension "
                    "corresponds to IQ elements."
                    f"Input data contains element with length {data.shape[-1]} != 2."
                )

        if self._validate:
            if isinstance(self._discriminator, list):
                if self._n_slots != len(self._discriminator):
                    raise DataProcessorError(
                        f"The Discriminator node has {len(self._discriminator)} which does "
                        f"not match the {self._n_slots} slots in the data."
                    )

        return unp.nominal_values(data)

    def _process(self, data: np.ndarray) -> np.ndarray:
        """Discriminate the data.

        Args:
            data: The IQ data as a list of points to discriminate. This data should have
                the shape dim_1 x dim_2 x ... x dim_k x 2.

        Returns:
            The discriminated data as a list of labels with shape dim_1 x ... x dim_k.
        """
        # Case where one discriminator is applied to all the data.
        if not isinstance(self._discriminator, list):
            # Reshape the IQ data to an array of size n x 2
            shape, data_length = data.shape, 1
            for dim in shape[:-1]:
                data_length *= dim

            data = data.reshape((data_length, 2))  # the last dim is guaranteed by _process

            # Classify the data using the discriminator and reshape it to dim_1 x ... x dim_k
            classified = np.array(self._discriminator.predict(data)).reshape(shape[0:-1])

        # case where a discriminator is applied to each slot.
        else:
            classified = np.empty((self._n_circs, self._n_shots, self._n_slots), dtype=str)
            for idx, discriminator in enumerate(self._discriminator):
                sub_data = data[:, :, idx, :].reshape((self._n_circs * self._n_shots, 2))
                sub_classified = np.array(discriminator.predict(sub_data))
                sub_classified = sub_classified.reshape((self._n_circs, self._n_shots))
                classified[:, :, idx] = sub_classified

        # Concatenate the bit-strings together.
        labeled_data = []
        for idx in range(self._n_circs):
            labeled_data.append(
                ["".join(classified[idx, jdx, :][::-1]) for jdx in range(self._n_shots)]
            )

        return np.array(labeled_data).reshape((self._n_circs, self._n_shots))


class MemoryToCounts(DataAction):
    """A data action that takes discriminated data and transforms it into a counts dict.

    This node is intended to be used after the :class:`.DiscriminatorNode` node. It will convert
    the classified memory into a list of count dictionaries wrapped in a numpy array.
    """

    def _format_data(self, data: np.ndarray) -> np.ndarray:
        """Validate the input data."""
        if self._validate:
            if len(data.shape) <= 1:
                raise DataProcessorError(
                    "The data should be an array with at least two dimensions."
                )

        return data

    def _process(self, data: np.ndarray) -> np.ndarray:
        """
        Args:
            data: The classified data to format into a counts dictionary. The first dimension
                is assumed to correspond to the different circuit executions.

        Returns:
            A list of dictionaries where each dict is a count dictionary over the labels of
            the input data.
        """
        all_counts = []
        for datum in data:
            counts = {}
            for bit_string in set(datum):
                counts[bit_string] = sum(datum == bit_string)

            all_counts.append(counts)

        return np.array(all_counts)


class CountsAction(DataAction):
    """An abstract DataAction that acts on count dictionaries."""

    @abstractmethod
    def _process(self, data: np.ndarray) -> np.ndarray:
        """Defines how the counts are processed."""

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


class MarginalizeCounts(CountsAction):
    r"""A data action to marginalize count dictionaries.

    This data action takes a count dictionary and returns a new count dictionary that
    is marginalized over a set of specified qubits. For example, given the count
    dictionary :code:`{"010": 1, "110": 10, "100": 100}` this node will return the
    count dictionary :code:`{"10": 11, "00": 100}` when marginalized over qubit 2.

    .. note::
        This data action can be used to discard one or more qubits in the counts
        dictionary. This is, for example, useful when processing two-qubit restless
        experiments but can be used in a more general context. In composite
        experiments the counts marginalization is already done in the data container.
    """

    def __init__(self, qubits_to_keep: Set[int], validate: bool = True):
        """Initialize a counts marginalization node.

        Args:
            qubits_to_keep: A set of qubits to retain during the marginalization process.
            validate: If set to False the DataAction will not validate its input.
        """
        super().__init__(validate)
        self._qubits_to_keep = sorted(qubits_to_keep, reverse=True)

    def _process(self, data: np.ndarray) -> np.ndarray:
        """Perform counts marginalization."""
        marginalized_counts = []

        for datum in data:
            new_counts = defaultdict(int)
            for bit_str, count in datum.items():
                new_counts["".join([bit_str[::-1][idx] for idx in self._qubits_to_keep])] += count

            marginalized_counts.append(new_counts)

        return np.array(marginalized_counts)

    def __repr__(self):
        """String representation of the node."""
        options_str = ", ".join(
            [
                f"qubits_to_keep={self._qubits_to_keep}",
                f"validate={self._validate}",
            ]
        )
        return f"{self.__class__.__name__}({options_str})"


class Probability(CountsAction):
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
            alpha_prior: A prior Beta distribution parameter ``[alpha0, alpha1]``.
                         If specified as float this will use the same value for
                         ``alpha0`` and ``alpha1`` (Default: 0.5).
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

            with np.errstate(invalid="ignore"):
                # Setting std_devs to NaN will trigger floating point exceptions
                # which we can ignore. See https://stackoverflow.com/q/75656026
                probabilities[idx] = ufloat(nominal_value=p_mean, std_dev=np.sqrt(p_var))

        return probabilities

    def __repr__(self):
        """String representation of the node."""
        options_str = ", ".join(
            [
                f"validate={self._validate}",
                f"outcome={self._outcome}",
                f"alpha_prior={self._alpha_prior}",
            ]
        )
        return f"{self.__class__.__name__}({options_str})"


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


class ProjectorType(Enum):
    """Types of projectors for data dimensionality reduction."""

    SVD = SVD
    ABS = ToAbs
    REAL = ToReal
    IMAG = ToImag


class ShotOrder(Enum):
    """Shot order allowed values.

    Generally, there are two possible modes in which a backend measures m
    circuits with n shots:

    - In the "circuit_first" mode, the backend subsequently first measures
      all m circuits and then repeats this n times.

    - In the "shot_first" mode, the backend first measures the 1st circuit
      n times, then the 2nd circuit n times, and it proceeds with the remaining
      circuits in the same way until it measures the m-th circuit n times.

    The current default mode of IBM Quantum devices is "circuit_first".
    """

    # pylint: disable=invalid-name
    circuit_first = "c"
    shot_first = "s"


class RestlessNode(DataAction, ABC):
    """An abstract node for restless data processing nodes.

    In restless measurements, the qubit is not reset after each measurement. Instead, the
    outcome of the previous quantum non-demolition measurement is the initial state for the
    current circuit. Restless measurements therefore require special data processing nodes
    that are implemented as sub-classes of ``RestlessNode``. Restless experiments provide a
    fast alternative for several calibration and characterization tasks, for details
    see https://arxiv.org/pdf/2202.06981.pdf.

    This node takes as input an array of arrays (2d array) where the sub-arrays are
    the memories of each measured circuit. The sub-arrays therefore have a length
    given by the number of shots. This data is reordered into a one dimensional array where
    the element at index j was the jth measured shot. This node assumes by default that
    a list of circuits :code:`[circ_1, cric_2, ..., circ_m]` is measured :code:`n_shots`
    times according to the following order:

    .. parsed-literal::

        [
            circuit 1 - shot 1,
            circuit 2 - shot 1,
            ...
            circuit m - shot 1,
            circuit 1 - shot 2,
            circuit 2 - shot 2,
            ...
            circuit m - shot 2,
            circuit 1 - shot 3,
            ...
            circuit m - shot n,
        ]

    Once the shots have been ordered in this fashion the data can be post-processed.
    """

    @deprecate_func(
        since="0.9",
        additional_msg=(
            "Restless data processing nodes will be removed from "
            "qiskit-experiments. Refer to the RestlessMixin code of "
            "qiskit-experiment 0.8 and the code for RestlessToIQ and "
            "RestlessToCounts for the way to use a custom restless "
            "processor on an experiment."
        ),
        package_name="qiskit-experiments",
    )
    def __init__(
        self, validate: bool = True, memory_allocation: ShotOrder = ShotOrder.circuit_first
    ):
        """Initialize a restless node.

        Args:
            validate: If set to True the node will validate its input.
            memory_allocation: If set to "c" the node assumes that the backend
                subsequently first measures all circuits and then repeats this
                n times, where n is the total number of shots. The default value
                is "c". If set to "s" it is assumed that the backend subsequently
                measures each circuit n times.
        """
        super().__init__(validate)
        self._n_shots = None
        self._n_circuits = None
        self._memory_allocation = memory_allocation

    def _format_data(self, data: np.ndarray) -> np.ndarray:
        """Convert the data to an array.

        This node will also set all the attributes needed to process the data such as
        the number of shots and the number of circuits.

        Args:
            data: An array representing the memory.

        Returns:
            The data that has been processed.

        Raises:
            DataProcessorError: If the datum has the wrong shape.
        """

        self._n_shots = len(data[0])
        self._n_circuits = len(data)

        if self._validate:
            if data.shape[:2] != (self._n_circuits, self._n_shots):
                raise DataProcessorError(
                    f"The datum given to {self.__class__.__name__} does not convert "
                    "of an array with dimension (number of circuit, number of shots)."
                )

        return data

    def _reorder(self, unordered_data: np.ndarray) -> np.ndarray:
        """Reorder the measured data according to the measurement sequence.

        Here, by default, it is assumed that the inner loop of the measurement
        is done over the circuits and the outer loop is done over the shots.
        The returned data is a one-dimensional array of time-ordered shots.
        """
        if unordered_data is None:
            return unordered_data

        if self._memory_allocation == ShotOrder.circuit_first:
            return unordered_data.T.flatten()
        else:
            return unordered_data.flatten()


class RestlessToCounts(RestlessNode):
    """Post-process restless data and convert restless memory to counts.

    This node first orders the measured restless data according to the measurement
    sequence and then compares each bit in a shot with its value in the previous shot.
    If they are the same then the bit corresponds to a 0, i.e. no state change, and if
    they are different then the bit corresponds to a 1, i.e. there was a state change.
    """

    def __init__(self, num_qubits: int, validate: bool = True):
        """
        Args:
            num_qubits: The number of qubits which is needed to construct the header needed
                by :code:`qiskit.result.postprocess.format_counts_memory` to convert the memory
                into a bit-string of counts.
            validate: If set to False the DataAction will not validate its input.
        """
        super().__init__(validate)
        self._num_qubits = num_qubits

    def _process(self, data: np.ndarray) -> np.ndarray:
        """Reorder the shots and assign values to them based on the previous outcome.

        Args:
            data: An array representing the memory.

        Returns:
            A counts dictionary processed according to the restless methodology.
        """

        # Step 1. Reorder the data.
        memory = self._reorder(data)

        # Step 2. Do the restless classification into counts.
        counts = [defaultdict(int) for _ in range(self._n_circuits)]
        prev_shot = "0" * self._num_qubits
        header = {"memory_slots": self._num_qubits}

        for idx, shot in enumerate(memory):
            shot = format_counts_memory(shot, header)

            restless_adjusted_shot = RestlessToCounts._restless_classify(shot, prev_shot)

            circuit_idx = idx % self._n_circuits

            counts[circuit_idx][restless_adjusted_shot] += 1

            prev_shot = shot

        return np.array([dict(counts_dict) for counts_dict in counts])

    @staticmethod
    def _restless_classify(shot: str, prev_shot: str) -> str:
        """Adjust the measured shot based on the previous shot.

        Each bit in shot is compared to its value in the previous shot. If both are equal
        the restless adjusted bit is 0 (no state change) otherwise it is 1 (the
        qubit changed state). This corresponds to taking the exclusive OR operation
        between each bit and its previous outcome.

        Args:
            shot: A measured shot as a binary string, e.g. "0110100".
            prev_shot: The shot that was measured in the previous circuit.

        Returns:
            The restless adjusted string computed by comparing the shot with the previous shot.
        """
        restless_adjusted_bits = []

        for idx, bit in enumerate(shot):
            restless_adjusted_bits.append("0" if bit == prev_shot[idx] else "1")

        return "".join(restless_adjusted_bits)


class RestlessToIQ(RestlessNode):
    """Post-process restless data and convert restless memory to IQ data.

    This node first orders the measured restless IQ point (measurement level 1) data
    according to the measurement sequence and then subtracts an IQ point from the previous
    one, i.e. :math:`(I_2 - I_1) + i(Q_2 - Q_1)` for consecutively measured IQ points
    :math:`I_1 + iQ_1` and :math:`I_2 + iQ_2`. Following this, it takes the absolute
    value of the in-phase and quadrature component and returns a sequence of circuit-
    ordered IQ values, e.g. containing :math:`|I_2 - I_1| + i|Q_2 - Q_1|`.
    This procedure is based on M. Werninghaus, et al., PRX Quantum 2, 020324 (2021).
    """

    def __init__(self, validate: bool = True):
        """
        Args:
            validate: If set to False the DataAction will not validate its input.
        """
        super().__init__(validate)

    def _process(self, data: np.ndarray) -> np.ndarray:
        """Reorder the IQ shots and assign values to them based on the previous outcome.

        Args:
            data: An array representing the memory.

        Returns:
            An array of arrays of IQ shots processed according to the restless methodology.
        """

        # Step 1. Reorder the data.
        memory = self._reorder_iq(data)

        # Step 2. Subtract and take absolute value of consecutive IQ points in
        # the reordered memory.
        post_processed_memory = np.abs(np.diff(memory, axis=0))

        # The first element of the post-processed data is the first element
        # of the reordered memory from step 1.
        post_processed_memory = np.insert(post_processed_memory, 0, memory[0], axis=0)

        # Step 3. Order post-processed IQ points by circuit.
        iq_memory = [[] for _ in range(self._n_circuits)]
        for idx, iq_point in enumerate(post_processed_memory):
            iq_memory[idx % self._n_circuits].append(iq_point)

        return np.array(iq_memory)

    def _reorder_iq(self, unordered_data: np.ndarray) -> np.ndarray:
        """Reorder IQ data according to the measurement sequence."""

        if unordered_data is None:
            return unordered_data

        ordered_data = [None] * self._n_shots * self._n_circuits

        count = 0
        if self._memory_allocation == ShotOrder.circuit_first:
            for shot_idx in range(self._n_shots):
                for circ_idx in range(self._n_circuits):
                    ordered_data[count] = unordered_data[circ_idx][shot_idx]
                    count += 1
        else:
            for circ_idx in range(self._n_circuits):
                for shot_idx in range(self._n_shots):
                    ordered_data[count] = unordered_data[circ_idx][shot_idx]
                    count += 1

        return np.array(ordered_data)
