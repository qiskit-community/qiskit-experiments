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
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from qiskit_experiments.data_processing.data_action import DataAction, TrainableDataAction
from qiskit_experiments.data_processing.exceptions import DataProcessorError


class AverageData(DataAction):
    """A node to average data representable as numpy arrays."""

    def __init__(self, axis: int = 0):
        """Initialize a data averaging node.

        Args:
            axis: The axis along which to average the data. If not given 0 is the
                default axis.
        """
        super().__init__()
        self._axis = axis

    def _format_data(self, datum: Any, error: Optional[Any] = None):
        """Format the data into numpy arrays."""
        datum = np.asarray(datum, dtype=float)

        if error is not None:
            error = np.asarray(error, dtype=float)

        return datum, error

    def _process(
        self, datum: np.array, error: Optional[np.array] = None
    ) -> Tuple[np.array, np.array]:
        """Average the data.

         Args:
             datum: an array of data.

         Returns:
             Two arrays with one less dimension than the given datum and error. The error
             is the standard error of the mean, i.e. the standard deviation of the datum
             divided by :math:`sqrt{N}` where :math:`N` is the number of data points.

        Raises:
            DataProcessorError: If the axis is not an int.
        """
        standard_error = np.std(datum, axis=self._axis) / np.sqrt(datum.shape[0])

        return np.average(datum, axis=self._axis), standard_error


class SVD(TrainableDataAction):
    """Singular Value Decomposition of averaged IQ data."""

    def __init__(self, validate: bool = True):
        """
        Args:
            validate: If set to False the DataAction will not validate its input.
        """
        super().__init__(validate=validate)
        self._main_axes = None
        self._means = None
        self._scales = None

    def _format_data(self, datum: Any, error: Optional[Any] = None) -> Tuple[Any, Any]:
        """Check that the IQ data is 2D and convert it to a numpy array.

        Args:
            datum: A single item of data which corresponds to single-shot IQ data.

        Returns:
            datum and any error estimate as a numpy array.

        Raises:
            DataProcessorError: If the datum does not have the correct format.
        """
        datum = np.asarray(datum, dtype=float)

        if error is not None:
            error = np.asarray(error, dtype=float)

        if self._validate:
            if len(datum.shape) != 2:
                raise DataProcessorError(
                    f"IQ data given to {self.__class__.__name__} must be an 2D array. "
                    f"Instead, a {len(datum.shape)}D array was given."
                )

            if error is not None and len(error.shape) != 2:
                raise DataProcessorError(
                    f"IQ data error given to {self.__class__.__name__} must be an 2D array."
                    f"Instead, a {len(error.shape)}D array was given."
                )

        return datum, error

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

    def _process(
        self, datum: np.array, error: Optional[np.array] = None
    ) -> Tuple[np.array, np.array]:
        """Project the IQ data onto the axis defined by an SVD and scale it.

        Args:
            datum: A 2D array of qubits, and an average complex IQ point as [real, imaginary].
            error: An optional 2D array of qubits, and an error on an average complex IQ
                point as [real, imaginary].

        Returns:
            A Tuple of 1D arrays of the result of the SVD and the associated error. Each entry
            is the real part of the averaged IQ data of a qubit.

        Raises:
            DataProcessorError: If the SVD has not been previously trained on data.
        """

        if not self.is_trained:
            raise DataProcessorError("SVD must be trained on data before it can be used.")

        n_qubits = datum.shape[0]
        processed_data = []

        if error is not None:
            processed_error = []
        else:
            processed_error = None

        # process each averaged IQ point with its own axis.
        for idx in range(n_qubits):

            centered = np.array(
                [datum[idx][iq] - self.means(qubit=idx, iq_index=iq) for iq in [0, 1]]
            )

            processed_data.append((self._main_axes[idx] @ centered) / self.scales[idx])

            if error is not None:
                angle = np.arctan(self._main_axes[idx][1] / self._main_axes[idx][0])
                error_value = np.sqrt(
                    (error[idx][0] * np.cos(angle)) ** 2 + (error[idx][1] * np.sin(angle)) ** 2
                )
                processed_error.append(error_value)

        return np.array(processed_data), processed_error

    def train(self, data: List[Any]):
        """Train the SVD on the given data.

        Each element of the given data will be converted to a 2D array of dimension
        n_qubits x 2. The number of qubits is inferred from the shape of the data.
        For each qubit the data is collected into an array of shape 2 x n_data_points.
        The mean of the in-phase a quadratures is subtracted before passing the data
        to numpy's svd function. The dominant axis and the scale is saved for each
        qubit so that future data points can be projected onto the axis.

        Args:
            data: A list of datums. Each datum will be converted to a 2D array.
        """
        if not data:
            return

        n_qubits = self._format_data(data[0])[0].shape[0]

        self._main_axes = []
        self._scales = []
        self._means = []

        for qubit_idx in range(n_qubits):
            datums = np.vstack([self._format_data(datum)[0][qubit_idx] for datum in data]).T

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
    def _process(self, datum: np.array, error: Optional[np.array] = None) -> np.array:
        """Defines how the IQ point is processed.

        Args:
            datum: A 2D or a 3D array of complex IQ points as [real, imaginary].
            error: A 2D or a 3D array of errors on complex IQ points as [real, imaginary].

        Returns:
            Processed IQ point and its associated error estimate.
        """

    def _format_data(self, datum: Any, error: Optional[Any] = None) -> Tuple[Any, Any]:
        """Check that the IQ data has the correct format and convert to numpy array.

        Args:
            datum: A single item of data which corresponds to single-shot IQ data. It's
                dimension will depend on whether it is single-shot IQ data (three-dimensional)
                or averaged IQ date (two-dimensional).

        Returns:
            datum and any error estimate as a numpy array.

        Raises:
            DataProcessorError: If the datum does not have the correct format.
        """
        datum = np.asarray(datum, dtype=float)

        if error is not None:
            error = np.asarray(error, dtype=float)

        if self._validate:
            if len(datum.shape) not in {2, 3}:
                raise DataProcessorError(
                    f"IQ data given to {self.__class__.__name__} must be an N dimensional"
                    f"array with N in (2, 3). Instead, a {len(datum.shape)}D array was given."
                )

            if error is not None and len(error.shape) not in {2, 3}:
                raise DataProcessorError(
                    f"IQ data error given to {self.__class__.__name__} must be an N dimensional"
                    f"array with N in (2, 3). Instead, a {len(error.shape)}D array was given."
                )

            if error is not None and len(error.shape) != len(datum.shape):
                raise DataProcessorError(
                    "Datum and error do not have the same shape: "
                    f"{len(datum.shape)} != {len(error.shape)}."
                )

        return datum, error

    def __repr__(self):
        """String representation of the node."""
        return f"{self.__class__.__name__}(validate: {self._validate}, scale: {self.scale})"


class ToReal(IQPart):
    """IQ data post-processing. Isolate the real part of single-shot IQ data."""

    def _process(
        self, datum: np.array, error: Optional[np.array] = None
    ) -> Tuple[np.array, np.array]:
        """Take the real part of the IQ data.

        Args:
            datum: A 2D or 3D array of shots, qubits, and a complex IQ point as [real, imaginary].
            error: An optional 2D or 3D array of shots, qubits, and an error on a complex IQ point
                as [real, imaginary].

        Returns:
            A 1D or 2D array, each entry is the real part of the given IQ data and error.
        """
        if error is not None:
            return datum[..., 0] * self.scale, error[..., 0] * self.scale
        else:
            return datum[..., 0] * self.scale, None


class ToImag(IQPart):
    """IQ data post-processing. Isolate the imaginary part of single-shot IQ data."""

    def _process(self, datum: np.array, error: Optional[np.array] = None) -> np.array:
        """Take the imaginary part of the IQ data.

        Args:
            datum: A 2D or 3D array of shots, qubits, and a complex IQ point as [real, imaginary].
            error: An optional 2D or 3D array of shots, qubits, and an error on a complex IQ point
                as [real, imaginary].

        Returns:
            A 1D or 2D array, each entry is the imaginary part of the given IQ data and error.
        """
        if error is not None:
            return datum[..., 1] * self.scale, error[..., 1] * self.scale
        else:
            return datum[..., 1] * self.scale, None


class Probability(DataAction):
    """Count data post processing. This returns the probabilities of the outcome string
    used to initialize an instance of Probability."""

    def __init__(self, outcome: str = "1", validate: bool = True):
        """Initialize a counts to probability data conversion.

        Args:
            outcome: The bitstring for which to compute the probability which defaults to "1".
            validate: If set to False the DataAction will not validate its input.
        """
        self._outcome = outcome
        super().__init__(validate)

    def _format_data(self, datum: dict, error: Optional[Any] = None) -> Tuple[dict, Any]:
        """
        Checks that the given data has a counts format.

        Args:
            datum: An instance of data the should be a dict with bit strings as keys
                and counts as values.

        Returns:
            The datum as given.

        Raises:
            DataProcessorError: if the data is not a counts dict.
        """
        if self._validate:
            if not isinstance(datum, dict):
                raise DataProcessorError(
                    f"Given counts datum {datum} to "
                    f"{self.__class__.__name__} is not a valid count format."
                )

            for bit_str, count in datum.items():
                if not isinstance(bit_str, str):
                    raise DataProcessorError(
                        f"Key {bit_str} is not a valid count key in{self.__class__.__name__}."
                    )

                if not isinstance(count, (int, float)):
                    raise DataProcessorError(
                        f"Count {bit_str} is not a valid count value in {self.__class__.__name__}."
                    )

        return datum, None

    def _process(self, datum: Dict[str, Any], error: Optional[Dict] = None) -> Tuple[float, float]:
        """
        Args:
            datum: The data dictionary,taking the data under counts and
                adding the corresponding probabilities.

        Returns:
            processed data: A dict with the populations.
        """

        shots = sum(datum.values())
        p_mean = datum.get(self._outcome, 0.0) / shots
        p_var = p_mean * (1 - p_mean) / shots

        return p_mean, p_var
