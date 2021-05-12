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

from qiskit_experiments.data_processing.data_action import DataAction
from qiskit_experiments.data_processing.exceptions import DataProcessorError


class IQPart(DataAction):
    """Abstract class for IQ data post-processing."""

    def __init__(self, scale: Optional[float] = None, validate: bool = True):
        """
        Args:
            scale: Float with which to multiply the IQ data.
            validate: If set to False the DataAction will not validate its input.
        """
        self.scale = scale
        super().__init__(validate)

    @abstractmethod
    def _process(self, datum: np.array) -> np.array:
        """Defines how the IQ point will be processed.

        Args:
            datum: A 3D array of shots, qubits, and a complex IQ point as [real, imaginary].

        Returns:
            Processed IQ point.
        """

    def _format_data(self, datum: Any) -> Any:
        """Check that the IQ data has the correct format and convert to numpy array.

        Args:
            datum: A single item of data which corresponds to single-shot IQ data. It should
                have dimension three: shots, qubits, iq-point as [real, imaginary].

        Returns:
            datum as a numpy array.

        Raises:
            DataProcessorError: If the datum does not have the correct format.
        """
        datum = np.asarray(datum, dtype=float)

        if self._validate and len(datum.shape) != 3:
            raise DataProcessorError(
                f"Single-shot data given {self.__class__.__name__}"
                f"must be a 3D array. Instead, a {len(datum.shape)}D "
                f"array was given."
            )

        return datum

    def __repr__(self):
        """String representation of the node."""
        return f"{self.__class__.__name__}(validate: {self._validate}, scale: {self.scale})"


class SVDAvg(IQPart):
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

    @property
    def axis(self) -> List[np.array]:
        """Return the axis of the trained SVD"""
        if self._main_axes:
            return self._main_axes

        raise DataProcessorError("SVD is not trained.")

    @property
    def scales(self) -> List[float]:
        """Return the scaling of the SVD."""
        if self._scales:
            return self._scales

        raise DataProcessorError("SVD is not trained.")

    def _process(self, datum: np.array) -> np.array:
        """Project the IQ data onto the axis defined by an SVD and scale it.

        Args:
            datum: A 2D array of qubits, and an average complex IQ point as [real, imaginary].

        Returns:
            A 1D array. Each entry is the real part of the averaged IQ data of a qubit.

        Raises:
            DataProcessorError: If the SVD has not been previously trained on data.
        """

        if not self._main_axes:
            raise DataProcessorError("SVD must be trained on data before it can be used.")

        n_qubits = datum.shape[0]
        processed_data = []

        # process each averaged IQ point with its own axis.
        for idx in range(n_qubits):

            centered = np.array([datum[idx][iq] - self._means[idx][iq] for iq in [0, 1]])

            processed_data.append((self._main_axes[idx] @ centered) / self._scales[idx])

        return np.array(processed_data)

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

        n_qubits = self._format_data(data[0]).shape[0]

        self._main_axes = []
        self._scales = []
        self._means = []

        for qubit_idx in range(n_qubits):
            datums = np.vstack([self._format_data(datum)[qubit_idx] for datum in data]).T

            # Calculate the mean of the data to recenter it in the IQ plane.
            mean_i = np.average(datums[0, :])
            mean_q = np.average(datums[1, :])

            self._means.append((mean_i, mean_q))

            datums[0, :] = datums[0, :] - mean_i
            datums[1, :] = datums[1, :] - mean_q

            u, s, vh = np.linalg.svd(datums)

            self._main_axes.append(u[:, 0])
            self._scales.append(s[0])


class ToReal(IQPart):
    """IQ data post-processing. Isolate the real part of the IQ data."""

    def _process(self, datum: np.array) -> np.array:
        """Take the real part of the IQ data.

        Args:
            datum: A 3D array of shots, qubits, and a complex IQ point as [real, imaginary].

        Returns:
            A 2D array of shots, qubits. Each entry is the real part of the given IQ data.
        """
        if self.scale is None:
            return datum[:, :, 0]

        return datum[:, :, 0] * self.scale


class ToImag(IQPart):
    """IQ data post-processing. Isolate the imaginary part of the IQ data."""

    def _process(self, datum: np.array) -> np.array:
        """Take the imaginary part of the IQ data.

        Args:
            datum: A 3D array of shots, qubits, and a complex IQ point as [real, imaginary].

        Returns:
            A 2D array of shots, qubits. Each entry is the imaginary part of the given IQ data.
        """
        if self.scale is None:
            return datum[:, :, 1]

        return datum[:, :, 1] * self.scale


class Probability(DataAction):
    """Count data post processing. This returns the probabilities of the outcome string
    used to initialize an instance of Probability."""

    def __init__(self, outcome: str, validate: bool = True):
        """Initialize a counts to probability data conversion.

        Args:
            outcome: The bitstring for which to compute the probability.
            validate: If set to False the DataAction will not validate its input.
        """
        self._outcome = outcome
        super().__init__(validate)

    def _format_data(self, datum: dict) -> dict:
        """
        Checks that the given data has a counts format.

        Args:
            datum: An instance of data the should be a dict with bit strings as keys
                and counts as values.
            validate: If True the DataAction checks that the format of the datum is valid.

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

        return datum

    def _process(self, datum: Dict[str, Any]) -> Tuple[float, float]:
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
