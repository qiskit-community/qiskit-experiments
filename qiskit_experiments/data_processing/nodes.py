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
from typing import Any, Dict, Optional, Tuple
import numpy as np

from qiskit_experiments.data_processing.data_action import DataAction
from qiskit_experiments.data_processing.exceptions import DataProcessorError


class IQPart(DataAction):
    """Abstract class for IQ data post-processing."""

    def __init__(self, scale: Optional[float] = None):
        """
        Args:
            scale: float with which to multiply the IQ data.
        """
        self.scale = scale
        super().__init__()

    @abstractmethod
    def _process_iq(self, datum: np.array) -> np.array:
        """Defines how the IQ point will be processed.

        Args:
            datum: a 3D array of shots, qubits, and a complex IQ point as [real, imaginary].

        Returns:
            Processed IQ point.
        """

    def _check_data_format(self, datum: Any) -> Any:
        """Check that the IQ data has the correct format.

        Args:
            datum: A single item of data which corresponds to single-shot IQ data. It should
                have dimension three: shots, qubits, iq-point as [real, imaginary].

        Returns:
            datum: as a numpy array.

        Raises:
            DataProcessorError: if the datum does not have the correct format.
        """
        if not isinstance(datum, (list, np.ndarray)):
            raise DataProcessorError(
                f"The IQ data given to {self.__class__.__name__} " f"must be a list or ndarray."
            )

        if isinstance(datum, list):
            datum = np.asarray(datum)

        if len(datum.shape) != 3:
            raise DataProcessorError(
                f"Single-shot data given {self.__class__.__name__}"
                f"must be a 3D array. Instead, a {len(datum.shape)}D "
                f"array was given."
            )

        return datum

    def _process(self, datum: np.array) -> np.array:
        """Wraps _process_iq.

        Args:
            datum: A single item of data.

        Returns:
            processed data
        """
        return self._process_iq(datum)


class ToReal(IQPart):
    """IQ data post-processing. Isolate the real part of the IQ data."""

    def _process_iq(self, datum: np.array) -> np.array:
        """Take the real part of the IQ data.

        Args:
            datum: a 3D array of shots, qubits, and a complex IQ point as [real, imaginary].

        Returns:
            A 2D array of shots, qubits. Each entry is the real part of the given IQ data.
        """
        if self.scale is None:
            return datum[:, :, 0]

        return datum[:, :, 0] * self.scale


class ToImag(IQPart):
    """IQ data post-processing. Isolate the imaginary part of the IQ data."""

    def _process_iq(self, datum: np.array) -> np.array:
        """Take the imaginary part of the IQ data.

        Args:
            datum: a 3D array of shots, qubits, and a complex IQ point as [real, imaginary].

        Returns:
            A 2D array of shots, qubits. Each entry is the imaginary part of the given IQ data.
        """
        if self.scale is None:
            return datum[:, :, 1]

        return datum[:, :, 1] * self.scale


class Probability(DataAction):
    """Count data post processing. This returns qubit 1 state probabilities."""

    def __init__(self, outcome: str):
        """Initialize a counts to probability data conversion.

        Args:
            outcome: The bitstring for which to compute the probability.
        """
        super().__init__()
        self._outcome = outcome

    def _check_data_format(self, datum: dict) -> dict:
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
        if not isinstance(datum, dict):
            raise DataProcessorError(
                f"Given counts datum {datum} to "
                f"{self.__class__.__name__} is not a valid count format."
            )

        for bit_str, count in datum.items():
            if not isinstance(bit_str, str):
                raise DataProcessorError(
                    f"Key {bit_str} is not a valid count key for " f"{self.__class__.__name__}."
                )

            if not isinstance(count, (int, float)):
                raise DataProcessorError(
                    f"Count {bit_str} is not a valid count key for" f"{self.__class__.__name__}."
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
        p_var = shots * p_mean * (1 - p_mean)

        return p_mean, p_var
