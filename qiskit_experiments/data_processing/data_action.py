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

"""Defines the steps that can be used to analyse data."""

from abc import ABCMeta, abstractmethod
from typing import Any, List, Optional, Tuple


class DataAction(metaclass=ABCMeta):
    """
    Abstract action done on measured data to process it. Each subclass of DataAction must
    define the way it formats, validates and processes data.
    """

    def __init__(self, validate: bool = True):
        """
        Args:
            validate: If set to False the DataAction will not validate its input.
        """
        self._validate = validate

    @abstractmethod
    def _process(self, datum: Any, error: Optional[Any] = None) -> Tuple[Any, Any]:
        """
        Applies the data processing step to the datum.

        Args:
            datum: A single item of data which will be processed.
            error: An optional error estimation on the datum that can be further propagated.

        Returns:
            processed data: The data that has been processed along with the propagated error.
        """

    @abstractmethod
    def _format_data(self, datum: Any, error: Optional[Any] = None) -> Tuple[Any, Any]:
        """Format and validate the input.

        Check that the given data and error has the correct structure. This method may
        additionally change the data type, e.g. converting a list to a numpy array.

        Args:
            datum: The data instance to check and format.
            error: An optional error estimation on the datum to check and format.

        Returns:
            datum, error: The formatted datum and its optional error.

        Raises:
            DataProcessorError: If either the data or the error do not have the proper format.
        """

    def __call__(self, data: Any, error: Optional[Any] = None) -> Tuple[Any, Any]:
        """Call the data action of this node on the data and propagate the error.

        Args:
            data: The data to process. The action nodes in the data processor will
                raise errors if the data does not have the appropriate format.
            error: An optional error estimation on the datum that can be further processed.

        Returns:
            processed data: The data processed by self as a tuple of processed datum and
                optionally the propagated error estimate.
        """
        return self._process(*self._format_data(data, error))

    def __repr__(self):
        """String representation of the node."""
        return f"{self.__class__.__name__}(validate={self._validate})"


class TrainableDataAction(DataAction):
    """A base class for data actions that need training."""

    @property
    @abstractmethod
    def is_trained(self) -> bool:
        """Return False if the DataAction needs to be trained.

        Subclasses must implement this property to communicate if they have been trained.

        Return:
            True if the data action has been trained.
        """

    @abstractmethod
    def train(self, data: List[Any]):
        """Train a DataAction.

        Certain data processing nodes, such as a SVD, require data to first train.

        Args:
            data: A list of datum. Each datum is a point used to train the node.
        """
