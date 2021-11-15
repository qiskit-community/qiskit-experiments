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

import numpy as np


class DataAction(metaclass=ABCMeta):
    """Abstract action done on measured data to process it.

    Each subclass of DataAction must define the way it formats, validates and processes data.
    """

    def __init__(self, validate: bool = True):
        """Create new node.

        Args:
            validate: If set to False the DataAction will not validate its input.
        """
        self._validate = validate

    @abstractmethod
    def _process(self, data: np.ndarray) -> np.ndarray:
        """Applies the data processing step to the data.

        Args:
            data: A data array to process. This is a single numpy array containing
                all circuit results input to the data processor.
                If the elements are ufloat objects consisting of a nominal value and
                a standard error, then the error propagation is automatically computed.

        Returns:
            The processed data.
        """

    def _format_data(self, data: np.ndarray) -> np.ndarray:
        """Format and validate the input.

        Check that the given data has the correct structure. This method may
        additionally change the data type, e.g. converting a list to a numpy array.

        Args:
            data: A data array to format. This is a single numpy array containing
                all circuit results input to the data processor.

        Returns:
            The data that has been validated and formatted.
        """
        return data

    def __call__(self, data: np.ndarray) -> np.ndarray:
        """Call the data action of this node on the data.

        Args:
            data: A numpy array with arbitrary dtype. If the elements are ufloat objects
                consisting of a nominal value and a standard error, then the error propagation
                is done automatically.

        Returns:
            The processed data.
        """
        return self._process(self._format_data(data))

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
    def train(self, data: np.ndarray):
        """Train a DataAction.

        Certain data processing nodes, such as a SVD, require data to first train.

        Args:
            data: A data array for training. This is a single numpy array containing
                all circuit results input to the data processor :meth:`~qiskit_experiments.\
                data_processing.data_processor.DataProcessor#train` method.
        """
