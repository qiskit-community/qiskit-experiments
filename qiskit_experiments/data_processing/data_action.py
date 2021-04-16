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
from typing import Any


class DataAction(metaclass=ABCMeta):
    """
    Abstract action which is a single action done on measured data to process it.
    Each subclass of DataAction must define the type of data that it accepts as input
    using decorators.
    """

    @abstractmethod
    def _process(self, datum: Any) -> Any:
        """
        Applies the data processing step to the data.

        Args:
            datum: A single item of data data which will be processed.

        Returns:
            processed data: The data that has been processed.
        """

    @abstractmethod
    def _check_data_format(self, datum: Any) -> Any:
        """
        Check that the given data has the correct structure. This method may
        additionally change the data type, e.g. converting a list to a numpy array.

        Args:
            datum: The data instance to check.

        Returns:
            datum: The data that was check.

        Raises:
            DataProcessorError: if the data does not have the proper format.
        """

    def __call__(self, data: Any) -> Any:
        """
        Call the data action of this node on the data.

        Args:
            data: The data to process. The action nodes in the data
                processor will raise errors if the data does not contain the
                appropriate data.

        Returns:
            processed data: The output data of the node contained in a dict.
        """
        return self._process(self._check_data_format(data))

    def __repr__(self):
        """String representation of the node."""
        return f"{self.__class__.__name__}"
