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
from typing import Any, Dict, List

from qiskit_experiments.data_processing.exceptions import DataProcessorError


class DataAction(metaclass=ABCMeta):
    """
    Abstract action which is a single action done on measured data to process it.
    Each subclass of DataAction must define the type of data that it accepts as input
    using decorators.
    """

    # Key under which the node will output the data.
    __node_output__ = None

    def __init__(self):
        """Create new data analysis routine."""
        self._child = None
        self._accepted_inputs = []

    @property
    def node_inputs(self) -> List[str]:
        """Returns a list of input data that the node can process."""
        return self._accepted_inputs

    def add_accepted_input(self, data_key: str):
        """
        Allows users to add an accepted input data format to this DataAction.

        Args:
            data_key: The key that the data action will require in the input data dict.
        """
        self._accepted_inputs.append(data_key)

    @abstractmethod
    def _process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Applies the data processing step to the data.

        Args:
            data: the data to which the data processing step will be applied.

        Returns:
            processed data: The data that has been processed.
        """

    def _check_required(self, data: Dict[str, Any]):
        """Checks that the given data contains the right key.

        Args:
            data: The data to check for the correct keys.

        Raises:
            DataProcessorError: if the key is not found.
        """
        for key in data.keys():
            if key in self.node_inputs:
                return

        raise DataProcessorError(f"None of {self.node_inputs} are in the given data.")

    def format_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply the data action of this node and call the child node's format_data method.

        Args:
            data: A dict containing the data. The action nodes in the data
                processor will raise errors if the data does not contain the
                appropriate data.

        Returns:
            processed data: The output data of the node contained in a dict.
        """
        self.check_required(data)
        processed_data = self.process(data)
        processed_data["metadata"] = data.get("metadata", {})
        return processed_data

    def __repr__(self):
        """String representation of the node."""
        return (
            f"{self.__class__.__name__}(inputs: {self.node_inputs}, "
            f"outputs: {self.__node_output__})"
        )
