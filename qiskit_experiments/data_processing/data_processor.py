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

"""Class that ties together actions on the data."""

from typing import Any, Dict, List, Tuple, Union

from qiskit_experiments.data_processing.base import DataAction
from qiskit_experiments.data_processing.exceptions import DataProcessorError


class DataProcessor:
    """
    Defines the actions done on the measured data to bring it in a form usable
    by the calibration analysis classes.
    """

    def __init__(self):
        """Create an empty chain of data processing actions."""
        self._nodes = []
        self._history = []

    @property
    def history(self) -> List[Tuple[str, Dict[str, Any]]]:
        """
        Returns:
            The history of the data processor. The ith tuple in the history corresponds to the
            output of the ith node. Each tuple corresponds to (node name, data dict).
        """
        return self._history

    def append(self, node: DataAction):
        """
        Append new data action node to this data processor.

        Args:
            node: A DataAction that will process the data.

        Raises:
            DataProcessorError: if the output of the last node does not match the input required
                by the node to be appended.
        """
        if len(self._nodes) == 0:
            self._nodes.append(node)
        else:
            if self._nodes[-1].node_output not in node.node_inputs:
                raise DataProcessorError(
                    f"Output of node {self._nodes[-1]} is not an acceptable " f"input to {node}."
                )

            self._nodes.append(node)

    def output_key(self) -> Union[str, None]:
        """Return the key to look for in the data output by the processor."""

        if len(self._nodes) > 0:
            return self._nodes[-1].node_output

        return None

    def format_data(self, data: Dict[str, Any], save_history: bool = False) -> Dict[str, Any]:
        """
        Format the given data.

        This method sequentially calls stored child data processing nodes
        with its `format_data` methods. Once all child nodes have called,
        input data is converted into expected data format.

        Args:
            data: The data, typically from an ExperimentData instance, that needs to
                be processed. This dict also contains the metadata of each experiment.
            save_history: If set to true a list of the formatted data at each step is returned.

        Returns:
            processed data: The last entry in the list is the end output of the data processor.
                If the history of the data actions is required the returned data is a list of
                length n+1 where n is the number of data actions in the data processor.
        """
        self._history = []

        for node in self._nodes:
            data = node.format_data(data)

            if save_history:
                self._history.append((node.__class__.__name__, dict(data)))

        return data
