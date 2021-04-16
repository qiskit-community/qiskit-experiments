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

"""Actions done on the data to bring it in a usable form."""

from typing import Any, Dict, List, Set, Tuple

from qiskit_experiments.data_processing.data_action import DataAction
from qiskit_experiments.data_processing.exceptions import DataProcessorError


class DataProcessor:
    """
    A DataProcessor defines a sequence of operations to perform on experimental data.
    The DataProcessor.format_data() method applies this sequence on its input argument.
    A DataProcessor is created with a list of DataAction objects. Each DataAction
    specifies a set of node_inputs that it accepts and a __node_output__ that it
    provides. The __node_output__ of each DataAction must be contained in the node_inputs
    of the following DataAction in the DataProcessor's list. DataProcessor.format_data()
    usually takes in one entry from the data property of an ExperimentData object
    (i.e. a dict containing metadata and memory keys and possibly counts, like the
    Result.data property) and produces a new dict containing the formatted data. The data
    passed to DataProcessor.format_data() is passed to the first DataAction and the
    output is passed on in turn to each DataAction. DataProcessor.format_data() returns
    the data produced by the last DataAction.
    """

    def __init__(self, input_key: str, data_actions: List[DataAction] = None):
        """Create a chain of data processing actions.

        Args:
            input_key: The initial key in the datum Dict[str, Any] under which the data processor
                will find the data to process.
            data_actions: A list of data processing actions to construct this data processor with.
                If None is given an empty DataProcessor will be created.
        """
        self._input_key = input_key
        self._nodes = []

        if data_actions:
            for node in data_actions:
                self.append(node)


    def append(self, node: DataAction):
        """
        Append new data action node to this data processor.

        Args:
            node: A DataAction that will process the data.
        """
        self._nodes.append(node)

    def __call__(self, datum: Dict[str, Any]) -> Tuple[Any, Any]:
        """
        Call self on the given datum. This method sequentially calls the stored data actions
        on the datum.

        Args:
            datum: A single item of data, typically from an ExperimentData instance, that needs
                to be processed. This dict also contains the metadata of each experiment.

        Returns:
            processed data: The data processed by the data processor.

        Raises:
            DataProcessorError: if no nodes are present.
        """
        if len(self._nodes) == 0:
            raise DataProcessorError("Cannot call an empty data processor.")

        datum_ = datum[self._input_key]

        for node in self._nodes:
            datum_ = node(datum_)

        return datum_

    def call_with_history(
        self, datum: Dict[str, Any], history_nodes: Set = None
    ) -> Tuple[Dict[str, Any], List]:
        """
        Call self on the given datum. This method sequentially calls the stored data actions
        on the datum and also saves the history of the processed data.

        Args:
            datum: A single item of data, typically from an ExperimentData instance, that
                needs to be processed.
            history_nodes: The nodes, specified by index in the data processing chain, to
                include in the history.

        Returns:
            processed data: The datum processed by the data processor.
            history: The datum processed at each node of the data processor.

        Raises:
            DataProcessorError: if no nodes are present.
        """
        if len(self._nodes) == 0:
            raise DataProcessorError("Cannot call an empty data processor.")

        datum_ = datum[self._input_key]

        history = []
        for index, node in enumerate(self._nodes):
            datum_ = node(datum_)

            if history_nodes is None or (history_nodes and index in history_nodes):
                history.append((node.__class__.__name__, datum_, index))

        return datum_, history
