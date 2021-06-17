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

from typing import Any, Dict, List, Set, Tuple, Union

from qiskit_experiments.data_processing.data_action import DataAction, TrainableDataAction
from qiskit_experiments.data_processing.exceptions import DataProcessorError


class DataProcessor:
    """
    A DataProcessor defines a sequence of operations to perform on experimental data.
    Calling an instance of DataProcessor applies this sequence on the input argument.
    A DataProcessor is created with a list of DataAction instances. Each DataAction
    applies its _process method on the data and returns the processed data. The nodes
    in the DataProcessor may also perform data validation and some minor formatting.
    The output of one data action serves as input for the next data action.
    DataProcessor.__call__(datum) usually takes in an entry from the data property of
    an ExperimentData object (i.e. a dict containing metadata and memory keys and
    possibly counts, like the Result.data property) and produces the formatted data.
    DataProcessor.__call__(datum) extracts the data from the given datum under
    DataProcessor._input_key (which is specified at initialization) of the given datum.
    """

    def __init__(
        self,
        input_key: str,
        data_actions: List[DataAction] = None,
    ):
        """Create a chain of data processing actions.

        Args:
            input_key: The initial key in the datum Dict[str, Any] under which the data processor
                will find the data to process.
            data_actions: A list of data processing actions to construct this data processor with.
                If None is given an empty DataProcessor will be created.
            to_array: Boolean indicating if the input data will be converted to a numpy array.
        """
        self._input_key = input_key
        self._nodes = data_actions if data_actions else []

    def append(self, node: DataAction):
        """
        Append new data action node to this data processor.

        Args:
            node: A DataAction that will process the data.
        """
        self._nodes.append(node)

    @property
    def is_trained(self) -> bool:
        """Return True if all nodes of the data processor have been trained."""
        for node in self._nodes:
            if isinstance(node, TrainableDataAction):
                if not node.is_trained:
                    return False

        return True

    def __call__(self, data: Union[Dict, List[Dict]], **options) -> Tuple[Any, Any]:
        """
        Call self on the given datum. This method sequentially calls the stored data actions
        on the datum.

        Args:
            data: The data, typically from ExperimentData.data(...), that needs to be processed.
            This dict or list of dicts also contains the metadata of each experiment.
            options: Run-time options given as keyword arguments that will be passed to the nodes.

        Returns:
            processed data: The data processed by the data processor.
        """
        return self._call_internal(data, **options)

    def call_with_history(
        self, data: Union[Dict, List[Dict]], history_nodes: Set = None
    ) -> Tuple[Any, Any, List]:
        """
        Call self on the given datum. This method sequentially calls the stored data actions
        on the datum and also returns the history of the processed data.

        Args:
            data: The data, typically from ExperimentData.data(...), that needs to be processed.
            This dict or list of dicts also contains the metadata of each experiment.
            history_nodes: The nodes, specified by index in the data processing chain, to
                include in the history. If None is given then all nodes will be included
                in the history.

        Returns:
            processed data: The datum processed by the data processor.
            history: The datum processed at each node of the data processor.
        """
        return self._call_internal(data, True, history_nodes)

    def _call_internal(
        self,
        data: Union[Dict, List[Dict]],
        with_history: bool = False,
        history_nodes: Set = None,
        call_up_to_node: int = None,
    ) -> Union[Tuple[Any, Any], Tuple[Any, Any, List]]:
        """Process the data with or without storing the history of the computation.

        Args:
            data: The data, typically from ExperimentData.data(...), that needs to be processed.
            This dict or list of dicts also contains the metadata of each experiment.
            with_history: if True the history is returned otherwise it is not.
            history_nodes: The nodes, specified by index in the data processing chain, to
                include in the history. If None is given then all nodes will be included
                in the history.
            call_up_to_node: The data processor will use each node in the processing chain
                up to the node indexed by call_up_to_node. If this variable is not specified
                then all nodes in the data processing chain will be called.

        Returns:
            datum_ and history if with_history is True or datum_ if with_history is False.
        """
        if call_up_to_node is None:
            call_up_to_node = len(self._nodes)

        datum_, error_ = self._data_extraction(data), None

        history = []
        for index, node in enumerate(self._nodes):

            if index < call_up_to_node:
                datum_, error_ = node(datum_, error_)

                if with_history and (
                    history_nodes is None or (history_nodes and index in history_nodes)
                ):
                    history.append((node.__class__.__name__, datum_, error_, index))

        if with_history:
            return datum_, error_, history
        else:
            return datum_, error_

    def train(self, data: List[Dict[str, Any]]):
        """Train the nodes of the data processor.

        Args:
            data: The data to use to train the data processor.
        """

        for index, node in enumerate(self._nodes):
            if isinstance(node, TrainableDataAction):
                if not node.is_trained:
                    # Process the data up to the untrained node.
                    node.train(self._call_internal(data, call_up_to_node=index)[0])

    def _data_extraction(self, data: Union[Dict, List[Dict]]) -> List:
        """Extracts the data on which to run the nodes.

        If the datum is a list of dicts then the data under self._input_key is extracted
        from each dict and appended to a list which therefore contains all the data. If the
        data processor has to_array set to True then the list will be converted to a numpy
        array.

        Args:
            data: A list of such dicts where the data is contained under the key self._input_key.

        Returns:
            The data formatted in such a way that it is ready to be processed by the nodes.

        Raises:
            DataProcessorError:
                - If the input datum is not a list or a dict.
                - If the data processor received a single datum but requires all the data to
                  process it properly.
                - If the input key of the data processor is not contained in the data.
        """
        if isinstance(data, dict):
            data = [data]

        try:
            data_ = [_datum[self._input_key] for _datum in iter(data)]
        except KeyError as error:
            raise DataProcessorError(
                f"The input key {self._input_key} was not found in the input datum."
            ) from error
        except TypeError as error:
            raise DataProcessorError(
                f"{self.__class__.__name__} only extracts data from "
                f"lists or dicts, received {type(data)}."
            ) from error

        return data_

    def __repr__(self):
        """String representation of data processors."""
        names = ", ".join(node.__class__.__name__ for node in self._nodes)

        return f"{self.__class__.__name__}(input_key={self._input_key}, nodes=[{names}])"
