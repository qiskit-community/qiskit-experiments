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

from typing import Any, Dict, List, Tuple, Union, Callable, Optional

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

    def __call__(
        self,
        data: Union[Dict, List[Dict]],
        up_to_index: Optional[int] = None,
        callback: Optional[Callable[[int, str, Any, Any], None]] = None,
    ) -> Tuple[Any, Any]:
        """
        Call self on the given datum. This method sequentially calls the stored data actions
        on the datum.

        Args:
            data: The data, typically from ExperimentData.data(...), that needs to be processed.
                This dict or list of dicts also contains the metadata of each experiment.
            up_to_index: The data processor will use each node in the processing chain
                up to the node indexed by call_up_to_node. If this variable is not specified
                then all nodes in the data processing chain will be called.
            callback: Arbitrary python callable that is called after each node execution.
                Processor passes (index of node, name of node, nominal values, standard errors)
                to the callback. This can be used to log the history of intermediate data.
                See :class:`qiskit_experiments.data_processing.data_processor.DataLogandger`
                for the preset logger.

        Returns:
            A tuple of (nominal values, standard errors) processed by the processor.
        """
        if up_to_index is None:
            up_to_index = len(self._nodes)

        datum_, error_ = self._data_extraction(data), None
        for index, node in enumerate(self._nodes[:up_to_index]):
            datum_, error_ = node(datum_, error_)

            if callback:
                callback(index, node.__class__.__name__, datum_, error_)

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
                    node.train(self.__call__(data, up_to_index=index)[0])

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


class DataLogger:
    """Data processor logger.

    This class implements the :meth:``__call__`` method so that it can be used as a callback.
    Once this instance is called with data in the data processor,
    this records intermediate data generated by a specific processor node.
    That can be accessed via :meth:`data` method after the processor is executed.
    """

    def __init__(self, history_nodes: Optional[List[int]] = None):
        """Create new data logger.

        Args:
            history_nodes: List of node index to record data.
        """
        self._history = list()
        self._history_nodes = history_nodes

    def __call__(self, index: int, name: str, nominals: Any, stdevs: Any):
        """Record data. This is invoked by the data processor.

        Args:
            index: Position of processing node in the entire processing chain.
            name: Name of node.
            nominals: Nominal values.
            stdevs: Standard errors.
        """
        if self._history_nodes is None or index in self._history_nodes:
            self._history.append((name, nominals, stdevs, index))

    def clear(self):
        """Clear previous data."""
        self._history.clear()

    def data(
        self, index: Optional[Union[str, int]] = None
    ) -> Union[Tuple[Any, Any], List[Tuple[Any, Any]]]:
        """Get intermediate data.

        Args:
            index: Index of target data, either node index or node name.
                Return all data if not specified.

        Returns:
            A tuple of (nominal values, standard errors) or list of it.

        Raises:
            DataProcessorError: When index is not found.
        """
        if index is None:
            return self._history

        colum = 0 if isinstance(index, str) else 3
        for data in self._history:
            if data[colum] == index:
                return data[1], data[2]

        raise DataProcessorError(f"Index {index} is not found.")
