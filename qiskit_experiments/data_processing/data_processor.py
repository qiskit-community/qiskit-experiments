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

from typing import Any, Dict, Union

from qiskit.qobj.utils import MeasLevel
from qiskit_experiments.data_processing.nodes import Kernel, Discriminator
from qiskit_experiments.data_processing.base import NodeType, DataAction


class DataProcessor:
    """
    Defines the actions done on the measured data to bring it in a form usable
    by the calibration analysis classes.
    """

    def __init__(self):
        """Create an empty chain of data ProcessingSteps."""
        self._root_node = None

    def append(self, node: DataAction):
        """
        Append new data action node to this data processor.

        Args:
            node: A DataAction that will process the data.
        """
        if self._root_node:
            self._root_node.append(node)
        else:
            self._root_node = node

    def meas_level(self) -> MeasLevel:
        """
        Returns:
            measurement level: MeasLevel.CLASSIFIED is returned if the end data is discriminated,
                MeasLevel.KERNELED is returned if a kernel is defined but no discriminator, and
                MeasLevel.RAW is returned is no kernel is defined.
        """
        kernel = DataProcessor.check_kernel(self._root_node)
        if kernel and isinstance(kernel, Kernel):
            discriminator = DataProcessor.check_discriminator(self._root_node)
            if discriminator and isinstance(discriminator, Discriminator):

                # classified level if both system kernel and discriminator are defined
                return MeasLevel.CLASSIFIED

            # kerneled level if only system kernel is defined
            return MeasLevel.KERNELED

        # otherwise raw level is requested
        return MeasLevel.RAW

    def output_key(self) -> str:
        """Return the key to look for in the data output by the processor."""
        if self._root_node:
            node = self._root_node
            while node.child:
                node = node.child

            if node.node_type in [NodeType.KERNEL, NodeType.IQDATA]:
                return 'memory'
            if node.node_type == NodeType.DISCRIMINATOR:
                return 'counts'
            if node.node_type == NodeType.POPULATION:
                return 'populations'

        return 'counts'

    def format_data(self, data: Dict[str, Any]):
        """
        Format Qiskit result data.

        This method sequentially calls stored child data processing nodes
        with its `format_data` methods. Once all child nodes have called,
        input data is converted into expected data format.

        Args:
            data: The data, typically from an ExperimentData instance, that needs to
                be processed. This dict also contains the metadata of each experiment.
        """
        if self._root_node:
            self._root_node.format_data(data)

    @classmethod
    def check_kernel(cls, node: DataAction) -> Union[None, DataAction]:
        """Return the stored kernel in the workflow."""
        if not node:
            return None

        if node.node_type == NodeType.KERNEL:
            return node
        else:
            if not node.child:
                return None
            return cls.check_kernel(node.child)

    @classmethod
    def check_discriminator(cls, node: DataAction):
        """Return stored discriminator in the workflow."""
        if not node:
            return None

        if node.node_type == NodeType.DISCRIMINATOR:
            return node
        else:
            if not node.child:
                return None
            return cls.check_discriminator(node.child)
