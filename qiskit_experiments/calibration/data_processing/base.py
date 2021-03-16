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
from enum import Enum
from typing import Any

from qiskit_experiments.calibration.metadata import CalibrationMetadata
from qiskit_experiments.calibration.exceptions import CalibrationError


class DataAction(metaclass=ABCMeta):
    """
    Abstract action which is a single action done on measured data to process it.
    Each subclass of DataAction must define the type of data that it accepts as input
    using decorators.
    """

    node_type = None
    prev_node = ()

    def __init__(self):
        """Create new data analysis routine."""
        self._child = None

    @property
    def child(self) -> 'DataAction':
        """Return the child of this data processing step."""
        return self._child

    def append(self, component: 'DataAction'):
        """Add new data processing routine.

        Args:
            component: New data processing routine.

        Raises:
            CalibrationError: If the previous node is None (i.e. a root node)
        """
        if not component.prev_node:
            raise CalibrationError(f'Analysis routine {component.__class__.__name__} is a root'
                                   f'node. This routine cannot be appended to another node.')

        if self._child is None:
            if isinstance(self, component.prev_node):
                self._child = component
            else:
                raise CalibrationError(f'Analysis routine {component.__class__.__name__} '
                                       f'cannot be appended after {self.__class__.__name__}')
        else:
            self._child.append(component)

    @abstractmethod
    def process(self, data: Any, **kwargs):
        """
        TODO

        Applies the data processing step to the data.

        Args:
            data:
            kwargs:
        """

    def format_data(self, data: Any, metadata: CalibrationMetadata, shots: int) -> Any:
        """
        TODO

        Args:
            data:
            metadata:
            shots:

        Returns:
            processed_data:
        """
        processed_data = self.process(data, metadata=metadata, shots=shots)

        if self._child:
            return self._child.format_data(processed_data, metadata, shots)
        else:
            return processed_data


class NodeType(Enum):
    """Type of node that can be supported by the analysis steps."""
    KERNEL = 1
    DISCRIMINATOR = 2
    IQDATA = 3
    COUNTS = 4


def kernel(cls: DataAction):
    """A decorator to give kernel attribute to node."""
    cls.node_type = NodeType.KERNEL
    return cls


def discriminator(cls: DataAction):
    """A decorator to give discriminator attribute to node."""
    cls.node_type = NodeType.DISCRIMINATOR
    return cls


def iq_data(cls: DataAction):
    """A decorator to give iqdata attribute to node."""
    cls.node_type = NodeType.IQDATA
    return cls


def counts(cls: DataAction):
    """A decorator to give counts attribute to node."""
    cls.node_type = NodeType.COUNTS
    return cls


def prev_node(*nodes: DataAction):
    """A decorator to specify the available previous nodes."""

    try:
        nodes = list(nodes)
    except TypeError:
        nodes = [nodes]

    def add_nodes(cls: DataAction):
        cls.prev_node = tuple(nodes)
        return cls

    return add_nodes
