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

"""Class that ties together data processing steps."""

import numpy as np
from typing import Union

from .nodes import SystemKernel, SystemDiscriminator
from .base import NodeType, DataAction
from qiskit_experiments.calibration.metadata import CalibrationMetadata
from qiskit_experiments.calibration.exceptions import CalibrationError
from qiskit.qobj.utils import MeasLevel, MeasReturnType
from qiskit.result import Result


class DataProcessor:
    """
    Defines the actions done on the measured data to bring it in a form usable
    by the calibration analysis classes.
    """

    def __init__(self, average: bool = True):
        """Create an empty chain of data ProcessingSteps.

        Args:
            average: Set `True` to average outcomes.
        """
        self._average = average
        self._root_node = None
        self._shots = None

    @property
    def shots(self):
        """Return the number of shots."""
        return self._shots

    @shots.setter
    def shots(self, val: int):
        """Set new shot value."""
        self._shots = val

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

    def meas_return(self) -> MeasReturnType:
        """
        Returns:
            MeasReturnType: The appropriate measurement format to use this analysis chain.
        """
        if DataProcessor.check_discriminator(self._root_node):
            # if discriminator is defined, return type should be single.
            # quantum state cannot be discriminated with averaged IQ coordinate.
            return MeasReturnType.SINGLE

        return MeasReturnType.AVERAGE if self._average else MeasReturnType.SINGLE

    def meas_level(self) -> MeasLevel:
        """
        TODO What about starting from MeasLevel 1?

        Returns:
            measurement level: MeasLevel.CLASSIFIED is returned if the end data is discriminated,
                MeasLevel.KERNELED is returned if a kernel is defined but no discriminator, and
                MeasLevel.RAW is returned is no kernel is defined.
        """
        kernel = DataProcessor.check_kernel(self._root_node)
        if kernel and isinstance(kernel, SystemKernel):
            discriminator = DataProcessor.check_discriminator(self._root_node)
            if discriminator and isinstance(discriminator, SystemDiscriminator):

                # classified level if both system kernel and discriminator are defined
                return MeasLevel.CLASSIFIED

            # kerneled level if only system kernel is defined
            return MeasLevel.KERNELED

        # otherwise raw level is requested
        return MeasLevel.RAW

    def format_data(self, result: Result, metadata: CalibrationMetadata, index: int):
        """
        Format Qiskit result data.

        This method sequentially calls stored child data processing nodes
        with its `format_data` methods. Once all child nodes have called,
        input data is converted into expected data format.

        Args:
            result: Qiskit Result object.
            metadata: Metadata for the target circuit.
            index: Index of target circuit in the experiment.

        Raises:
            CalibrationError: if
        """
        if not self._root_node:
            return result

        # extract outcome with marginalize. note that the pulse experiment data
        # is not marginalized on the backend.

        if self.meas_level() == MeasLevel.CLASSIFIED:
            data = result.get_counts(experiment=index)

        elif self.meas_level() == MeasLevel.KERNELED:
            data = np.asarray(result.get_memory(index), dtype=complex)

        elif self.meas_level() == MeasLevel.RAW:
            raise CalibrationError('Raw data analysis is not supported.')

        else:
            raise CalibrationError('Invalid measurement level is specified.')

        if not self._root_node:
            return data

        return self._root_node.format_data(data, metadata=metadata, shots=self.shots)

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
