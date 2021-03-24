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

"""Different data analysis steps."""

from typing import Any, Dict, Optional
import numpy as np

from qiskit_experiments.data_processing.exceptions import DataProcessorError
from qiskit_experiments.data_processing.base import (DataAction, iq_data, kernel,
                                                     discriminator, prev_node, population)


@kernel
@prev_node()
class Kernel(DataAction):
    """User provided kernel."""

    def __init__(self, kernel_, name: Optional[str] = None):
        """
        Args:
            kernel_: Kernel to kernel the data.
            name: Optional name for the node.
        """
        self.kernel = kernel_
        self.name = name
        super().__init__()

    def process(self, data: Dict[str, Any]):
        """
        Args:
            data: The data dictionary to process.

        Raises:
            DataProcessorError: if the data has no memory.
        """
        if 'memory' not in data:
            raise DataProcessorError(f'Data does not have memory. '
                                     f'Cannot apply {self.__class__.__name__}')

        data['memory'] = self.kernel.kernel(np.array(data['memory']))


@discriminator
@prev_node(Kernel)
class Discriminator(DataAction):
    """Backend system discriminator."""

    def __init__(self, discriminator_, name: Optional[str] = None):
        """
        Args:
            discriminator_: The discriminator used to transform the data to counts.
                For example, transform IQ data to counts.
            name: Optional name for the node.
        """
        self.discriminator = discriminator_
        self.name = name
        super().__init__()

    def process(self, data: Dict[str, Any]):
        """
        Discriminate the data to transform it into counts.

        Args:
            data: The data in a format that can be understood by the discriminator.

        Raises:
            DataProcessorError: if the data does not contain memory.
        """
        if 'memory' not in data:
            raise DataProcessorError(f'Data does not have memory. '
                                     f'Cannot apply {self.__class__.__name__}')

        data['counts'] = self.discriminator.discriminate(np.array(data['memory']))


@iq_data
@prev_node(Kernel)
class ToReal(DataAction):
    """IQ data post-processing. This returns real part of IQ data."""

    def __init__(self, scale: Optional[float] = 1.0, average: bool = False):
        """
        Args:
            scale: scale by which to multiply the real part of the data.
            average: if True the single-shots are averaged.
        """
        self.scale = scale
        self.average = average
        super().__init__()

    def process(self, data: Dict[str, Any]):
        """
        Modifies the data inplace by taking the real part of the memory and
        scaling it by the given factor.

        Args:
            data: The data dict. IQ data is stored under memory.

        Raises:
            DataProcessorError: if the data does not contain memory.
        """
        if 'memory' not in data:
            raise DataProcessorError(f'Data does not have memory. '
                                     f'Cannot apply {self.__class__.__name__}')

        # Single shot data
        if isinstance(data['memory'][0][0], list):
            new_mem = []
            for shot in data['memory']:
                new_mem.append([self.scale*_[0] for _ in shot])

            if self.average:
                new_mem = list(np.mean(np.array(new_mem), axis=0))

        # Averaged data
        else:
            new_mem = [self.scale*_[0] for _ in data['memory']]

        data['memory'] = new_mem

@iq_data
@prev_node(Kernel)
class ToImag(DataAction):
    """IQ data post-processing. This returns imaginary part of IQ data."""

    def __init__(self, scale: Optional[float] = 1.0, average: bool = False):
        """
        Args:
            scale: scale by which to multiply the imaginary part of the data.
        """
        self.scale = scale
        self.average = average
        super().__init__()

    def process(self, data: Dict[str, Any]):
        """
        Scales the imaginary part of IQ data.

        Args:
            data: The data dict. IQ data is stored under memory.

        Raises:
            DataProcessorError: if the data does not contain memory.
        """
        if 'memory' not in data:
            raise DataProcessorError(f'Data does not have memory. '
                                     f'Cannot apply {self.__class__.__name__}')

        # Single shot data
        if isinstance(data['memory'][0][0], list):
            new_mem = []
            for shot in data['memory']:
                new_mem.append([self.scale*_[1] for _ in shot])

            if self.average:
                new_mem = list(np.mean(np.array(new_mem), axis=0))

        # Averaged data
        else:
            new_mem = [self.scale*_[0] for _ in data['memory']]

        data['memory'] = new_mem


@population
@prev_node(Discriminator)
class Population(DataAction):
    """Count data post processing. This returns population."""

    def process(self, data: Dict[str, Any]):
        """
        Args:
            data: The data dictionary. This will modify the dict in place,
                taking the data under counts and adding the corresponding
                populations.

        Raises:
            DataProcessorError: if counts are not in the given data.
        """
        if 'counts' not in data:
            raise DataProcessorError(f'Data does not have counts. '
                                     f'Cannot apply {self.__class__.__name__}')

        counts = data.get('counts')

        populations = np.zeros(len(list(counts.keys())[0]))

        shots = 0
        for bit_str, count in counts.items():
            shots += 1
            for ind, bit in enumerate(bit_str):
                if bit == '1':
                    populations[ind] += count

        data['populations'] = populations / shots
