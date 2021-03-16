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

from typing import Optional, Any, Union

import numpy as np
from . import base
from . import DataAction
from qiskit.result.counts import Counts


@base.kernel
@base.prev_node()
class SystemKernel(DataAction):
    """Backend system kernel."""

    def __init__(self, name: Optional[str] = None):
        self.name = name
        super().__init__()

    def process(self, data: Any, **kwargs) -> Union[float, np.ndarray]:
        """
        Args:
            data:

        Returns:
            data: The data after applying the integration kernel.
        """
        raise NotImplementedError


@base.discriminator
@base.prev_node(SystemKernel)
class SystemDiscriminator(DataAction):
    """Backend system discriminator."""

    def __init__(self, discriminator, name: Optional[str] = None):
        """
        Args:
            discriminator: The discriminator used to transform the data to counts.
                For example, transform IQ data to counts.
        """
        self.discriminator = discriminator
        self.name = name
        super().__init__()

    def process(self, data: Union[float, np.ndarray], **kwargs) -> Counts:
        """
        Discriminate the data to transform it into counts.

        Args:
            data: The data in a format that can be understood by the discriminator.
        """
        return self.discriminator.discriminate(data)


@base.iq_data
@base.prev_node(SystemKernel)
class ToReal(DataAction):
    """IQ data post-processing. This returns real part of IQ data."""

    def __init__(self, scale: Optional[float] = 1.0):
        """
        Args:
            scale: scale by which to multiply the real part of the data.
        """
        self.scale = scale
        super().__init__()

    def process(self, data: Union[float, np.ndarray], **kwargs):
        """
        Scales the real part of IQ data.

        Args:
            data: IQ Data.

        Returns:
             data: The scaled imaginary part of the data.
        """
        return self.scale * data.real


@base.iq_data
@base.prev_node(SystemKernel)
class ToImag(DataAction):
    """IQ data post-processing. This returns imaginary part of IQ data."""

    def __init__(self, scale: Optional[float] = 1.0):
        """
        Args:
            scale: scale by which to multiply the imaginary part of the data.
        """
        self.scale = scale
        super().__init__()

    def process(self, data: Union[float, np.ndarray], **kwargs) -> Union[float, np.ndarray]:
        """
        Scales the imaginary part of IQ data.

        Args:
            data: IQ Data

        Returns:
             data: The scaled imaginary part of the data.
        """
        return self.scale * data.imag


@base.counts
@base.prev_node(SystemDiscriminator)
class Population(DataAction):
    """Count data post processing. This returns population."""

    def process(self, data: Counts, **kwargs):
        """
        Args:
            data: in Count format.

        Returns:
            populations: The counts divided by the number of shots.
        """

        populations = np.zeros(len(list(data.keys())[0]))

        shots = 0
        for bit_str, count in data.items():
            shots += 1
            for ind, bit in enumerate(bit_str):
                if bit == '1':
                    populations[ind] += count

        return populations / shots
