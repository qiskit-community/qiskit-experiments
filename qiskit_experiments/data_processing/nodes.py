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

from abc import abstractmethod
from typing import Any, Dict, Optional
import numpy as np

from qiskit_experiments.data_processing.base import DataAction


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
        self._accepted_inputs = ["memory"]

    @property
    def node_output(self) -> str:
        """Key under which Kernel stores the data."""
        return "memory"

    def process(self, data: Dict[str, Any]):
        """
        Args:
            data: The data dictionary to process.

        Raises:
            DataProcessorError: if the data has no memory.
        """
        data[self.node_output] = self.kernel.kernel(np.array(data["memory"]))


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
        self._accepted_inputs = ["memory", "memory_real", "memory_imag"]

    @property
    def node_output(self) -> str:
        """Key under which Discriminator stores the data."""
        return "counts"

    def process(self, data: Dict[str, Any]):
        """
        Discriminate the data to transform it into counts.

        Args:
            data: The data in a format that can be understood by the discriminator.

        Raises:
            DataProcessorError: if the data does not contain memory.
        """
        data[self.node_output] = self.discriminator.discriminate(np.array(data["memory"]))


class IQPart(DataAction):
    """Abstract class for IQ data post-processing."""

    def __init__(self, scale: Optional[float] = 1.0, average: bool = False):
        """
        Args:
            scale: scale by which to multiply the real part of the data.
            average: if True the single-shots are averaged.
        """
        self.scale = scale
        self.average = average
        super().__init__()
        self._accepted_inputs = ["memory"]

    @abstractmethod
    def _index(self) -> int:
        """Return 0 for real and 1 for imaginary part."""

    def process(self, data: Dict[str, Any]):
        """
        Modifies the data inplace by taking the real part of the memory and
        scaling it by the given factor.

        Args:
            data: The data dict. IQ data is stored under memory.

        Raises:
            DataProcessorError: if the data does not contain memory.
        """

        # Single shot data
        if isinstance(data["memory"][0][0], list):
            new_mem = []
            for shot in data["memory"]:
                new_mem.append([self.scale * iq_qubit[self._index()] for iq_qubit in shot])

            if self.average:
                new_mem = list(np.mean(np.array(new_mem), axis=0))

        # Averaged data
        else:
            new_mem = [self.scale * iq_qubit[self._index] for iq_qubit in data["memory"]]

        data[self.node_output] = new_mem


class ToReal(IQPart):
    """IQ data post-processing. Isolate the real part of the IQ data."""

    @property
    def node_output(self) -> str:
        """Key under which ToReal stores the data."""
        return "memory_real"

    def _index(self) -> int:
        """Return 0 for real part."""
        return 0


class ToImag(IQPart):
    """IQ data post-processing. Isolate the imaginary part of the IQ data."""

    @property
    def node_output(self) -> str:
        """Key under which ToImag stores the data."""
        return "memory_imag"

    def _index(self) -> int:
        """Return 0 for real part."""
        return 1


class Population(DataAction):
    """Count data post processing. This returns population."""

    def __init__(self):
        """Initialize a counts to population data conversion."""
        super().__init__()
        self._accepted_inputs = ["counts"]

    @property
    def node_output(self) -> str:
        """Key under which Population stores the data."""
        return "populations"

    def process(self, data: Dict[str, Any]):
        """
        Args:
            data: The data dictionary. This will modify the dict in place,
                taking the data under counts and adding the corresponding
                populations.

        Raises:
            DataProcessorError: if counts are not in the given data.
        """

        counts = data.get("counts")

        populations = np.zeros(len(list(counts.keys())[0]))

        shots = 0
        for bit_str, count in counts.items():
            shots += count
            for ind, bit in enumerate(bit_str):
                if bit == "1":
                    populations[ind] += count

        data[self.node_output] = populations / shots
