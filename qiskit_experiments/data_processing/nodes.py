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
from typing import Any, Dict, Optional, Tuple
import numpy as np

from qiskit_experiments.data_processing.base import DataAction


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
    def _process(self, point: Tuple[float, float]) -> float:
        """Defines how the IQ point will be processed.

        Args:
            point: An IQ point as a tuple of two float, i.e. (real, imaginary).

        Returns:
            Processed IQ point.
        """

    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Modifies the data inplace by taking the real part of the memory and
        scaling it by the given factor.

        Args:
            data: The data dict. IQ data is stored under memory.

        Returns:
            processed data: A dict with the data.
        """

        # Single shot data
        if isinstance(data["memory"][0][0], list):
            new_mem = []
            for shot in data["memory"]:
                new_mem.append([self.scale * self._process(iq_point) for iq_point in shot])

            if self.average:
                new_mem = list(np.mean(np.array(new_mem), axis=0))

        # Averaged data
        else:
            new_mem = [self.scale * self._process(iq_point) for iq_point in data["memory"]]

        return {self.__node_output__: new_mem}


class ToReal(IQPart):
    """IQ data post-processing. Isolate the real part of the IQ data."""

    __node_output__ = "memory_real"

    def _process(self, point: Tuple[float, float]) -> float:
        """Defines how the IQ point will be processed.

        Args:
            point: An IQ point as a tuple of two float, i.e. (real, imaginary).

        Returns:
            The real part of the IQ point.
        """
        return point[0]


class ToImag(IQPart):
    """IQ data post-processing. Isolate the imaginary part of the IQ data."""

    __node_output__ = "memory_imag"

    def _process(self, point: Tuple[float, float]) -> float:
        """Defines how the IQ point will be processed.

        Args:
            point: An IQ point as a tuple of two float, i.e. (real, imaginary).

        Returns:
            The imaginary part of the IQ point.
        """
        return point[1]


class Population(DataAction):
    """Count data post processing. This returns population."""

    __node_output__ = "populations"

    def __init__(self):
        """Initialize a counts to population data conversion."""
        super().__init__()
        self._accepted_inputs = ["counts"]

    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Args:
            data: The data dictionary. This will modify the dict in place,
                taking the data under counts and adding the corresponding
                populations.

        Returns:
            processed data: A dict with the populations.
        """

        counts = data.get("counts")

        populations = np.zeros(len(list(counts.keys())[0]))

        shots = 0
        for bit_str, count in counts.items():
            shots += count
            for ind, bit in enumerate(bit_str):
                if bit == "1":
                    populations[ind] += count

        return {self.__node_output__: populations / shots}
