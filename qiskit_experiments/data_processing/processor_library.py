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

"""A collection of functions that return various data processors."""

from enum import Enum
from typing import Union

from qiskit.qobj.utils import MeasLevel

from qiskit_experiments.data_processing.exceptions import DataProcessorError
from qiskit_experiments.data_processing.data_processor import DataProcessor
from qiskit_experiments.data_processing import nodes


class ProjectorType(Enum):
    """Types of projectors for data dimensionality reduction."""
    SVD = nodes.SVD
    ABS = nodes.ToAbs
    REAL = nodes.ToReal
    IMAG = nodes.ToImag


def get_processor(
    meas_level: MeasLevel = MeasLevel.CLASSIFIED,
    meas_return: str = "avg",
    normalize: bool = True,
    dimensionality_reduction: Union[str, ProjectorType] = ProjectorType.SVD,
) -> DataProcessor:
    """Get a DataProcessor that produces a continuous signal given the options.

    Args:
        meas_level: The measurement level of the data to process.
        meas_return: The measurement return (single or avg) of the data to process.
        normalize: Add a data normalization node to the Kerneled data processor.
        dimensionality_reduction: A string to represent the dimensionality reduction
            node. Must be one of SVD, ABS, REAL, IMAG.

    Returns:
        An instance of DataProcessor capable of dealing with the given options.

    Raises:
        DataProcessorError: if the measurement level is not supported.
        DataProcessorError: if the wrong dimensionality reduction for kerneled data
            is specified.
    """
    if meas_level == MeasLevel.CLASSIFIED:
        return DataProcessor("counts", [nodes.Probability("1")])

    if meas_level == MeasLevel.KERNELED:

        try:
            if isinstance(dimensionality_reduction, ProjectorType):
                projector_name = dimensionality_reduction.name
            else:
                projector_name = dimensionality_reduction

            projector = ProjectorType[projector_name].value

        except KeyError as error:
            raise DataProcessorError(
                f"Invalid dimensionality reduction: {dimensionality_reduction}."
            ) from error

        if meas_return == "single":
            processor = DataProcessor("memory", [nodes.AverageData(axis=1), projector()])
        else:
            processor = DataProcessor("memory", [projector()])

        if normalize:
            processor.append(nodes.MinMaxNormalize())

        return processor

    raise DataProcessorError(f"Unsupported measurement level {meas_level}.")
