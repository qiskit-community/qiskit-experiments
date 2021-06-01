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

from qiskit.qobj.utils import MeasLevel

from qiskit_experiments.data_processing.exceptions import DataProcessorError
from qiskit_experiments.data_processing.data_processor import DataProcessor
from qiskit_experiments.data_processing.nodes import AverageData, Probability, SVD


def get_to_signal_processor(
    meas_level: MeasLevel = MeasLevel.CLASSIFIED, meas_return: str = "avg"
) -> DataProcessor:
    """Get a DataProcessor that produces a continuous signal given the options.

    Args:
        meas_level: The measurement level of the data to process.
        meas_return: The measurement return (single or avg) of the data to process.

    Returns:
        An instance of DataProcessor capable of dealing with the given options.

    Raises:
        DataProcessorError: if the measurement level is not supported.
    """
    if meas_level == MeasLevel.CLASSIFIED:
        return DataProcessor("counts", [Probability("1")])

    if meas_level == MeasLevel.KERNELED:
        if meas_return == "single":
            return DataProcessor("memory", [AverageData(), SVD()])
        else:
            return DataProcessor("memory", [SVD()])

    raise DataProcessorError(f"Unsupported measurement level {meas_level}.")
