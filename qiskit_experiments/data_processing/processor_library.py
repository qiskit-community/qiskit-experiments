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
from qiskit_experiments.data_processing import nodes


def get_processor(
    meas_level: MeasLevel = MeasLevel.CLASSIFIED,
    meas_return: str = "avg",
    normalize: bool = True,
    init_qubits: bool = True,
    memory: bool = False,
    rep_delay: float = None,
    num_qubits: int = None,
) -> DataProcessor:
    """Get a DataProcessor that produces a continuous signal given the options.

    Args:
        meas_level: The measurement level of the data to process.
        meas_return: The measurement return (single or avg) of the data to process.
        normalize: Add a data normalization node to the Kerneled data processor.
        init_qubits: If False, the qubits are not reset to the ground state after a measurement.
        memory: If True, single-shot measurement bitstrings are returned.
        rep_delay: The delay between a measurement and the subsequent circuit.
        num_qubits: The number of qubits.

    Returns:
        An instance of DataProcessor capable of dealing with the given options.

    Raises:
        DataProcessorError: if the measurement level is not supported.
    """

    # restless data processing.
    if meas_level == MeasLevel.CLASSIFIED and not init_qubits and memory and rep_delay < 100e-6:
        processor = DataProcessor(
            "memory",
            [
                nodes.RestlessToCounts(header={"memory_slots": num_qubits}),
                nodes.Probability("1" * num_qubits),
            ],
        )

        return processor

    if meas_level == MeasLevel.CLASSIFIED:
        return DataProcessor("counts", [nodes.Probability("1")])

    if meas_level == MeasLevel.KERNELED:
        if meas_return == "single":
            processor = DataProcessor("memory", [nodes.AverageData(axis=1), nodes.SVD()])
        else:
            processor = DataProcessor("memory", [nodes.SVD()])

        if normalize:
            processor.append(nodes.MinMaxNormalize())

        return processor

    raise DataProcessorError(f"Unsupported measurement level {meas_level}.")
