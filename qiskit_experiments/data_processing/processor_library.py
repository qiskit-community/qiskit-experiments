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
from qiskit_experiments.framework import Options


def get_processor(
    num_qubits: int = None, analysis_options: Options = Options(normalization=False), **run_options
) -> DataProcessor:
    """Get a DataProcessor that produces a continuous signal given the options.

    Args:
        num_qubits: The number of qubits.
        analysis_options: The experiment analysis options.

    Returns:
        An instance of DataProcessor capable of dealing with the given options.

    Raises:
        DataProcessorError: if the measurement level is not supported.
    """

    meas_level = run_options.get("meas_level", MeasLevel.CLASSIFIED)
    meas_return = run_options.get("meas_return", None)
    normalize = analysis_options.normalization
    init_qubits = run_options.get("init_qubits", True)
    memory = run_options.get("memory", False)
    rep_delay = run_options.get("rep_delay", None)

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
