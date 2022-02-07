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

from qiskit.qobj.utils import MeasLevel, MeasReturnType

from qiskit_experiments.framework import ExperimentData, Options
from qiskit_experiments.data_processing.exceptions import DataProcessorError
from qiskit_experiments.data_processing.data_processor import DataProcessor
from qiskit_experiments.data_processing import nodes


def get_processor(
    experiment_data: ExperimentData, analysis_options: Options, index: int = -1
) -> DataProcessor:
    """Get a DataProcessor that produces a continuous signal given the options.

    Args:
        experiment_data: The experiment data that holds all the data and metadata needed
             to determine the data processor to use to process the data for analysis.
        analysis_options: The analysis options with which to analyze the data. The options that
             are relevant for the configuration of a data processor are:
             - normalization (bool): A boolean to specify if the data should be normalized to
               the interval [0, 1]. The default is True. This option is only relevant if
               kerneled data is used.
             - outcome (string): The measurement outcome that will be passed to a Probability node.
               The default value is a string of 1's where the length of the string is the number of
               qubits, e.g. '111' for three qubits.
             - restless_threshold (float): If (repetition delay / T1) is below a given threshold for
               all physical qubit T1 values and active reset is turned off in the experiment run options
               (init_qubit=False) the experiment is recognized as a restless experiment and will be
               analyzed accordingly. The default value is 1.
        index: The index of the job for which to get a data processor. The default value is -1.

    Returns:
        An instance of DataProcessor capable of processing the data for the corresponding job.

    Notes:
        The following relevant arguments are extracted from the experiment_data metadata run options:
            - meas_level (MeasLevel): The measurement level of the data to process which is by default
              set to MeasLevel.CLASSIFIED.
            - meas_return (MeasReturnType): The measurement return (single or avg) of the data to process.
              The default is MeasReturnType.AVERAGE.
            - init_qubits (bool): If False, the qubits are not reset to the ground state after a measurement.
              The default is True.
            - physical qubits: The physical qubits used in the experiment.
            - memory (bool): If True, single-shot measurement bitstrings are returned. The default is False.
            - rep_delay (float): The delay between a measurement and the subsequent circuit. The default is
              None.

        In addition, the following argument is extracted from the experiment_data:
            - t1_values (List[float]): The T1 values of the physical qubits at the time of the experiment.

    Raises:
        DataProcessorError: if the measurement level is not supported.
        DataProcessorError: if no single-shot memory is present but the run options suggest that a
                            restless experiment was run.
    """

    run_options = experiment_data.metadata["job_metadata"][index].get("run_options", {})

    meas_level = run_options.get("meas_level", MeasLevel.CLASSIFIED)
    meas_return = run_options.get("meas_return", MeasReturnType.AVERAGE)
    normalize = analysis_options.get("normalization", True)
    esp_enabled = analysis_options.get("use_measure_esp", False)

    physical_qubits = experiment_data.metadata["physical_qubits"]
    num_qubits = len(physical_qubits)

    outcome = analysis_options.get("outcome", "1" * num_qubits)

    init_qubits = run_options.get("init_qubits", True)
    memory = run_options.get("memory", False)
    rep_delay = run_options.get("rep_delay", None)
    restless_threshold = analysis_options.get("restless_threshold", 1)

    # restless data processing.
    restless = False
    if rep_delay and not init_qubits:
        if esp_enabled:
            raise DataProcessorError(
                f"Restless experiments are not compatible with the excited "
                f"state promotion readout analysis option."
            )
        if num_qubits > 1:
            raise DataProcessorError(
                f"To date, only single-qubit restless measurements can be processed."
            )
        t1_values = [
            experiment_data.backend.properties().qubit_property(physical_qubit)["T1"][0]
            for physical_qubit in physical_qubits
        ]
        if [rep_delay / t1_value < restless_threshold for t1_value in t1_values] == [
            True
        ] * num_qubits:
            restless = True

    if meas_level == MeasLevel.CLASSIFIED and memory and restless:
        return DataProcessor(
            "memory",
            [
                nodes.RestlessToCounts(num_qubits),
                nodes.Probability(outcome),
            ],
        )

    if restless and not memory:
        raise DataProcessorError(
            f"Run options suggest restless data but no single-shot memory is present."
        )

    if meas_level == MeasLevel.CLASSIFIED:
        return DataProcessor("counts", [nodes.Probability(outcome)])

    if meas_level == MeasLevel.KERNELED:
        if meas_return == MeasReturnType.SINGLE:
            processor = DataProcessor("memory", [nodes.AverageData(axis=1), nodes.SVD()])
        else:
            processor = DataProcessor("memory", [nodes.SVD()])

        if normalize:
            processor.append(nodes.MinMaxNormalize())

        return processor

    raise DataProcessorError(f"Unsupported measurement level {meas_level}.")
