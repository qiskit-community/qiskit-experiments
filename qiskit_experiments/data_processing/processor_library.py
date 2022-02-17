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
from qiskit_experiments.data_processing.nodes import ProjectorType
from qiskit_experiments.data_processing import nodes


def get_processor(
    experiment_data: ExperimentData,
    analysis_options: Options,
    index: int = -1,
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
            - dimensionality_reduction: An optional string or instance of :class:`ProjectorType`
              to represent the dimensionality reduction node for Kerneled data. For the
              supported nodes, see :class:`ProjectorType`. Typically, these nodes convert
              complex IQ data to real data, for example by performing a singular-value
              decomposition. This argument is only needed for Kerneled data (i.e. level 1)
              and can thus be ignored if Classified data (the default) is used.
            - outcome (string): The measurement outcome that will be passed to a Probability node.
              The default value is a string of 1's where the length of the string is the number of
              qubits, e.g. '111' for three qubits.
        index: The index of the job for which to get a data processor. The default value is -1.

    Returns:
        An instance of DataProcessor capable of processing the data for the corresponding job.

    Notes:
        The `num_qubits` argument is extracted from the `experiment_data` metadata and is used
        to determine the default `outcome` to extract from classified data if it was not given in the
        analysis options.

    Raises:
        DataProcessorError: if the measurement level is not supported.
        DataProcessorError: if the wrong dimensionality reduction for kerneled data
            is specified.
        DataProcessorError: if no single-shot memory is present but the run options suggest that a
                            restless experiment was run.
    """

    run_options = experiment_data.metadata["job_metadata"][index].get("run_options", {})

    meas_level = run_options.get("meas_level", MeasLevel.CLASSIFIED)
    meas_return = run_options.get("meas_return", MeasReturnType.AVERAGE)
    normalize = analysis_options.get("normalization", True)
    dimensionality_reduction = analysis_options.get("dimensionality_reduction", ProjectorType.SVD)
    memory = run_options.get("memory", False)

    restless = is_restless(experiment_data, analysis_options, index)

    if restless and not memory:
        raise DataProcessorError(
            "Run options suggest restless data but no single-shot memory is present."
        )

    if meas_level == MeasLevel.CLASSIFIED:
        num_qubits = experiment_data.metadata.get("num_qubits", 1)
        outcome = analysis_options.get("outcome", "1" * num_qubits)
        if memory and restless:
            return DataProcessor(
                "memory",
                [
                    nodes.RestlessToCounts(num_qubits),
                    nodes.Probability(outcome),
                ],
            )
        return DataProcessor("counts", [nodes.Probability(outcome)])

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


def is_restless(experiment_data: ExperimentData, analysis_options: Options, index: -1) -> bool:
    """Check if the experiment data and analysis options correspond to a restless experiment.
    An experiment is run in a restless setting if the qubits are not reset to the ground
    state after the measurement and the repetition delay is set to a value much smaller
    than the T1 value of the physical qubits in the experiment.

    Args:
        experiment_data: The experiment data that holds all the data and metadata needed
            to determine the data processor to use to process the data for analysis.
        analysis_options: The analysis options with which to analyze the data.
        index: The index of the job for which to get a data processor. The default value is -1.

    Returns:
        True if the experiment is classified as a restless experiment.

    Notes:
        The following relevant arguments are extracted from the experiment_data metadata run options:
            - init_qubits (bool): If False, the qubits are not reset to the ground state after a
              measurement. The default is True.
            - physical qubits: The physical qubits used in the experiment.
            - rep_delay (float): The delay between a measurement and the subsequent circuit.
              The default is None.

        In addition, the following argument is extracted from the experiment_data:
            - t1_values (List[float]): The T1 values of the physical qubits at the time
              of the experiment.

    Raises:
        DataProcessorError: if excited state promotion readout is enabled.
        DataProcessorError: if the experiment is run with more than on qubit.
    """

    run_options = experiment_data.metadata["job_metadata"][index].get("run_options", {})
    init_qubits = run_options.get("init_qubits", True)
    rep_delay = run_options.get("rep_delay", None)
    restless_threshold = analysis_options.get("restless_threshold", 1)
    esp_enabled = analysis_options.get("use_measure_esp", False)


    if rep_delay and not init_qubits:

        if esp_enabled:
            raise DataProcessorError(
                "Restless experiments are not compatible with the excited "
                "state promotion readout analysis option."
            )

        physical_qubits = experiment_data.metadata["physical_qubits"]
        num_qubits = len(physical_qubits)

        if num_qubits > 1:
            raise DataProcessorError(
                "To date, only single-qubit restless measurements can be processed."
            )
        t1_values = [
            experiment_data.backend.properties().qubit_property(physical_qubit)["T1"][0]
            for physical_qubit in physical_qubits
        ]
        if [rep_delay / t1_value < restless_threshold for t1_value in t1_values] == [
            True
        ] * num_qubits:
            restless = True

    return False
