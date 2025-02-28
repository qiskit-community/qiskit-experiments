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

import warnings
from typing import Union, Optional, List
from qiskit.qobj.utils import MeasLevel, MeasReturnType

from qiskit_experiments.framework import ExperimentData, Options
from qiskit_experiments.data_processing.exceptions import DataProcessorError
from qiskit_experiments.data_processing.data_action import DataAction
from qiskit_experiments.data_processing.data_processor import DataProcessor
from qiskit_experiments.data_processing.nodes import ProjectorType
from qiskit_experiments.data_processing import nodes


def get_kerneled_processor(
    dimensionality_reduction: Union[ProjectorType, str],
    meas_return: str,
    normalize: bool,
    pre_nodes: Optional[List[DataAction]] = None,
) -> DataProcessor:
    """Get a DataProcessor for `meas_level=1` data that returns a one-dimensional signal.

    Args:
        dimensionality_reduction: Type of the node that will reduce the two-dimensional data to
            one dimension.
        meas_return: Type of data returned by the backend, i.e., averaged data or single-shot data.
        normalize: If True then normalize the output data to the interval ``[0, 1]``.
        pre_nodes: any nodes to be applied first in the data processing chain

    Returns:
        An instance of DataProcessor capable of processing `meas_level=MeasLevel.KERNELED` data for
        the corresponding job.

    Raises:
        DataProcessorError: If the wrong dimensionality reduction for kerneled data
                is specified.
    """

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

    node = pre_nodes or []

    if meas_return == "single":
        node.append(nodes.AverageData(axis=1))

    node.append(projector())

    if normalize:
        node.append(nodes.MinMaxNormalize())

    return DataProcessor("memory", node)


def get_processor(experiment_data: ExperimentData, analysis_options: Options) -> DataProcessor:
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

    Returns:
        An instance of DataProcessor capable of processing the data for the corresponding job.

    Notes:
        The `physical_qubits` argument is extracted from the `experiment_data`
        metadata and is used to determine the default `outcome` to extract from
        classified data if it was not given in the analysis options.

    Raises:
        DataProcessorError: If the measurement level is not supported.
    """
    metadata = experiment_data.metadata
    if "job_metadata" in metadata:
        # Backwards compatibility for old experiment data
        # remove job metadata and add required fields to new location in metadata
        job_meta = metadata.pop("job_metadata")
        run_options = job_meta[-1].get("run_options", {})
        for opt in ["meas_level", "meas_return"]:
            if opt in run_options:
                metadata[opt] = run_options[opt]
        warnings.warn(
            "The analyzed ExperimentData contains deprecated data processor "
            " job_metadata which has been been updated to current metadata format. "
            "If this data was loaded from a database service you should re-save it "
            "to update the metadata in the database.",
            DeprecationWarning,
        )

    meas_level = metadata.get("meas_level", MeasLevel.CLASSIFIED)
    meas_return = metadata.get("meas_return", MeasReturnType.AVERAGE)
    normalize = analysis_options.get("normalization", True)
    dimensionality_reduction = analysis_options.get("dimensionality_reduction", ProjectorType.SVD)

    if meas_level == MeasLevel.CLASSIFIED:
        num_qubits = len(metadata.get("physical_qubits", [0]))
        outcome = analysis_options.get("outcome", "1" * num_qubits)
        return DataProcessor("counts", [nodes.Probability(outcome)])

    if meas_level == MeasLevel.KERNELED:
        return get_kerneled_processor(dimensionality_reduction, meas_return, normalize)

    raise DataProcessorError(f"Unsupported measurement level {meas_level}.")
