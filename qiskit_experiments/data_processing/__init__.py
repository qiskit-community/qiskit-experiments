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
"""
===========================================================
Data Processing (:mod:`qiskit_experiments.data_processing`)
===========================================================

.. currentmodule:: qiskit_experiments.data_processing

Data processing is the act of taking the data returned by the backend and
converting it into a format that can be analyzed.
It is implemented as a chain of data processing steps that transform various input data,
e.g. IQ data, into a desired format, e.g. population, which can be analyzed.

These data transformations may consist of multiple steps, such as kerneling and discrimination.
Each step is implemented by a :class:`~qiskit_experiments.data_processing.data_action.DataAction`
also called a `node`.

The data processor implements the :meth:`__call__` method. Once initialized, it
can thus be used as a standard python function:

.. code-block:: python

    processor = DataProcessor(input_key="memory", [Node1(), Node2(), ...])
    out_data = processor(in_data)

The data input to the processor is a sequence of dictionaries each representing the result
of a single circuit. The output of the processor is a numpy array whose shape and data type
depend on the combination of the nodes in the data processor.

Uncertainties that arise from quantum measurements or finite sampling can be taken into account
in the nodes: a standard error can be generated in a node and can be propagated
through the subsequent nodes in the data processor.
Correlation between computed values is also considered.


Classes
=======
.. autosummary::
    :toctree: ../stubs/

    DataProcessor
    DataAction
    TrainableDataAction


Data Processing Nodes
=====================
.. autosummary::
    :toctree: ../stubs/

    Probability
    MarginalizeCounts
    ToImag
    ToReal
    SVD
    DiscriminatorNode
    MemoryToCounts
    AverageData
    BasisExpectationValue
    MinMaxNormalize
    ShotOrder
    RestlessNode
    RestlessToCounts
    RestlessToIQ


Discriminators
==============
.. autosummary::
    :toctree: ../stubs/

    BaseDiscriminator
    SkLDA
    SkQDA

Mitigators
==========
.. autosummary::
    :toctree: ../stubs/

    BaseReadoutMitigator
    LocalReadoutMitigator
    CorrelatedReadoutMitigator
"""

from .data_action import DataAction, TrainableDataAction
from .nodes import (
    Probability,
    MarginalizeCounts,
    ToImag,
    ToReal,
    SVD,
    DiscriminatorNode,
    MemoryToCounts,
    AverageData,
    BasisExpectationValue,
    MinMaxNormalize,
    ShotOrder,
    RestlessNode,
    RestlessToCounts,
    RestlessToIQ,
)

from .data_processor import DataProcessor
from .discriminator import BaseDiscriminator
from .mitigation.base_readout_mitigator import BaseReadoutMitigator
from .mitigation.correlated_readout_mitigator import CorrelatedReadoutMitigator
from .mitigation.local_readout_mitigator import LocalReadoutMitigator
from .sklearn_discriminators import SkLDA, SkQDA
