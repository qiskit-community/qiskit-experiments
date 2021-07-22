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

Data processing is the act of taking taking the data returned by the backend and
converting it into a format that can be analyzed. For instance, counts can be
converted to a probability while two-dimensional IQ data may be converted to a
one-dimensional signal.

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
    ToImag
    ToReal
    SVD
    AverageData
"""

from .data_action import DataAction, TrainableDataAction
from .nodes import (
    Probability,
    ToImag,
    ToReal,
    SVD,
    AverageData,
)

from .data_processor import DataProcessor
