# This code is part of Qiskit.
#
# (C) Copyright IBM 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
"""
=======================================================================================
Discriminator Experiments (:mod:`qiskit_experiments.measurememt.discriminator`)
=======================================================================================
.. currentmodule:: qiskit_experiments.discriminator

The discriminator classifies kerneled (level 1) data to state (level 2) data. These 
discriminator experiments run calibration circuits to obtain kerneled data corresponding
to different output states, then uses various methods to classify the data into output
states.

Experiments
===========
.. autosummary::
    :toctree: ../stubs/
    :template: autosummary/experiment.rst

    Discriminator


Analysis
========
.. autosummary::
    :toctree: ../stubs/

    DiscriminatorAnalysis
"""

from .discriminator_experiment import Discriminator
from .discriminator_analysis import DiscriminatorAnalysis
