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
=======================================================================================
Quantum Volume Experiment (:mod:`qiskit_experiments.quantum_volume`)
=======================================================================================

.. currentmodule:: qiskit_experiments.quantum_volume

Quantum Volume (QV) is a single-number metric that can be measured using a concrete protocol
on near-term quantum computers of modest size. The QV method quantifies the largest random circuit
of equal width and depth that the computer successfully implements.
Quantum computing systems with high-fidelity operations, high connectivity, large calibrated gate sets,
and circuit rewriting toolchains are expected to have higher quantum volumes.
See `Qiskit Textbook
<https://qiskit.org/textbook/ch-quantum-hardware/measuring-quantum-volume.html>`_
for an explanation on the QV method.

Experiments
===========
.. autosummary::
    :toctree: ../stubs/

    QuantumVolume


Analysis
========

.. autosummary::
    :toctree: ../stubs/

    QuantumVolumeAnalysis
"""
from .qv_experiment import QuantumVolume
from .qv_analysis import QuantumVolumeAnalysis
