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
===============================================================================================
Randomized Benchmarking Experiments (:mod:`qiskit_experiments.library.randomized_benchmarking`)
===============================================================================================

.. currentmodule:: qiskit_experiments.library.randomized_benchmarking

Experiments
===========
.. autosummary::
    :toctree: ../stubs/
    :template: autosummary/experiment.rst

    StandardRB
    InterleavedRB
    LayerFidelity
    LayerFidelityUnitary


Analysis
========

.. autosummary::
    :toctree: ../stubs/
    :template: autosummary/analysis.rst

    RBAnalysis
    InterleavedRBAnalysis
    LayerFidelityAnalysis

Synthesis
=========

.. autosummary::
    :toctree: ../stubs/

    RBDefaultCliffordSynthesis

Utilities
=========

.. autosummary::
    :toctree: ../stubs/

    RBUtils
    CliffordUtils

.. _synth-methods-lbl:

Synthesis Methods
=================

There are a few built-in options for the Clifford synthesis method:

* ``rb_default`` (default) for n<=2 Cliffords this methods will transpile using ``optimization_level=1``. 
  For 3 or more qubits the behavior is similar but a custom transpilation sequence is used to avoid
  the transpiler changing the layout of the circuit.

* ``clifford_synthesis_method='basis_only'`` will use ``optimization_level=0``.

* ``clifford_synthesis_method='1Q_fixed`` will use a ``rz-sx-rz-sx-rz`` decomposition for the 1Q 
  Cliffords and the default for the 2Q cliffords. This is most relevant for :class:`.LayerFidelity` 
  experiments because it will keep a fixed structure.

"""
from .standard_rb import StandardRB
from .interleaved_rb_experiment import InterleavedRB
from .rb_analysis import RBAnalysis
from .interleaved_rb_analysis import InterleavedRBAnalysis
from .clifford_utils import CliffordUtils
from .rb_utils import RBUtils
from .clifford_synthesis import RBDefaultCliffordSynthesis
from .layer_fidelity import LayerFidelity
from .layer_fidelity_unitary import LayerFidelityUnitary
from .layer_fidelity_analysis import LayerFidelityAnalysis
