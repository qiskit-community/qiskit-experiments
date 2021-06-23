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
Auto docstring generation package for Qiskit expeirments.

This tool guarantees the standardized docstring among the experiments and removes
variation of docstring quality depending on the coders.
In addition, this provides auto docstring generation of experiment options, considering
the inheritance of experiment classes, i.e. a documentation for options in a parent class
also appears in all subclasses. This drastically reduces the overhead of docstring management,
while providing users with the standardized and high quality documentation.
"""

from .experiment_docs import auto_experiment_documentation
from .set_option_docs import auto_options_method_documentation
from .analysis_docs import auto_analysis_documentation
from .descriptions import OptionsField, Reference, to_options
