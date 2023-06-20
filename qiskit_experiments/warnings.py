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

"""A collection of typical warnings."""

from qiskit.utils.lazy_tester import LazyImportTester


HAS_SKLEARN = LazyImportTester(
    {
        "sklearn.discriminant_analysis": (
            "LinearDiscriminantAnalysis",
            "QuadraticDiscriminantAnalysis",
        )
    },
    name="scikit-learn",
    install="pip install scikit-learn",
)

HAS_DYNAMICS = LazyImportTester(
    "qiskit_dynamics", name="qiskit-dynamics", install="pip install qiskit-dynamics"
)
