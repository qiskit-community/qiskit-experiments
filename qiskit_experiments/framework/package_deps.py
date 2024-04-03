# This code is part of Qiskit.
#
# (C) Copyright IBM 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
"""
Functions for checking and reporting installed package versions.
"""

from __future__ import annotations

import warnings
from functools import lru_cache
from importlib.metadata import version as metadata_version

from packaging.version import InvalidVersion, Version

from qiskit.utils.lazy_tester import LazyImportTester


__all__ = ["HAS_SKLEARN", "HAS_DYNAMICS", "qiskit_version", "version_is_at_least"]


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
    "qiskit_dynamics",
    name="qiskit-dynamics",
    install="pip install qiskit-dynamics",
)


def qiskit_version() -> dict[str, str]:
    """Return a dict with Qiskit names and versions."""
    return {p: metadata_version(p) for p in ("qiskit", "qiskit-experiments")}


@lru_cache(maxsize=None)
def version_is_at_least(package: str, version: str) -> bool:
    """Return True if the installed version of package greater than minimum version

    Args:
        package: Name of the package
        version: Minimum version name as a string. This should just include
            major, minor, and micro parts. The function will add ``.dev0`` to
            also catch any pre-release versions (otherwise ``0.5.0a1`` would
            evaluate as less than ``0.5.0``).

    Returns:
        True if installed version greater than ``version``. False if it is less
        or if the installed version of ``package`` can not be parsed using the
        specifications of PEP440.

    Raises:
        PackageNotFoundError:
            If ``package`` is not installed.
    """
    raw_installed_version = metadata_version(package)
    try:
        installed_version = Version(raw_installed_version)
    except InvalidVersion:
        warnings.warn(
            (
                f"Version string of installed {package} does not match PyPA "
                f"specification. Treating as less than {version}."
            ),
            RuntimeWarning,
        )
        return False
    return installed_version >= Version(f"{version}.dev0")
