# This code is part of Qiskit.
#
# (C) Copyright IBM 2017.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"The Qiskit Experiments setup file."

import os
from setuptools import setup, find_packages

with open("requirements.txt", encoding="utf-8") as f:
    REQUIREMENTS = f.read().splitlines()

with open("requirements-extras.txt", encoding="utf-8") as f:
    EXTRAS = f.read().splitlines()

version_path = os.path.abspath(
    os.path.join(os.path.join(os.path.dirname(__file__), "qiskit_experiments"), "VERSION.txt")
)
with open(version_path, "r", encoding="utf-8") as fd:
    version = fd.read().rstrip()

# Read long description from README.
README_PATH = os.path.join(os.path.abspath(os.path.dirname(__file__)), "README.md")
with open(README_PATH, encoding="utf-8") as readme_file:
    README = readme_file.read()

setup(
    name="qiskit-experiments",
    version=version,
    description="Software for developing quantum computing programs",
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://github.com/Qiskit-Community/qiskit-experiments",
    author="Qiskit Development Team",
    author_email="qiskit@us.ibm.com",
    license="Apache 2.0",
    classifiers=[
        "Environment :: Console",
        "License :: OSI Approved :: Apache Software License",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: MacOS",
        "Operating System :: POSIX :: Linux",
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering",
    ],
    keywords="qiskit sdk quantum",
    packages=find_packages(exclude=["test*"]),
    install_requires=REQUIREMENTS,
    extras_require={"extras": EXTRAS},
    include_package_data=True,
    python_requires=">=3.9",
    project_urls={
        "Bug Tracker": "https://github.com/Qiskit-Community/qiskit-experiments/issues",
        "Documentation": "https://qiskit-community.github.io/qiskit-experiments",
        "Source Code": "https://github.com/Qiskit-Community/qiskit-experiments",
    },
    zip_safe=False,
    entry_points={
        "qiskit.synthesis": [
            "clifford.rb_default = qiskit_experiments.library.randomized_benchmarking.clifford_synthesis:RBDefaultCliffordSynthesis",
        ],
    },
)
