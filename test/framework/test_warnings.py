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

# pylint: disable=unused-argument, unused-variable
"""Test warning helper."""

import subprocess
import sys
import textwrap
from test.base import QiskitExperimentsTestCase

from qiskit_experiments.framework import BaseExperiment


class TempExperiment(BaseExperiment):
    """Fake experiment."""

    def __init__(self, physical_qubits):
        super().__init__(physical_qubits)

    def circuits(self):
        pass


class TestWarningsHelper(QiskitExperimentsTestCase):
    """Test case for warnings decorator with tricky behavior."""

    def test_warn_sklearn(self):
        """Test that a suggestion to import scikit-learn is given when appropriate"""
        script = """
        import builtins
        disallowed_imports = {"sklearn"}
        old_import = builtins.__import__
        def guarded_import(name, *args, **kwargs):
            if name == "sklearn" or name.startswith("sklearn."):
                raise import_error(f"Import of {name} not allowed!")
            return old_import(name, *args, **kwargs)
        builtins.__import__ = guarded_import
        # Raise Exception on imports so that ImportError can't be caught
        import_error = Exception
        import qiskit_experiments
        print("qiskit_experiments imported!")
        # Raise ImportError so the guard can catch it
        import_error = ImportError
        from qiskit_experiments.data_processing.sklearn_discriminators import SkLDA
        SkLDA.from_config({})
        """
        script = textwrap.dedent(script)

        proc = subprocess.run(
            [sys.executable, "-c", script], check=False, text=True, capture_output=True
        )

        self.assertTrue(
            proc.stdout.startswith("qiskit_experiments imported!"),
            msg="Failed to import qiskit_experiments without sklearn",
        )

        self.assertNotEqual(
            proc.returncode,
            0,
            msg="scikit-learn usage did not error without scikit-learn available",
        )
        self.assertTrue(
            "qiskit.exceptions.MissingOptionalLibraryError" in proc.stderr
            and "scikit-learn" in proc.stderr,
            msg="scikit-learn import guard did not run on scikit-learn usage",
        )
