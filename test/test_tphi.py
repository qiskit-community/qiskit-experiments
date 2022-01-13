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
Test T2Ramsey experiment
"""
import numpy as np
import time

from qiskit.utils import apply_prefix
from qiskit.providers import BackendV1
from qiskit.providers.options import Options
from qiskit.providers.models import QasmBackendConfiguration
from qiskit.result import Result
from qiskit.test import QiskitTestCase
from qiskit_experiments.framework.composite.composite_experiment import CompositeExperiment
from qiskit_experiments.library import T1, T2Ramsey, Tphi
from qiskit_experiments.test.utils import FakeJob
from qiskit_experiments.test.tphi_backend import TphiBackend


class TestTphi(QiskitTestCase):
    """Test Tphi experiment"""

    def test_tphi(self):
        """
        Run the Tphi backend
        """
        unit = "us"
        dt_factor = apply_prefix(1, unit)
        qubit = 0
        delays_t1 = list(range(1, 40, 3))
        delays_t2 = list(range(1, 51, 2))
        exp = Tphi(qubit=0, delays_t1=delays_t1, delays_t2=delays_t2, unit="s", osc_freq=0.1)

        backend = TphiBackend(t1=10, t2ramsey=25, freq=0.1)
        expdata = exp.run(backend, experiment_data=None).block_for_results()
        display(expdata.figure(0))
