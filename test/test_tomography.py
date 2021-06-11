# -*- coding: utf-8 -*-

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
A Tester for the RB experiment
"""

import itertools as it
import ddt

from qiskit.test import QiskitTestCase
import qiskit.quantum_info as qi
from qiskit.providers.aer import AerSimulator
import qiskit_experiments.tomography as tomo

# TODO: tests for CVXPY fitters
FITTERS = [None, "linear_inversion", "scipy_linear_lstsq", "scipy_gaussian_lstsq"]


@ddt.ddt
class TestStateTomographyExperiment(QiskitTestCase):
    """Test StateTomographyExperiment"""

    @ddt.data(*list(it.product([1, 2], FITTERS)))
    @ddt.unpack
    def test_full_qst(self, num_qubits, fitter):
        """Test 1-qubit QST experiment"""
        backend = AerSimulator()
        seed = 1234
        f_threshold = 0.97
        target = qi.random_statevector(2 ** num_qubits, seed=seed)
        qstexp = tomo.StateTomographyExperiment(target)
        if fitter:
            qstexp.set_analysis_options(fitter=fitter)
        expdata = qstexp.run(backend)
        result = expdata.analysis_result(-1)

        self.assertTrue(result.get("success", False), msg="analysis failed")

        # Check state is density matrix
        state = result.get("state")
        self.assertTrue(
            isinstance(state, qi.DensityMatrix), msg="fitted state is not density matrix"
        )

        # Check fit state fidelity
        self.assertGreater(result.get("value", 0), f_threshold, msg="fit fidelity is low")

        # Manually check fidelity
        fid = qi.state_fidelity(state, target, validate=False)
        self.assertGreater(fid, f_threshold, msg="fitted state fidelity is low")


@ddt.ddt
class TestProcessTomographyExperiment(QiskitTestCase):
    """Test QuantumProcessTomography"""

    @ddt.data(*list(it.product([1, 2], FITTERS)))
    @ddt.unpack
    def test_full_qpt(self, num_qubits, fitter):
        """Test QPT experiment"""
        backend = AerSimulator()
        seed = 1234
        f_threshold = 0.95
        target = qi.random_unitary(2 ** num_qubits, seed=seed)
        qstexp = tomo.ProcessTomographyExperiment(target)
        if fitter:
            qstexp.set_analysis_options(fitter=fitter)
        expdata = qstexp.run(backend)
        result = expdata.analysis_result(-1)

        self.assertTrue(result.get("success", False), msg="analysis failed")

        # Check state is density matrix
        state = result.get("state")
        self.assertTrue(
            isinstance(state, qi.Choi), msg="fitted state is not a Choi matrix"
        )

        # Check fit state fidelity
        self.assertGreater(result.get("value", 0), f_threshold, msg="fit fidelity is low")

        # Manually check fidelity
        fid = qi.process_fidelity(state, target, require_tp=False, require_cp=False)
        self.assertGreater(fid, f_threshold, msg="fitted process fidelity is low")
