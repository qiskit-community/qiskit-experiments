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

"""Tests for base experiment framework."""

from test.fake_backend import FakeBackend
from test.fake_experiment import FakeExperiment

import ddt

from qiskit import QuantumCircuit
from qiskit.test import QiskitTestCase


@ddt.ddt
class TestFramework(QiskitTestCase):
    """Test Base Experiment"""

    @ddt.data(None, 1, 2, 3)
    def test_job_splitting(self, max_experiments):
        """Test job splitting"""

        num_circuits = 10
        backend = FakeBackend(max_experiments=max_experiments)

        class Experiment(FakeExperiment):
            """Fake Experiment to test job splitting"""

            def _circuits(self, backend=None):
                """Generate fake circuits"""
                qc = QuantumCircuit(1)
                qc.measure_all()
                return num_circuits * [qc]

        exp = Experiment(0)
        expdata = exp.run(backend)
        job_ids = expdata.job_ids

        # Comptue expected number of jobs
        if max_experiments is None:
            num_jobs = 1
        else:
            num_jobs = num_circuits // max_experiments
            if num_circuits % max_experiments:
                num_jobs += 1
        self.assertEqual(len(job_ids), num_jobs)

    def test_logical_circuits(self):
        """Test if user can get logical circuits."""
        backend = FakeBackend(n_qubits=3)

        class Experiment(FakeExperiment):
            """Fake Experiment to test transpile."""

            def _circuits(self, backend=None):
                """Generate fake circuits"""
                qc = QuantumCircuit(1, 1)
                qc.x(0)
                qc.measure(0, 0)

                return [qc]

        exp = Experiment(1)
        test_circ = exp.circuits(backend)[0]

        ref_circ = QuantumCircuit(1, 1)
        ref_circ.x(0)
        ref_circ.measure(0, 0)

        self.assertEqual(test_circ, ref_circ)

    def test_physical_circuits(self):
        """Test if user can get physical circuits."""
        backend = FakeBackend(n_qubits=3)

        class Experiment(FakeExperiment):
            """Fake Experiment to test transpile."""

            def _circuits(self, backend=None):
                """Generate fake circuits"""
                qc = QuantumCircuit(1, 1)
                qc.x(0)
                qc.measure(0, 0)

                return [qc]

        exp = Experiment(2)
        test_circ = exp.circuits(backend, run_transpile=True)[0]

        ref_circ = QuantumCircuit(3, 1)
        ref_circ.x(2)
        ref_circ.measure(2, 0)

        self.assertEqual(test_circ, ref_circ)

    def test_pre_transpile_action(self):
        """Test pre transpile."""
        backend = FakeBackend(n_qubits=1)

        class Experiment(FakeExperiment):
            """Fake Experiment to test transpile."""

            def _circuits(self, backend=None):
                """Generate fake circuits"""
                qc = QuantumCircuit(1, 1)
                qc.y(0)
                qc.measure(0, 0)

                return [qc]

            def _pre_transpile_action(self, backend):
                """Update basis gates with y gate."""
                basis_gates = backend.configuration().basis_gates

                if "y" not in basis_gates:
                    basis_gates.append("y")

                self.set_transpile_options(basis_gates=basis_gates)

        exp = Experiment(0)
        test_circ = exp.circuits(backend, run_transpile=True)[0]

        ref_circ = QuantumCircuit(1, 1)
        ref_circ.y(0)
        ref_circ.measure(0, 0)

        self.assertEqual(test_circ, ref_circ)

    def test_post_transpile_action(self):
        """Test post transpile."""
        backend = FakeBackend(n_qubits=1)

        class Experiment(FakeExperiment):
            """Fake Experiment to test transpile."""

            def _circuits(self, backend=None):
                """Generate fake circuits"""
                qc = QuantumCircuit(1, 1)
                qc.x(0)
                qc.measure(0, 0)

                return [qc]

            def _post_transpile_action(self, circuits, backend):
                """Add test metadata."""
                for circ in circuits:
                    circ.metadata = {"test": "test"}

                return circuits

        exp = Experiment(0)
        test_circ = exp.circuits(backend, run_transpile=True)[0]

        ref_metadata = {"test": "test"}

        self.assertDictEqual(test_circ.metadata, ref_metadata)

    def test_post_analysis(self):
        """Test post analysis."""
        backend = FakeBackend(n_qubits=1)
        mutable_target_obj = {"type": "dummy_data"}

        class Experiment(FakeExperiment):
            """Fake Experiment to test transpile."""

            def __init__(self, qubit=0, target_obj=None):
                """Initialise the fake experiment."""
                super().__init__(qubit)
                self.target = target_obj

            def _circuits(self, backend=None):
                """Generate fake circuits"""
                qc = QuantumCircuit(1)
                qc.measure_all()

                return [qc]

            def _post_analysis_action(self, experiment_data):
                """Update target object."""
                self.target["type"] = experiment_data.metadata["experiment_type"]

        exp = Experiment(0, target_obj=mutable_target_obj)
        exp.run(backend).block_for_results()

        self.assertEqual(mutable_target_obj["type"], "fake_test_experiment")

    def test_post_analysis_from_run_analysis(self):
        """Test post analysis called from run analysis."""
        backend = FakeBackend(n_qubits=1)
        mutable_target_obj = {"type": "dummy_data"}

        class Experiment(FakeExperiment):
            """Fake Experiment to test transpile."""

            def __init__(self, qubit=0, target_obj=None):
                """Initialise the fake experiment."""
                super().__init__(qubit)
                self.target = target_obj

            def _circuits(self, backend=None):
                """Generate fake circuits"""
                qc = QuantumCircuit(1)
                qc.measure_all()

                return [qc]

            def _post_analysis_action(self, experiment_data):
                """Update target object."""
                self.target["type"] = experiment_data.metadata["experiment_type"]

        exp = Experiment(0, target_obj=mutable_target_obj)
        exp_data = exp.run(backend, analysis=False).block_for_results()
        self.assertEqual(mutable_target_obj["type"], "dummy_data")

        # post analysis should be called
        exp.run_analysis(experiment_data=exp_data)
        self.assertEqual(mutable_target_obj["type"], "fake_test_experiment")
