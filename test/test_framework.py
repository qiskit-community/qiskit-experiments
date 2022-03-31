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

from test.fake_experiment import FakeExperiment, FakeAnalysis
from test.base import QiskitExperimentsTestCase
import ddt

from qiskit import QuantumCircuit
from qiskit_experiments.framework import ExperimentData
from qiskit_experiments.test.fake_backend import FakeBackend


@ddt.ddt
class TestFramework(QiskitExperimentsTestCase):
    """Test Base Experiment"""

    @ddt.data(None, 1, 2, 3)
    def test_job_splitting(self, max_experiments):
        """Test job splitting"""

        num_circuits = 10
        backend = FakeBackend(max_experiments=max_experiments)

        class Experiment(FakeExperiment):
            """Fake Experiment to test job splitting"""

            def circuits(self):
                """Generate fake circuits"""
                qc = QuantumCircuit(1)
                qc.measure_all()
                return num_circuits * [qc]

        exp = Experiment([0])
        expdata = exp.run(backend)
        self.assertExperimentDone(expdata)
        job_ids = expdata.job_ids

        # Comptue expected number of jobs
        if max_experiments is None:
            num_jobs = 1
        else:
            num_jobs = num_circuits // max_experiments
            if num_circuits % max_experiments:
                num_jobs += 1
        self.assertEqual(len(job_ids), num_jobs)

    def test_analysis_replace_results_true(self):
        """Test running analysis with replace_results=True"""
        analysis = FakeAnalysis()
        expdata1 = analysis.run(ExperimentData(), seed=54321)
        self.assertExperimentDone(expdata1)
        result_ids = [res.result_id for res in expdata1.analysis_results()]
        expdata2 = analysis.run(expdata1, replace_results=True, seed=12345)
        self.assertExperimentDone(expdata2)

        self.assertEqual(expdata1, expdata2)
        self.assertEqual(expdata1.analysis_results(), expdata2.analysis_results())
        self.assertEqual(result_ids, list(expdata2._deleted_analysis_results))

    def test_analysis_replace_results_false(self):
        """Test running analysis with replace_results=False"""
        analysis = FakeAnalysis()
        expdata1 = analysis.run(ExperimentData(), seed=54321)
        self.assertExperimentDone(expdata1)
        expdata2 = analysis.run(expdata1, replace_results=False, seed=12345)
        self.assertExperimentDone(expdata2)

        self.assertNotEqual(expdata1, expdata2)
        self.assertNotEqual(expdata1.experiment_id, expdata2.experiment_id)
        self.assertNotEqual(expdata1.analysis_results(), expdata2.analysis_results())

    def test_analysis_config(self):
        """Test analysis config dataclass"""
        analysis = FakeAnalysis(arg1=10, arg2=20)
        analysis.set_options(option1=False, option2=True)
        config = analysis.config()
        loaded = config.analysis()
        self.assertEqual(analysis.config(), loaded.config())
        self.assertEqual(analysis.options, loaded.options)

    def test_analysis_from_config(self):
        """Test analysis config dataclass"""
        analysis = FakeAnalysis(arg1=10, arg2=20)
        analysis.set_options(option1=False, option2=True)
        config = analysis.config()
        loaded = FakeAnalysis.from_config(config)
        self.assertEqual(config, loaded.config())

    def test_analysis_from_dict_config(self):
        """Test analysis config dataclass for dict type."""
        analysis = FakeAnalysis(arg1=10, arg2=20)
        analysis.set_options(option1=False, option2=True)
        config = analysis.config()
        loaded = FakeAnalysis.from_config({"kwargs": config.kwargs, "options": config.options})
        self.assertEqual(config, loaded.config())

    def test_analysis_runtime_opts(self):
        """Test runtime options don't modify instance"""
        opts = {"opt1": False, "opt2": False}
        run_opts = {"opt1": True, "opt2": True, "opt3": True}
        analysis = FakeAnalysis()
        analysis.set_options(**opts)
        analysis.run(ExperimentData(), **run_opts)
        self.assertEqual(analysis.options.__dict__, opts)
