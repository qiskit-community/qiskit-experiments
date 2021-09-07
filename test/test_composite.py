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

"""Test suite for composite experiments.

This test assumes some simple but enough general virtual experiment.
The prepared fake experiment just flips qubit state and measure it,
while attached analysis class calculate the excited state population and create analysis result.

This experiment can cover the situation that deals with several hook methods and configurations.
This simplicity may benefit the debugging when some unexpected error is induced.
"""

from qiskit.circuit import QuantumCircuit
from qiskit.pulse import Schedule
from qiskit.test import QiskitTestCase
from qiskit.test.mock import FakeBogota
from qiskit.test.mock.utils import ConfigurableFakeBackend

from qiskit_experiments.framework import (
    BaseExperiment,
    BaseAnalysis,
    AnalysisResultData,
    ParallelExperiment,
    BatchExperiment,
    Options,
)


class FakeAnalysisProbability(BaseAnalysis):
    """A fake analysis that calculate probability from counts."""

    @classmethod
    def _default_options(cls) -> Options:
        return Options(dummyoption="default_value")

    def _run_analysis(self, experiment_data, **options):
        """Calculate probability."""

        expdata = experiment_data.data(0)
        counts = expdata["counts"]
        probability = counts.get("1", 0) / sum(counts.values())

        fake_data_entry = AnalysisResultData(
            name="probability",
            value=probability,
            extra={
                "run_options": options,
                "shots": sum(counts.values()),
            },
        )

        return [fake_data_entry], []


class FakeExperimentCommon(BaseExperiment):
    """A fake experiment that just flip qubit state and measure."""

    __analysis_class__ = FakeAnalysisProbability

    @classmethod
    def _default_experiment_options(cls):
        return Options(dummyoption="default_value")

    @classmethod
    def _default_transpile_options(cls):
        return Options(basis_gates=["x"])

    @classmethod
    def _default_run_options(cls):
        return Options(shots=1024)

    def circuits(self, backend=None):
        """Fake circuits."""
        test_circ = QuantumCircuit(1, 1)
        test_circ.x(0)
        test_circ.measure(0, 0)
        test_circ.metadata = {"dummy": "test_value"}

        return [test_circ]


class TestParallelExperiment(QiskitTestCase):
    """Test parallel experiment."""

    def test_standard_circuit_construction(self):
        """Test standard parallel experiment construction."""
        backend = ConfigurableFakeBackend("test", 2)
        exp0 = FakeExperimentCommon(qubits=[0])
        exp1 = FakeExperimentCommon(qubits=[1])

        par_exp = ParallelExperiment([exp0, exp1])
        test_circ = par_exp.run_transpile(backend=backend)[0]

        ref_circ = QuantumCircuit(*test_circ.qregs, *test_circ.cregs)
        ref_circ.x(0)
        ref_circ.x(1)
        ref_circ.measure(0, 0)
        ref_circ.measure(1, 1)

        self.assertEqual(test_circ, ref_circ)

    def test_pulse_gate_experiment_with_post_transpile_hook(self):
        """Test pulse gate parallel experiment with different transpile hook."""

        class FakeExperimentPulseGate(FakeExperimentCommon):
            """Add transpiler hook to insert calibration."""

            def _post_transpile_hook(self, circuits, backend):
                """Add pulse gate."""
                for circ in circuits:
                    circ.add_calibration(
                        "x", self.physical_qubits, Schedule(name="test_calibration")
                    )

                return circuits

        backend = ConfigurableFakeBackend("test", 2)
        exp0 = FakeExperimentCommon(qubits=[0])
        exp1 = FakeExperimentPulseGate(qubits=[1])

        par_exp = ParallelExperiment([exp0, exp1])
        test_circ = par_exp.run_transpile(backend=backend)[0]

        ref_circ = QuantumCircuit(*test_circ.qregs, *test_circ.cregs)
        ref_circ.x(0)
        ref_circ.x(1)
        ref_circ.measure(0, 0)
        ref_circ.measure(1, 1)

        # only q1 has pulse gate
        ref_circ.add_calibration("x", (1, ), Schedule(name="test_calibration"))

        self.assertEqual(test_circ, ref_circ)

    def test_update_circuit_metadata_with_post_transpile_hook(self):
        """Test new metadata with different transpile hook."""

        class FakeExperimentUpdateCircuitMetadata(FakeExperimentCommon):
            """Add transpiler hook to update metadata."""

            def _post_transpile_hook(self, circuits, backend):
                """Add pulse gate."""
                for circ in circuits:
                    circ.metadata["new_data"] = "test_value"

                return circuits

        backend = ConfigurableFakeBackend("test", 2)
        exp0 = FakeExperimentCommon(qubits=[0])
        exp1 = FakeExperimentUpdateCircuitMetadata(qubits=[1])

        par_exp = ParallelExperiment([exp0, exp1])
        test_circ = par_exp.run_transpile(backend=backend)[0]

        ref_metadata = [
            {"dummy": "test_value"},
            {"dummy": "test_value", "new_data": "test_value"},
        ]

        self.assertListEqual(test_circ.metadata["composite_metadata"], ref_metadata)

    def test_retain_transpile_configuration(self):
        """Test retain transpile configurations of nested experiments."""

        backend = ConfigurableFakeBackend("test", 2)
        exp0 = FakeExperimentCommon(qubits=[0])
        exp1 = FakeExperimentCommon(qubits=[1])

        # update exp1 basis gate
        exp1.set_transpile_options(basis_gates=["sx", "rz"])

        par_exp = ParallelExperiment([exp0, exp1])
        test_circ = par_exp.run_transpile(backend=backend)[0]

        ref_circ = QuantumCircuit(*test_circ.qregs, *test_circ.cregs)
        ref_circ.x(0)

        # q1 x is decomposed into two sx
        ref_circ.sx(1)
        ref_circ.sx(1)

        ref_circ.measure(0, 0)
        ref_circ.measure(1, 1)

        self.assertEqual(test_circ, ref_circ)

    def test_analyze_standard(self):
        """Test analyze standard parallel experiment result."""

        backend = FakeBogota()
        exp0 = FakeExperimentCommon(qubits=[0])
        exp1 = FakeExperimentCommon(qubits=[1])

        par_exp = ParallelExperiment([exp0, exp1])
        par_exp_data = par_exp.run(backend)
        par_exp_data.block_for_results()

        exp_data0 = par_exp_data.component_experiment_data(0)
        exp_data1 = par_exp_data.component_experiment_data(1)

        prob_entry_exp0 = exp_data0.analysis_results("probability")
        self.assertGreater(prob_entry_exp0.value, 0.8)

        prob_entry_exp1 = exp_data1.analysis_results("probability")
        self.assertGreater(prob_entry_exp1.value, 0.8)

    def test_retain_analysis_options(self):
        """Test retain analysis configurations of nested experiments."""

        backend = FakeBogota()
        exp0 = FakeExperimentCommon(qubits=[0])
        exp1 = FakeExperimentCommon(qubits=[1])

        # update exp1 analysis option
        exp1.set_analysis_options(new_config="test_value")

        par_exp = ParallelExperiment([exp0, exp1])
        par_exp_data = par_exp.run(backend)
        par_exp_data.block_for_results()

        exp_data0 = par_exp_data.component_experiment_data(0)
        prob_entry_exp0 = exp_data0.analysis_results("probability")
        ref_config0 = {"dummyoption": "default_value"}
        self.assertDictEqual(prob_entry_exp0.extra["run_options"], ref_config0)

        # this keeps updated analysis configuration
        exp_data1 = par_exp_data.component_experiment_data(1)
        prob_entry_exp1 = exp_data1.analysis_results("probability")
        ref_config1 = {"dummyoption": "default_value", "new_config": "test_value"}
        self.assertDictEqual(prob_entry_exp1.extra["run_options"], ref_config1)

    def test_run_option_overriden(self):
        """Test if run option is overridden."""

        backend = FakeBogota()
        exp0 = FakeExperimentCommon(qubits=[0])
        exp1 = FakeExperimentCommon(qubits=[1])

        # update exp1 run option
        exp1.set_run_options(shots=2048)

        par_exp = ParallelExperiment([exp0, exp1])
        par_exp_data = par_exp.run(backend, shots=1024)
        par_exp_data.block_for_results()

        exp_data1 = par_exp_data.component_experiment_data(1)
        prob_entry_exp1 = exp_data1.analysis_results("probability")
        self.assertEqual(prob_entry_exp1.extra["shots"], 1024)

    def test_analysis_hook(self):
        """Test update class variable with analysis result."""

        class FakeExperimentAnalysisHook(FakeExperimentCommon):

            def __init__(self, qubits):
                super().__init__(qubits)
                self.probability = None

            def _post_analysis_hook(self, experiment_data):
                """Extract probability and update instance variable."""
                prob_val = experiment_data.analysis_results("probability").value
                self.probability = prob_val

        backend = FakeBogota()
        exp0 = FakeExperimentCommon(qubits=[0])
        exp1 = FakeExperimentAnalysisHook(qubits=[1])

        par_exp = ParallelExperiment([exp0, exp1])
        par_exp_data = par_exp.run(backend, shots=1024)
        par_exp_data.block_for_results()

        exp_data1 = par_exp_data.component_experiment_data(1)
        prob_entry_exp1 = exp_data1.analysis_results("probability")

        self.assertEqual(exp1.probability, prob_entry_exp1.value)


class TestBatchExperiment(QiskitTestCase):
    """Test batch experiment.

    Note:
        The only difference of this from ``ParallelExperiment`` is the circuit construction.
        Thus the run and analysis tests can be omitted while keeping the good coverage.
    """

    def test_standard_circuit_construction(self):
        """Test standard batch experiment construction."""
        backend = ConfigurableFakeBackend("test", 2)
        exp0 = FakeExperimentCommon(qubits=[0])
        exp1 = FakeExperimentCommon(qubits=[1])

        par_exp = BatchExperiment([exp0, exp1])
        test_circs = par_exp.run_transpile(backend=backend)

        ref_circ0 = QuantumCircuit(2, 1)
        ref_circ0.x(0)
        ref_circ0.measure(0, 0)

        ref_circ1 = QuantumCircuit(2, 1)
        ref_circ1.x(1)
        ref_circ1.measure(1, 0)

        self.assertListEqual(test_circs, [ref_circ0, ref_circ1])
