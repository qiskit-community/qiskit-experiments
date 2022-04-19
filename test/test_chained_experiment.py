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

"""Class to test chained experiments."""

import copy
from typing import List, Optional, Union
import numpy as np

from qiskit.circuit import QuantumCircuit
from qiskit.test.mock import FakeBelem
from qiskit.providers import Backend

from test.base import QiskitExperimentsTestCase

from qiskit_experiments.framework.composite.chained_experiment import (
    ChainedExperiment,
    BaseTransitionCallable,
)
from qiskit_experiments.framework.base_experiment import BaseExperiment, ExperimentData
from qiskit_experiments.framework.base_analysis import BaseAnalysis
from qiskit_experiments.calibration_management.calibrations import Calibrations
from qiskit_experiments.calibration_management.basis_gate_library import FixedFrequencyTransmon
from qiskit_experiments.library.calibration import FineXAmplitudeCal, RoughDragCal
from qiskit_experiments.test.mock_iq_backend import MockFineAmp, DragBackend
from qiskit_experiments.exceptions import AnalysisError


class FineXAmplitudeCalTest(FineXAmplitudeCal):
    """A class so that we can access the inst_map that was used."""

    def _transpiled_circuits(self) -> List[QuantumCircuit]:
        """Wrap the transpile step to save the inst map so we can test it."""
        self.inst_map = copy.deepcopy(self.calibrations.default_inst_map)
        return super()._transpiled_circuits()


class DefectiveTransitionCallable(BaseTransitionCallable):
    """A defective callable that does not transition."""

    def __call__(self, *args, **kwargs):
        """The pointer index is never incremented."""
        return 0


class DummyExperiment(BaseExperiment):
    """A dummy experiment."""

    def run(
        self,
        backend: Optional[Backend] = None,
        analysis: Optional[Union[BaseAnalysis, None]] = "default",
        timeout: Optional[float] = None,
        **run_options,
    ) -> ExperimentData:
        """Returns an empty experiment data instance."""
        return ExperimentData()

    def circuits(self) -> List[QuantumCircuit]:
        """Return an empty list."""
        return []


class TestChained(QiskitExperimentsTestCase):
    """Test the chained experiments."""

    def setUp(self):
        """Setup the test."""
        super().setUp()

        self.cals = Calibrations.from_backend(backend=FakeBelem(), library=FixedFrequencyTransmon())

        self.drag_backend = DragBackend(gate_name="Drag(x)")
        self.amp_backend = MockFineAmp(0.05 * np.pi, np.pi, "x")

    def test_single_experiment(self):
        """Test the chained experiment with a single experiment."""

        drag = RoughDragCal(0, self.cals, backend=self.drag_backend)

        cal_chain = ChainedExperiment([drag])

        self.assertEqual(len(self.cals.parameters_table(parameters=["β"])["data"]), 2)

        exp_data = cal_chain.run()

        self.assertEqual(len(exp_data.child_data()), 1)

        self.assertExperimentDone(exp_data)
        self.assertEqual(len(self.cals.parameters_table(parameters=["β"])["data"]), 3)

    def test_watchdog(self):
        """Test that we do not get endless loops on ill-defined transition functions."""

        exp = DummyExperiment((0, ), backend=DragBackend(gate_name="Drag(x)"))
        exp_chain = ChainedExperiment([exp], DefectiveTransitionCallable())
        exp_chain.analysis = None

        with self.assertRaisesRegex(AnalysisError, expected_regex=""):
            exp_chain.run()

    def test_simple_cal_chain(self):
        """Test a simple calibration chain made of a Rabi and Drag.

        The parameters in the Calibrations should be properly updated if the chain is successful.
        """
        drag = RoughDragCal(0, self.cals, backend=self.drag_backend)
        fine_amp = FineXAmplitudeCalTest(0, self.cals, backend=self.amp_backend, schedule_name="x")

        cal_chain = ChainedExperiment([drag, fine_amp])

        # Sanity checks to ensure future checks.
        old_beta = self.cals.get_parameter_value("β", 0, "x")
        self.assertAlmostEqual(old_beta, 0.0, places=1)
        self.assertEqual(len(self.cals.parameters_table(parameters=["β"])["data"]), 2)
        self.assertEqual(len(self.cals.parameters_table(parameters=["amp"])["data"]), 2)

        exp_data = cal_chain.run()

        # We can only start checking the cals once the chain has run.
        self.assertExperimentDone(exp_data)

        self.assertEqual(len(exp_data.child_data()), 2)

        # Check the update of the drag experiment.
        new_beta = self.cals.get_parameter_value("β", 0, "x")
        self.assertAlmostEqual(new_beta, self.drag_backend.ideal_beta, places=1)
        self.assertEqual(len(self.cals.parameters_table(parameters=["β"])["data"]), 3)

        # Check that the fine amplitude experiment ran with the updated beta
        x_sched = fine_amp.inst_map.get("x", qubits=(0, ))
        self.assertAlmostEqual(x_sched.blocks[0].pulse.beta, new_beta, places=7)
        self.assertEqual(len(self.cals.parameters_table(parameters=["amp"])["data"]), 3)

        new_amp = self.cals.get_parameter_value("amp", 0, "x")
        expected_amp = 0.5 * np.pi / (np.pi + exp_data.child_data(1).analysis_results(1).value.n)
        self.assertAlmostEqual(new_amp, expected_amp, places=7)

        # Check the experiments that were run.
        for idx, expected in enumerate([RoughDragCal, FineXAmplitudeCalTest]):
            self.assertEqual(type(exp_data.child_data(idx).experiment), expected)

    def test_transition_callback(self):
        """Test that we can change the behaviour of the transition callback with options."""