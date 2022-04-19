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

from typing import List, Optional, Union

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
from qiskit_experiments.library.calibration import RoughAmplitudeCal, RoughDragCal
from qiskit_experiments.test.mock_iq_backend import RabiBackend, DragBackend
from qiskit_experiments.exceptions import AnalysisError


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

    def test_single_experiment(self):
        """Test the chained experiment with a single experiment."""

        rabi = RoughAmplitudeCal(0, self.cals, backend=RabiBackend())

        cal_chain = ChainedExperiment([rabi])

        exp_data = cal_chain.run()

        self.assertEqual(len(exp_data.child_data()), 1)

    def test_watchdog(self):
        """Test that we do not get endless loops on ill-defined transition functions."""

        exp = DummyExperiment((0, ))
        exp_chain = ChainedExperiment([exp], DefectiveTransitionCallable())

        with self.assertRaisesRegex(AnalysisError, msg=""):
            exp_chain.run()

    def test_simple_cal_chain(self):
        """Test a simple calibration chain made of a Rabi and Drag.

        The parameters in the Calibrations should be properly updated if the chain is successful.
        """
        rabi = RoughAmplitudeCal(0, self.cals, backend=RabiBackend())
        drag = RoughDragCal(0, self.cals, backend=DragBackend())

        cal_chain = ChainedExperiment([rabi, drag])

        exp_data = cal_chain.run()

        self.assertEqual(len(exp_data.child_data()), 2)

        # Run some tests on the cals.
