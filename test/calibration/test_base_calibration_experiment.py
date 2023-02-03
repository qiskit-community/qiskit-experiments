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

"""Tests for the base class for calibration-type experiments."""

from test.base import QiskitExperimentsTestCase

from qiskit.circuit import Parameter
from qiskit.pulse import Play, Constant, DriveChannel, ScheduleBlock

from qiskit_experiments.library import QubitSpectroscopy
from qiskit_experiments.calibration_management.base_calibration_experiment import (
    BaseCalibrationExperiment,
    Calibrations,
)
from qiskit_experiments.framework.composite import ParallelExperiment, BatchExperiment
from qiskit_experiments.test.fake_backend import FakeBackend
from .utils import MockCalExperiment, DoNothingAnalysis


class TestBaseCalibrationClass(QiskitExperimentsTestCase):
    """Tests for base calibration experiment classes."""

    def test_class_order(self):
        """Test warnings when the BaseCalibrationExperiment is not the first parent."""

        class CorrectOrder(BaseCalibrationExperiment, QubitSpectroscopy):
            """A class with the correct order should not produce warnings.."""

            def __init__(self):
                """A dummy class for parent order testing."""
                super().__init__(Calibrations(coupling_map=[]), 0, [0, 1, 2])

            def _attach_calibrations(self, circuit):
                """Needed as this method is abstract"""
                pass

        CorrectOrder()

        with self.assertWarns(Warning):

            # pylint: disable=unused-variable
            class WrongOrder(QubitSpectroscopy, BaseCalibrationExperiment):
                """Merely defining this class is enough to raise the warning."""

                def __init__(self):
                    """A dummy class for parent order testing."""
                    super().__init__(Calibrations(coupling_map=[]), 0, [0, 1, 2])

    def test_update_calibration(self):
        """Test updating calibrations with execution of calibration experiment."""
        backend = FakeBackend()
        ref_new_value = 0.3

        param = Parameter("to_calibrate")
        schedule = ScheduleBlock(name="test")
        schedule.append(Play(Constant(100, param), DriveChannel(0)), inplace=True)
        cals = Calibrations()
        cals.add_schedule(schedule, 0, 1)

        exp = MockCalExperiment(
            qubits=(0,),
            calibrations=cals,
            new_value=ref_new_value,
            param_name="to_calibrate",
            sched_name="test",
        )
        exp.run(backend).block_for_results()

        new_value = cals.get_parameter_value("to_calibrate", (0,), "test")
        self.assertEqual(new_value, ref_new_value)

        new_schedule = cals.get_schedule("test", (0,))
        ref_schedule = schedule.assign_parameters({param: ref_new_value}, inplace=False)
        self.assertEqual(new_schedule, ref_schedule)

    def test_update_calibration_update_analysis(self):
        """Test updating calibrations with experiment with updated analysis option."""
        backend = FakeBackend()
        ref_new_value = 0.3

        param = Parameter("to_calibrate")
        schedule = ScheduleBlock(name="test")
        schedule.append(Play(Constant(100, param), DriveChannel(0)), inplace=True)
        cals = Calibrations()
        cals.add_schedule(schedule, 0, 1)

        exp = MockCalExperiment(
            qubits=(0,),
            calibrations=cals,
            new_value=999999,
            param_name="to_calibrate",
            sched_name="test",
        )
        exp.analysis.set_options(return_value=ref_new_value)
        exp.run(backend).block_for_results()

        new_value = cals.get_parameter_value("to_calibrate", (0,), "test")
        self.assertEqual(new_value, ref_new_value)

        new_schedule = cals.get_schedule("test", (0,))
        ref_schedule = schedule.assign_parameters({param: ref_new_value}, inplace=False)
        self.assertEqual(new_schedule, ref_schedule)

    def test_update_calibration_custom_analysis(self):
        """Test updating calibrations with experiment instance with user analysis."""
        backend = FakeBackend()
        ref_new_value = 0.3

        param = Parameter("to_calibrate")
        schedule = ScheduleBlock(name="test")
        schedule.append(Play(Constant(100, param), DriveChannel(0)), inplace=True)
        cals = Calibrations()
        cals.add_schedule(schedule, 0, 1)

        exp = MockCalExperiment(
            qubits=(0,),
            calibrations=cals,
            new_value=99999,
            param_name="to_calibrate",
            sched_name="test",
        )

        user_analysis = DoNothingAnalysis()
        user_analysis.set_options(return_value=ref_new_value)
        exp.analysis = user_analysis
        exp.run(backend).block_for_results()

        new_value = cals.get_parameter_value("to_calibrate", (0,), "test")
        self.assertEqual(new_value, ref_new_value)

        new_schedule = cals.get_schedule("test", (0,))
        ref_schedule = schedule.assign_parameters({param: ref_new_value}, inplace=False)
        self.assertEqual(new_schedule, ref_schedule)

    def test_update_calibration_batch(self):
        """Test updating calibrations from batch experiment."""
        backend = FakeBackend()
        ref_new_value1 = 100
        ref_new_value2 = 0.4

        param1 = Parameter("to_calibrate1")
        param2 = Parameter("to_calibrate2")
        schedule = ScheduleBlock(name="test")
        schedule.append(Play(Constant(param1, param2), DriveChannel(0)), inplace=True)
        cals = Calibrations()
        cals.add_schedule(schedule, 0, 1)

        exp1 = MockCalExperiment(
            qubits=(0,),
            calibrations=cals,
            new_value=ref_new_value1,
            param_name="to_calibrate1",
            sched_name="test",
        )
        exp2 = MockCalExperiment(
            qubits=(0,),
            calibrations=cals,
            new_value=ref_new_value2,
            param_name="to_calibrate2",
            sched_name="test",
        )
        batch_exp = BatchExperiment([exp1, exp2], backend=backend)
        batch_exp.run(backend).block_for_results()

        new_value1 = cals.get_parameter_value("to_calibrate1", (0,), "test")
        self.assertEqual(new_value1, ref_new_value1)

        new_value2 = cals.get_parameter_value("to_calibrate2", (0,), "test")
        self.assertEqual(new_value2, ref_new_value2)

        new_schedule = cals.get_schedule("test", (0,))
        ref_schedule = schedule.assign_parameters(
            {
                param1: ref_new_value1,
                param2: ref_new_value2,
            },
            inplace=False,
        )
        self.assertEqual(new_schedule, ref_schedule)

    def test_update_calibration_parallel(self):
        """Test updating calibrations from parallel experiment."""
        backend = FakeBackend()
        ref_new_value1 = 0.3
        ref_new_value2 = 0.4

        param1 = Parameter("to_calibrate1")
        param2 = Parameter("to_calibrate2")
        schedule1 = ScheduleBlock(name="test1")
        schedule1.append(Play(Constant(100, param1), DriveChannel(0)), inplace=True)
        schedule2 = ScheduleBlock(name="test2")
        schedule2.append(Play(Constant(100, param2), DriveChannel(1)), inplace=True)
        cals = Calibrations()
        cals.add_schedule(schedule1, 0, 1)
        cals.add_schedule(schedule2, 1, 1)

        exp1 = MockCalExperiment(
            qubits=(0,),
            calibrations=cals,
            new_value=ref_new_value1,
            param_name="to_calibrate1",
            sched_name="test1",
        )
        exp2 = MockCalExperiment(
            qubits=(1,),
            calibrations=cals,
            new_value=ref_new_value2,
            param_name="to_calibrate2",
            sched_name="test2",
        )
        batch_exp = ParallelExperiment([exp1, exp2], backend=backend)
        batch_exp.run(backend).block_for_results()

        new_value1 = cals.get_parameter_value("to_calibrate1", (0,), "test1")
        self.assertEqual(new_value1, ref_new_value1)

        new_value2 = cals.get_parameter_value("to_calibrate2", (1,), "test2")
        self.assertEqual(new_value2, ref_new_value2)

        new_schedule1 = cals.get_schedule("test1", (0,))
        ref_schedule1 = schedule1.assign_parameters({param1: ref_new_value1}, inplace=False)
        self.assertEqual(new_schedule1, ref_schedule1)

        new_schedule2 = cals.get_schedule("test2", (1,))
        ref_schedule2 = schedule2.assign_parameters({param2: ref_new_value2}, inplace=False)
        self.assertEqual(new_schedule2, ref_schedule2)
