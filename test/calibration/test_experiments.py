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

"""Test the rough amplitude experiment."""

import qiskit.pulse as pulse
from qiskit.test import QiskitTestCase
from qiskit.pulse import Drag, DriveChannel
from qiskit.test.mock import FakeAlmaden
from qiskit.circuit import Parameter, Gate
from qiskit_experiments.calibration import CalibrationsDefinition
from qiskit_experiments.calibration import ParameterValue
from qiskit_experiments.calibration import RoughAmplitude
from qiskit_experiments.calibration import DataProcessor


class TestCalibrationExperiments(QiskitTestCase):

    """Class to test calibration experiments."""
    def setUp(self):
        """Setup variables used for testing."""
        self.backend = FakeAlmaden()
        self.cals = CalibrationsDefinition(self.backend)

        sigma_xp = Parameter('σ_xp')
        d0 = Parameter('0')
        amp_xp = Parameter('amp_xp')
        amp_x90p = Parameter('amp_x90p')
        amp_y90p = Parameter('amp_y90p')
        beta_xp = Parameter('β_xp')

        with pulse.build(name='xp') as xp:
            pulse.play(Drag(160, amp_xp, sigma_xp, beta_xp), DriveChannel(d0))

        with pulse.build(name='xm') as xm:
            pulse.play(Drag(160, -amp_xp, sigma_xp, beta_xp), DriveChannel(d0))

        with pulse.build(name='x90p') as x90p:
            pulse.play(Drag(160, amp_x90p, sigma_xp, beta_xp), DriveChannel(d0))

        with pulse.build(name='y90p') as y90p:
            pulse.play(Drag(160, amp_y90p, sigma_xp, beta_xp), DriveChannel(d0))

        self.cals.add_schedules([xp, x90p, y90p, xm])

        self.cals.add_parameter_value('σ_xp', ParameterValue(40), ch_type=DriveChannel)
        self.cals.add_parameter_value('amp_xp', ParameterValue(0.2), ch_type=DriveChannel)
        self.cals.add_parameter_value('amp_x90p', ParameterValue(0.1), ch_type=DriveChannel)
        self.cals.add_parameter_value('amp_y90p', ParameterValue(0.1j), ch_type=DriveChannel)
        self.cals.add_parameter_value('β_xp', ParameterValue(0.0), ch_type=DriveChannel)

    def test_rough_amplitude(self):
        """The the rough amplitude calibration."""

        qubit = 3
        amps = [-0.5, 0.5]
        amp = RoughAmplitude(qubit, self.cals, 'xp', 'amp_xp', DataProcessor())
        circs = amp.transpiled_circuits(self.backend, amplitudes=amps)

        # Check that there is a gate on qubit 3.
        self.assertEqual(circs[1].data[0][1][0].index, qubit)
        self.assertTrue(isinstance(circs[1].data[0][0], Gate))

        # Check that we have calibrations for the xp gate on qubit 3
        for idx, amp in enumerate(amps):
            self.assertTrue(((qubit, ), (amp, )) in circs[idx].calibrations['xp'])
