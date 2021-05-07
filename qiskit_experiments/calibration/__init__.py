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

r"""
Qiskit Experiments Calibration Root.

Calibrations are managed by the Calibrations class. This class stores schedules which are
intended to be fully parameterized, including the index of the channels. This class:
- supports having different schedules share parameters
- allows default schedules for qubits that can be overridden for specific qubits.

Parametric channel naming convention
=========================

Parametrized channel indices must be named according to a predefined pattern to properly
identify the channels and control channels when assigning values to the parametric
channel indices. A channel must have a name that starts with `ch` followed by an integer.
For control channels this integer can be followed by a sequence `.integer`.
Optionally, the name can end with `$integer` to specify the index of a control channel
for the case when a set of qubits share multiple control channels. For example,
valid channel names include "ch0", "ch1", "ch0.1", "ch0$", "ch2$3", and "ch1.0.3$2".
The "." delimiter is used to specify the different qubits when looking for control
channels. The optional $ delimiter is used to specify which control channel to use
if several control channels work together on the same qubits. For example, if the
control channel configuration is {(3,2): [ControlChannel(3), ControlChannel(12)]}
then given qubits (2, 3) the name "ch1.0$1" will resolve to ControlChannel(12) while
"ch1.0$0" will resolve to ControlChannel(3). A channel can only have one parameter.

Parameter naming restriction
===================

Each parameter must have a unique name within each schedule. For example, it is
acceptable to have a parameter named 'amp' in the schedule 'xp' and a different
parameter instance named 'amp' in the schedule named 'xm'. It is not acceptable
to have two parameters named 'amp' in the same schedule. The naming restriction
only applies to parameters used in the immediate scope of the schedule. Schedules
called by Call instructions have their own scope for Parameter names.

The code block below illustrates the creation of a template schedule for a echoed cross-
resonance gate.

.. code-block:: python

    amp_cr = Parameter("amp")
    amp = Parameter("amp")
    d0 = DriveChannel(Parameter("ch0"))
    c1 = ControlChannel(Parameter("ch0.1"))
    sigma = Parameter("σ")
    width = Parameter("w")
    dur_xp = Parameter("duration")
    dur_cr = Parameter("duration")

    with pulse.build(name="xp") as xp:
        pulse.play(Gaussian(dur_xp, amp, sigma), d0)

    with pulse.build(name="cr") as cr:
        with pulse.align_sequential():
                pulse.play(GaussianSquare(dur_cr, amp_cr, sigma, width), c1)
                pulse.call(xp)
                pulse.play(GaussianSquare(dur_cr, -amp_cr, sigma, width), c1)
                pulse.call(xp)
"""

from .calibrations import Calibrations
from .backend_calibrations import BackendCalibrations
from .exceptions import CalibrationError
from .parameter_value import ParameterValue
