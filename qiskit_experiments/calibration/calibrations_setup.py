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
A collections of helper methods to setup Calibrations.

Note that the set of available functions will be extended in future releases.
"""

from typing import Optional

from qiskit.circuit import Parameter
import qiskit.pulse as pulse
from qiskit.providers.backend import BackendV1 as Backend

from qiskit_experiments.calibration.backend_calibrations import BackendCalibrations


def standard_single_qubit_gates(
    backend: Backend,
    calibrations: Optional[BackendCalibrations] = None,
    link_amplitudes: bool = False,
    link_drag: bool = False,
    default_duration: int = 160,
    default_amplitude: float = 0.2,
    default_sigma: int = 40,
    default_beta: float = 0.0,
) -> BackendCalibrations:
    """Setup calibrations from a backend and populate them with single-qubit gates.

    This methods helps users setup an initial set of calibrations for single-qubit
    gates. Each qubit receives six single-qubit pulses, namely xp, xm, x90p, x90m,
    y90p, and y90m. Each pulse is a Drag pulse. All pulses share the same duration
    and sigma.

    Args:
        backend: The backend object for which to build the calibrations.
        calibrations: An optional calibrations instance to which the schedules and
            parameter values will be added. If this argument is not created then a
            BackendCalibrations instance will be initialized from the backend object.
        link_amplitudes: If set to True then the amplitudes of the x-minus pulses will
            be the negative of the amplitude of the x-plus pulses.
        link_drag: If set to True then all pulses will share the same Drag parameter.
        default_duration: The default duration for the single-qubit gates given as
            samples. This variable defaults to 160.
        default_amplitude: The default amplitude for the pulses. The default value is 0.2
            for the xp pulse which gives default amplitudes of -0.2, 0.1, and -0.1 for
            xm, x90p, and x90m, respectively.
        default_sigma: The default standard deviation of the pulses.
        default_beta: The default Drag parameter for all pulses.

    Returns:
        A BackendCalibration instance populate with the schedules and default parameter values.
    """

    if calibrations is None:
        calibrations = BackendCalibrations(backend)

    chan = Parameter("ch0")
    duration = Parameter("duration")
    sigma = Parameter("σ")
    amp_xp = Parameter("amp")
    amp_x90p = Parameter("amp")
    beta_xp = Parameter("β")

    if link_amplitudes:
        amp_xm = -amp_xp
        amp_x90m = -amp_x90p
        amp_y90p = 1.0j * amp_x90p
        amp_y90m = -1.0j * amp_x90p
    else:
        amp_xm = Parameter("amp")
        amp_x90m = Parameter("amp")
        amp_y90p = Parameter("amp")
        amp_y90m = Parameter("amp")

    if link_drag:
        beta_xm = beta_xp
        beta_x90p = beta_xp
        beta_x90m = beta_xp
        beta_y90p = beta_xp
        beta_y90m = beta_xp
    else:
        beta_xm = Parameter("β")
        beta_x90p = Parameter("β")
        beta_x90m = Parameter("β")
        beta_y90p = Parameter("β")
        beta_y90m = Parameter("β")

    pulse_config = [
        ("xp", amp_xp, beta_xp),
        ("xm", amp_xm, beta_xm),
        ("x90p", amp_x90p, beta_x90p),
        ("x90m", amp_x90m, beta_x90m),
        ("y90p", amp_y90p, beta_y90p),
        ("y90m", amp_y90m, beta_y90m),
    ]

    for name, amp, beta in pulse_config:
        with pulse.build(backend=backend, name=name) as schedule:
            pulse.play(
                pulse.Drag(duration=duration, sigma=sigma, amp=amp, beta=beta),
                pulse.DriveChannel(chan),
            )

        calibrations.add_schedule(schedule)

    betas = [
        ("xp", beta_xp),
        ("xm", beta_xm),
        ("x90p", beta_x90p),
        ("x90m", beta_x90m),
        ("y90p", beta_y90p),
        ("y90m", beta_y90m),
    ]

    amps = [
        ("xp", 1, amp_xp),
        ("xm", -1, amp_xm),
        ("x90p", 0.5, amp_x90p),
        ("x90m", -0.5, amp_x90m),
        ("y90p", 0.5j, amp_y90p),
        ("y90m", -0.5j, amp_y90m),
    ]

    # Register the default parameter values common to all qubits.
    for sched_name in ["xp", "xm", "x90p", "x90m"]:
        calibrations.add_parameter_value(default_duration, duration, schedule=sched_name)
        calibrations.add_parameter_value(default_sigma, sigma, schedule=sched_name)

    # Register parameter values for amplitude and beta for each qubit.
    for qubit in range(backend.configuration().n_qubits):
        for sched_name, param in betas:
            calibrations.add_parameter_value(default_beta, param, qubit, sched_name)

        for sched_name, sign, param in amps:
            calibrations.add_parameter_value(sign * default_amplitude, param, qubit, sched_name)

    return calibrations
