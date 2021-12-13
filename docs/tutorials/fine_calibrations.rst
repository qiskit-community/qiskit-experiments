Fine Calibrations
=================

The amplitude of a pulse can be precisely calibrated using error amplifying gate sequences. These gate sequences apply the same gate a variable number of times. Therefore, if each gate has a small error :math:`\delta\theta` in the rotation angle then a sequence of :math:`n` gates will have a rotation error of :math:`n\cdot\delta\theta`. We will work with ``ibmq_armonk`` and compare our results to those reported by the backend.

.. jupyter-execute::

    import numpy as np

    from qiskit import IBMQ
    from qiskit.pulse import InstructionScheduleMap
    import qiskit.pulse as pulse

    from qiskit_experiments.library import FineXAmplitude, FineSXAmplitude

.. jupyter-execute::

    IBMQ.load_account()
    provider = IBMQ.get_provider(hub='ibm-q', group='open', project='main')
    backend = provider.get_backend('ibmq_armonk')

.. jupyter-execute::

    qubit = 0

Instruction schedule map
^^^^^^^^^^^^^^^^^^^^^^^^

We will run the fine calibration experiments with our own pulse schedules. To do this we create an instruction to schedule map which we populate with the schedules we wish to work with. This instruction schedule map is then given to the transpile options of the calibration experiments so that the Qiskit transpiler can attach the pulse schedules to the gates in the experiments. We will base all our pulses on the default ``X`` pulse of Armonk.

.. jupyter-execute::

    x_pulse = backend.defaults().instruction_schedule_map.get('x', (qubit,)).instructions[0][1].pulse
    x_pulse

.. jupyter-execute::

    # create the schedules we need and add them to an instruction schedule map.
    sx_pulse = pulse.Drag(x_pulse.duration, 0.5*x_pulse.amp, x_pulse.sigma, x_pulse.beta, name="SXp_d0")
    y_pulse = pulse.Drag(x_pulse.duration, 1.0j*x_pulse.amp, x_pulse.sigma, x_pulse.beta, name="Yp_d0")

    d0, inst_map = pulse.DriveChannel(qubit), InstructionScheduleMap()

    for name, pulse_ in [("x", x_pulse), ("y", y_pulse), ("sx", sx_pulse)]:
        with pulse.build(name=name) as sched:
            pulse.play(pulse_, d0)

        inst_map.add(name, (qubit,), sched)

Fine Amplitude Calibration
--------------------------

.. jupyter-execute::

    ideal_amp = x_pulse.amp
    print(f"The reported amplitude of the X pulse is {ideal_amp:.4f}.")

Detecting an over-rotated pulse
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

We now take the x pulse reported by the backend and add a 2% overrotation to it by scaling the amplitude and see if the experiment can detect this overrotation. We replace the default ``X`` pulse in the instruction schedule map with this overrotated pulse.

.. jupyter-execute::

    pulse_amp = ideal_amp*1.02
    target_angle = np.pi

    with pulse.build(backend=backend, name="x") as x_over:
        pulse.play(pulse.Drag(x_pulse.duration, pulse_amp, x_pulse.sigma, x_pulse.beta), d0)

    inst_map.add("x", (qubit,), x_over)

.. jupyter-execute::

    amp_cal = FineXAmplitude(qubit, backend=backend)
    amp_cal.set_transpile_options(inst_map=inst_map)

Observe here that we added a square-root of X pulse before appyling the error amplifying sequence. This is done to be able to distinguish between over-rotated and under-rotated pulses.

.. jupyter-execute::

    amp_cal.circuits()[5].draw(output="mpl")

.. jupyter-execute::

    data_over = amp_cal.run().block_for_results()

.. jupyter-execute::

    data_over.figure(0)

.. jupyter-execute::

    print(data_over.analysis_results("d_theta"))

.. jupyter-execute::

    dtheta = data_over.analysis_results("d_theta").value.value
    scale = target_angle / (target_angle + dtheta)
    print(f"The ideal angle is {target_angle:.2f} rad. We measured a deviation of {dtheta:.3f} rad.")
    print(f"Thus, scale the {pulse_amp:.4f} pulse amplitude by {scale:.3f} to obtain {pulse_amp*scale:.5f}.")
    print(f"Amplitude reported by the backend {ideal_amp:.4f}.")

Detecting an under-rotated pulse
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. jupyter-execute::

    pulse_amp = ideal_amp*0.98
    target_angle = np.pi

    with pulse.build(backend=backend, name="xp") as x_under:
        pulse.play(pulse.Drag(x_pulse.duration, pulse_amp, x_pulse.sigma, x_pulse.beta), d0)

    inst_map.add("x", (qubit,), x_under)

.. jupyter-execute::

    amp_cal = FineXAmplitude(qubit, backend=backend)
    amp_cal.set_transpile_options(inst_map=inst_map)

.. jupyter-execute::

    data_under = amp_cal.run().block_for_results()

.. jupyter-execute::

    data_under.figure(0)

.. jupyter-execute::

    print(data_under.analysis_results("d_theta"))

.. jupyter-execute::

    dtheta = data_under.analysis_results("d_theta").value.value
    scale = target_angle / (target_angle + dtheta)
    print(f"The ideal angle is {target_angle:.2f} rad. We measured a deviation of {dtheta:.3f} rad.")
    print(f"Thus, scale the {pulse_amp:.4f} pulse amplitude by {scale:.3f} to obtain {pulse_amp*scale:.5f}.")
    print(f"Amplitude reported by the backend {ideal_amp:.4f}.")

Analyzing a :math:`\frac{\pi}{2}` pulse
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

We now consider the :math:`\frac{\pi}{2}` rotation. Note that in this case we do not need to add a :math:`\frac{\pi}{2}` rotation to the circuits.

.. jupyter-execute::

    # restor the x_pulse
    inst_map.add("x", (qubit,), backend.defaults().instruction_schedule_map.get('x', (qubit,)))

.. jupyter-execute::

    amp_cal = FineSXAmplitude(qubit, backend)
    amp_cal.set_transpile_options(inst_map=inst_map)

.. jupyter-execute::

    amp_cal.circuits()[5].draw(output="mpl")

.. jupyter-execute::

    data_x90p = amp_cal.run().block_for_results()

.. jupyter-execute::

    data_x90p.figure(0)

.. jupyter-execute::

    print(data_x90p.analysis_results("d_theta"))

.. jupyter-execute::

    sx = backend.defaults().instruction_schedule_map.get('sx', (qubit,))
    sx_ideal_amp = sx.instructions[0][1].pulse.amp

    target_angle = np.pi / 2
    dtheta = data_x90p.analysis_results("d_theta").value.value
    scale = target_angle / (target_angle + dtheta)
    print(f"The ideal angle is {target_angle:.2f} rad. We measured a deviation of {dtheta:.3f} rad.")
    print(f"Thus, scale the {sx_pulse.amp:.4f} pulse amplitude by {scale:.3f} to obtain {sx_pulse.amp*scale:.5f}.")
    print(f"Amplitude reported by the backend {sx_ideal_amp:.4f}.")

Let's rerun this calibration using the updated value of the amplitude of the :math:`\frac{\pi}{2}` pulse.

.. jupyter-execute::

    pulse_amp = sx_pulse.amp*scale

    with pulse.build(backend=backend, name="sx") as sx_new:
        pulse.play(pulse.Drag(x_pulse.duration, pulse_amp, x_pulse.sigma, x_pulse.beta), d0)

    inst_map.add("sx", (qubit,), sx_new)

.. jupyter-execute::

    data_x90p = amp_cal.run().block_for_results()

.. jupyter-execute::

    data_x90p.figure(0)

.. jupyter-execute::

    print(data_x90p.analysis_results("d_theta"))

.. jupyter-execute::

    dtheta = data_x90p.analysis_results("d_theta").value.value
    scale = target_angle / (target_angle + dtheta)
    print(f"The ideal angle is {target_angle:.2f} rad. We measured a deviation of {dtheta:.3f} rad.")
    print(f"Thus, scale the {pulse_amp:.4f} pulse amplitude by {scale:.3f} to obtain {pulse_amp*scale:.5f}.")
    print(f"Amplitude reported by the backend {sx_ideal_amp:.4f}.")

Fine DRAG Calibrations
----------------------

.. jupyter-execute::

    from qiskit_experiments.library import FineXDrag

.. jupyter-execute::

    ideal_beta = x_pulse.beta
    print(f"The reported beta of the X pulse is {ideal_beta:.4f}.")

.. jupyter-execute::

    pulse_beta = ideal_beta*1.25
    target_angle = np.pi

    with pulse.build(backend=backend, name="x") as x_over:
        pulse.play(pulse.Drag(x_pulse.duration, x_pulse.amp, x_pulse.sigma, pulse_beta), d0)

    inst_map.add("x", (qubit,), x_over)

.. jupyter-execute::

    drag_cal = FineXDrag(qubit, backend)
    drag_cal.set_transpile_options(inst_map=inst_map)

.. jupyter-execute::

    drag_cal.circuits()[2].draw("mpl")

.. jupyter-execute::

    data_drag_x = drag_cal.run().block_for_results()

.. jupyter-execute::

    data_drag_x.figure(0)

.. jupyter-execute::

    print(data_drag_x.analysis_results(0))

.. jupyter-execute::

    data_drag_x.analysis_results("d_theta").value.value

.. jupyter-execute::

    dtheta = data_drag_x.analysis_results("d_theta").value.value

    ddelta = -0.25 * np.sqrt(np.pi) * dtheta * x_pulse.sigma / ((target_angle**2) / 4)

    print(f"Adjust β={pulse_beta:.3f} by ddelta={ddelta:.3f} to get {ddelta + pulse_beta:.3f} as new β.")
    print(f"The backend reports β={x_pulse.beta:.3f}")

Half angle calibrations
-----------------------

Phase errors imply that it is possible for the ``sx`` and ``x`` pulse to be misaligned. This can occure, for example, due to non-linearities in the mixer skew. The half angle experiment allows us to measure such issues.

.. jupyter-execute::

    from qiskit_experiments.library import HalfAngle

.. jupyter-execute::

    hac = HalfAngle(qubit, backend)
    hac.set_transpile_options(inst_map=inst_map)

.. jupyter-execute::

    hac.circuits()[5].draw("mpl")

.. jupyter-execute::

    exp_data = hac.run().block_for_results()

.. jupyter-execute::

    exp_data.figure(0)

.. jupyter-execute::

    print(exp_data.analysis_results(0))

.. jupyter-execute::

    dhac = exp_data.analysis_results("d_hac").value.value

.. jupyter-execute::

    sx = backend.defaults().instruction_schedule_map.get('sx', (qubit,))
    sx_amp = sx.instructions[0][1].pulse.amp

    print(f"Adjust the phase of {np.angle(sx_pulse.amp)} of the sx pulse by {-dhac/2:.3f} rad.")
    print(f"The backend reports an angle of {np.angle(sx_amp):.3f} for the sx pulse.")

.. jupyter-execute::

    import qiskit.tools.jupyter
    %qiskit_copyright