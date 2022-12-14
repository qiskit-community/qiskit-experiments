================================================
Fine Calibrations of a pulse amplitude
================================================
Calibrating quantum gates is the task of finding the parameters of the underlying pulses that best implement the target gate.
The amplitude of a pulse can be precisely calibrated using
error amplifying gate sequences. These gate sequences apply 
the same gate a variable number of times. Therefore, if each gate
has a small error :math:`d\theta` in the rotation angle then 
a sequence of :math:`n` gates will have a rotation error of :math:`n` * :math:`d\theta`.

We will illustrate how the `FineXAmplitude` experiment works with the `PulseBackend`, 
i.e., a backend that simulates the underlying pulses with Qiskit Dynamics 
on a three-level model of a transmon. This simulator backend 
can be replaced with a standard hardware IBM Quantum backend.

.. jupyter-execute:: 

    import numpy as np
    from qiskit.pulse import InstructionScheduleMap
    import qiskit.pulse as pulse
    from qiskit_experiments.library import FineXAmplitude, FineSXAmplitude
    from qiskit_experiments.test.pulse_backend import SingleTransmonTestBackend

.. jupyter-execute::

    backend = SingleTransmonTestBackend()
    qubit = 0
-----------------------------------------------------
Fine X gate Amplitude Calibration
-----------------------------------------------------
We will run the error amplifying experiments with our own pulse schedules
on which we purposefully add over and under rotations.
To do this we create an instruction to schedule map which we populate with 
the schedules we wish to work with. This instruction schedule map is then 
given to the transpile options of the experiment so that 
the Qiskit transpiler can attach the pulse schedules to the gates in the experiments. 
We base all our pulses on the default X pulse of "SingleTransmonTestBackend".

.. jupyter-execute::

    x_pulse = backend.defaults().instruction_schedule_map.get('x', (qubit,)).instructions[0][1].pulse
    d0, inst_map = pulse.DriveChannel(qubit), pulse.InstructionScheduleMap()


We now take the ideal x pulse amplitude reported by the backend and 
add/subtract a 2% over/underrotation to it by scaling the ideal amplitude and see 
if the experiment can detect this over/underrotation. We replace the default X pulse 
in the instruction schedule map with this over/underrotated pulse.

.. jupyter-execute::

    ideal_amp = x_pulse.amp
    over_amp = ideal_amp*1.02
    under_amp = ideal_amp*0.98
    print(f"The reported amplitude of the X pulse is {ideal_amp:.4f} which we set as ideal_amp.") 
    print(f"we use {over_amp:.4f} amplitude for overroation pulse and {under_amp:.4f} for underrotation pulse.")
    # build the over rotated pulse and add it to the instruction schedule map
    with pulse.build(backend=backend, name="x") as x_over:
        pulse.play(pulse.Drag(x_pulse.duration, over_amp, x_pulse.sigma, x_pulse.beta), d0)
    inst_map.add("x", (qubit,), x_over)

Let's look at one of the circuits of the FineXAmplitude experiment. 
To calibrate the X gate we add an SX gate before the X gates to move the ideal population
to the equator of the Bloch sphere where the sensitivity to over/under rotations is the highest.

.. jupyter-execute::
    
    overamp_cal = FineXAmplitude(qubit, backend=backend)
    overamp_cal.set_transpile_options(inst_map=inst_map)
    overamp_cal.circuits()[4].draw(output='mpl')

.. jupyter-execute::

    # do the experiment
    exp_data_over = overamp_cal.run(backend).block_for_results()
    print(f"The ping-pong pattern points on the figure below indicate")
    print(f"an over rotation which makes the initial state rotate more than pi.")
    print(f"Therefore, the miscalibrated X gate makes the qubit stay away from the Bloch sphere equator.")
    exp_data_over.figure(0)

We now look at a pulse with an under rotation to see how the FineXAmplitude experiment 
detects this error. We will compare the results to the over rotation above.

.. jupyter-execute::

    # build the under rotated pulse and add it to the instruction schedule map
    with pulse.build(backend=backend, name="x") as x_under:
        pulse.play(pulse.Drag(x_pulse.duration, under_amp, x_pulse.sigma, x_pulse.beta), d0)
    inst_map.add("x", (qubit,), x_under)

    # do the experiment
    underamp_cal = FineXAmplitude(qubit, backend=backend)
    underamp_cal.set_transpile_options(inst_map=inst_map)
        
    exp_data_under = underamp_cal.run(backend).block_for_results()
    exp_data_under.figure(0)

Similarly to the over rotation, the under rotated pulse creates 
qubit populations that do not lie on the equator of the Bloch sphere. 
However, compared to the ping-pong pattern of the over rotated pulse, 
the under rotated pulse produces a flipped ping-pong pattern. 
This allows us to determine not only the magnitude of the rotation error 
but also its sign.

.. jupyter-execute::
    
    # analyze the results
    target_angle = np.pi
    dtheta_over = exp_data_over.analysis_results("d_theta").value.nominal_value
    scale_over = target_angle / (target_angle + dtheta_over)
    dtheta_under = exp_data_under.analysis_results("d_theta").value.nominal_value
    scale_under = target_angle / (target_angle + dtheta_under)
    print(f"The ideal angle is {target_angle:.2f} rad. We measured a deviation of {dtheta_over:.3f} rad in over-rotated pulse case.")
    print(f"Thus, scale the {over_amp:.4f} pulse amplitude by {scale_over:.3f} to obtain {over_amp*scale_over:.5f}.")
    print(f"On the other hand, we measued a deviation of {dtheta_under:.3f} rad in under-rotated pulse case.")
    print(f"Thus, scale the {under_amp:.4f} pulse amplitude by {scale_under:.3f} to obtain {under_amp*scale_under:.5f}.")

-----------------------------------------------------------------------------------
Analyzing a pi/2 pulse
-----------------------------------------------------------------------------------
The amplitude of the SX gate is calibrated with the FineSXAmplitude experiment.
Unlike the FineXAmplitude experiment, the FineSXAmplitude experiment 
does not require other gates than the SX gate since the number of repetitions
can be chosen such that the ideal population is always on the equator of the 
Bloch sphere.
To demonstrate the FineSXAmplitude experiment, we now create a SX pulse by
dividing the amplitude of the X pulse by two.
We expect that this pulse might have a small rotation error which we want to correct.


.. jupyter-execute::

    # build sx_pulse with the default x_pulse from defaults and add it to the InstructionScheduleMap
    sx_pulse = pulse.Drag(x_pulse.duration, 0.5*x_pulse.amp, x_pulse.sigma, x_pulse.beta, name="SXp_d0")
    with pulse.build(name='sx') as sched:
        pulse.play(sx_pulse,d0)
    inst_map.add("sx", (qubit,), sched)

    # do the expeirment
    amp_cal = FineSXAmplitude(qubit, backend)
    amp_cal.set_transpile_options(inst_map=inst_map)
    exp_data_x90p = amp_cal.run().block_for_results()
    exp_data_x90p.figure(0)

From the analysis result, we can see that there is a small rotation error. 

.. jupyter-execute::

    # check how much more the given sx_pulse makes over or under roatation
    print(exp_data_x90p.analysis_results("d_theta"))
    target_angle = np.pi / 2
    dtheta = exp_data_x90p.analysis_results("d_theta").value.nominal_value
    scale = target_angle / (target_angle + dtheta)
    print(f"The ideal angle is {target_angle:.2f} rad. We measured a deviation of {dtheta:.3f} rad.")
    print(f"Thus, scale the {sx_pulse.amp:.4f} pulse amplitude by {scale:.3f} to obtain {sx_pulse.amp*scale:.5f}.")

Let's change the amplitude of the SX pulse by a factor :math:`\pi/2 / (\pi/2 + d\theta)`
to turn it into a sharp :math:`\pi/2` rotation.

.. jupyter-execute::

    pulse_amp = sx_pulse.amp*scale

    with pulse.build(backend=backend, name="sx") as sx_new:
        pulse.play(pulse.Drag(x_pulse.duration, pulse_amp, x_pulse.sigma, x_pulse.beta), d0)

    inst_map.add("sx", (qubit,), sx_new)
    inst_map.get('sx',(qubit,))

    # do the experiment
    data_x90p = amp_cal.run().block_for_results()
    data_x90p.figure(0)

You can now see that the correction to the pulse amplitude has allowed us 
to improve our SX gate as shown by the analysis result below. 

.. jupyter-execute::

    # check the dtheta 
    print(data_x90p.analysis_results("d_theta"))

