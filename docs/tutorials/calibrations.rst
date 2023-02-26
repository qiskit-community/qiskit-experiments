Calibrations: Schedules and gate parameters from experiments 
############################################################

To produce high fidelity quantum operations, we want to be able to run good gates. The 
calibration module in Qiskit Experiments allows users to run experiments to find the 
pulse shapes and parameter values that maximize the fidelity of the resulting quantum 
operations. Calibration experiments encapsulate the internal processes and allow 
experimenters to perform calibration operations in a quicker way. Without the experiments 
module, we would need to define pulse schedules and plot the resulting measurement 
data manually (see also the `Qiskit textbook <https://qiskit.org/textbook/ch-quantum-hardware/calibrating-qubits-pulse.html>`_ for calibrating qubits with Qiskit Terra). 

Calibrating single-qubit gates on a pulse backend
=================================================

In this tutorial, we demonstrate how to calibrate single-qubit gates using the 
calibration framework in Qiskit Experiments. You can run these experiments on any 
backend with Pulse enabled:

.. jupyter-execute::
    :hide-code:
    :hide-output:

    from qiskit.test.ibmq_mock import mock_get_backend
    backend = mock_get_backend('FakeLima')

.. jupyter-execute::

	from qiskit import IBMQ
	IBMQ.load_account()
	provider = IBMQ.get_provider(hub='ibm-q', group='open', project='main')
	backend = provider.get_backend('ibmq_lima')

We can verify whether the backend supports Pulse features by checking the 
backend configuration:

.. jupyter-execute::	
	
	backend_config = backend.configuration()
	assert backend_config.open_pulse, "Backend doesn't support Pulse"

For the purposes of the tutorial, we will run experiments on our test pulse 
backend, ``SingleTransmonTestBackend``, a backend that simulates the underlying pulses 
with Qiskit Dynamics on a three-level model of a transmon. We will run experiments to 
find the qubit frequency, calibrate the amplitude of DRAG pulses, and choose the value 
of the DRAG parameter that minimizes leakage. The calibration framework requires 
the user to

- Setup an instance of Calibrations,

- Run calibration experiments, found in ``qiskit_experiments.library.calibration``.

Note that the values of the parameters stored in the instance of the ``Calibrations`` class 
will automatically be updated by the calibration experiments. 
This automatic updating can also be disabled using the ``auto_update`` flag.

.. jupyter-execute::

    import pandas as pd
    import numpy as np
    import qiskit.pulse as pulse
    from qiskit.circuit import Parameter
    from qiskit_experiments.calibration_management.calibrations import Calibrations
    from qiskit import schedule
    from qiskit_experiments.test.pulse_backend import SingleTransmonTestBackend

.. jupyter-execute::

    backend = SingleTransmonTestBackend(5.2e9,-.25e9, 1e9, 0.8e9, noise=False)
    qubit = 0 
    cals=Calibrations.from_backend(backend)
    print(cals.get_inst_map())

The two functions below show how to setup an instance of Calibrations. 
To do this the user defines the template schedules to calibrate. 
These template schedules are fully parameterized, even the channel indices 
on which the pulses are played. Furthermore, the name of the parameter in the channel 
index must follow the convention laid out in the documentation 
of the calibration module. Note that the parameters in the channel indices 
are automatically mapped to the channel index when get_schedule is called.

.. jupyter-execute::
    
    # A function to instantiate calibrations and add a couple of template schedules.
    def setup_cals(backend) -> Calibrations:
    
        cals = Calibrations.from_backend(backend)
        
        dur = Parameter("dur")
        amp = Parameter("amp")
        sigma = Parameter("σ")
        beta = Parameter("β")
        drive = pulse.DriveChannel(Parameter("ch0"))

        # Define and add template schedules.
        with pulse.build(name="xp") as xp:
            pulse.play(pulse.Drag(dur, amp, sigma, beta), drive)

        with pulse.build(name="xm") as xm:
            pulse.play(pulse.Drag(dur, -amp, sigma, beta), drive)

        with pulse.build(name="x90p") as x90p:
            pulse.play(pulse.Drag(dur, Parameter("amp"), sigma, Parameter("β")), drive)

        cals.add_schedule(xp, num_qubits=1)
        cals.add_schedule(xm, num_qubits=1)
        cals.add_schedule(x90p, num_qubits=1)

        return cals

    # Add guesses for the parameter values to the calibrations.
    def add_parameter_guesses(cals: Calibrations):
        
        for sched in ["xp", "x90p"]:
            cals.add_parameter_value(80, "σ", schedule=sched)
            cals.add_parameter_value(0.5, "β", schedule=sched)
            cals.add_parameter_value(320, "dur", schedule=sched)
            cals.add_parameter_value(0.5, "amp", schedule=sched)

When setting up the calibrations we add three pulses: a :math:`\pi`-rotation, 
with a schedule named ``xp``, a schedule ``xm`` identical to ``xp`` 
but with a nagative amplitude, and a :math:`\pi/2`-rotation, with a schedule 
named ``x90p``. Here, we have linked the amplitude of the ``xp`` and ``xm`` pulses. 
Therefore, calibrating the parameters of ``xp`` will also calibrate 
the parameters of ``xm``.

.. jupyter-execute::

    cals = setup_cals(backend)
    add_parameter_guesses(cals)

A similar setup is achieved by using a pre-built library of gates. 
The library of gates provides a standard set of gates and some initial guesses 
for the value of the parameters in the template schedules. 
This is shown below using the ``FixedFrequencyTransmon`` library which provides the ``x``,
``y``, ``sx``, and ``sy`` pulses. Note that in the example below 
we change the default value of the pulse duration to 320 samples

.. jupyter-execute::

    from qiskit_experiments.calibration_management.basis_gate_library import FixedFrequencyTransmon

    library = FixedFrequencyTransmon(default_values={"duration": 320})
    cals = Calibrations.from_backend(backend, libraries=[library])
    print(library.default_values()) # check what parameter values this library has
    print(cals.get_inst_map()) # check the new cals's InstructionScheduleMap made from the library
    print(cals.get_schedule('x',(0,))) # check one of the schedules built from the new calibration

We are going to run the spectroscopy, Rabi, DRAG, and fine-amplitude calibration experiments 
one after another and update the parameters after every experiment, keeping track of
parameter values. 

Finding qubits with spectroscopy
--------------------------------

Here, we are using a backend for which we already know the qubit frequency. 
We will therefore use the spectroscopy experiment to confirm that 
there is a resonance at the qubit frequency reported by the backend.

.. jupyter-execute::

    from qiskit_experiments.library.calibration.rough_frequency import RoughFrequencyCal

We first show the contents of the calibrations for qubit 0. 
Note that the guess values that we added before apply to all qubits on the chip. 
We see this in the table below as an empty tuple ``()`` in the qubits column. 
Observe that the parameter values of ``y`` do not appear in this table as they are given by the values of ``x``.

.. jupyter-execute::

    columns_to_show = ["parameter", "qubits", "schedule", "value", "date_time"]    
    pd.DataFrame(**cals.parameters_table(qubit_list=[qubit, ()]))[columns_to_show]


.. jupyter-execute::

    freq01_estimate = backend.defaults().qubit_freq_est[qubit]
    frequencies = np.linspace(freq01_estimate -15e6, freq01_estimate + 15e6, 51)
    spec = RoughFrequencyCal(qubit, cals, frequencies, backend=backend)
    spec.set_experiment_options(amp=0.005)

.. jupyter-execute::

    circuit = spec.circuits()[0]
    circuit.draw(output="mpl")

.. jupyter-execute::

    next(iter(circuit.calibrations["Spec"].values())).draw() # let's check the schedule   
    

.. jupyter-execute::

    spec_data = spec.run().block_for_results()
    spec_data.figure(0) 


.. jupyter-execute::

    print(spec_data.analysis_results("f01"))


The instance of ``calibrations`` has been automatically updated with the measured
frequency, as shown below.
In addition to the columns shown below, the calibrations also store the group to which a value belongs, 
whether a values is valid or not and the experiment id that produce a value.

.. jupyter-execute::

    pd.DataFrame(**cals.parameters_table(qubit_list=[qubit]))[columns_to_show]
    
.. _Rabi Calibration:

Calibrating the pulse amplitudes with a Rabi experiment
-------------------------------------------------------

In the Rabi experiment we apply a pulse at the frequency of the qubit 
and scan its amplitude to find the amplitude that creates a rotation 
of a desired angle. We do this with the calibration experiment ``RoughXSXAmplitudeCal``.
This is a specialization of the ``Rabi`` experiment that will update the calibrations 
for both the ``X`` pulse and the ``SX`` pulse using a single experiment.

.. jupyter-execute:: 

    from qiskit_experiments.library.calibration import RoughXSXAmplitudeCal
    rabi = RoughXSXAmplitudeCal(qubit, cals, backend=backend, amplitudes=np.linspace(-0.1, 0.1, 51))

The rough amplitude calibration is therefore a Rabi experiment in which 
each circuit contains a pulse with a gate. Different circuits correspond to pulses 
with different amplitudes.

.. jupyter-execute::

    rabi.circuits()[0].draw("mpl")

After the experiment completes the value of the amplitudes in the calibrations 
will automatically be updated. This behaviour can be controlled using the ``auto_update``
argument given to the calibration experiment at initialization.

.. jupyter-execute::

    rabi_data = rabi.run().block_for_results()
    rabi_data.figure(0)

.. jupyter-execute::

    print(rabi_data.analysis_results("rabi_rate"))

.. jupyter-execute::

    pd.DataFrame(**cals.parameters_table(qubit_list=[qubit, ()], parameters="amp"))[columns_to_show]

The table above shows that we have now updated the amplitude of our :math:`\pi` pulse 
from 0.5 to the value obtained in the most recent Rabi experiment. 
Importantly, since we linked the amplitudes of the ``x`` and ``y`` schedules 
we will see that the amplitude of the ``y`` schedule has also been updated 
as seen when requesting schedules form the ``Calibrations`` instance. 
Furthermore, we used the result from the Rabi experiment to also update 
the value of the ``sx`` pulse. 

.. jupyter-execute::

    cals.get_schedule("sx", qubit)

.. jupyter-execute::

    cals.get_schedule("x", qubit)
   
.. jupyter-execute::

    cals.get_schedule("y", qubit)

Saving and loading calibrations
-------------------------------

The values of the calibrated parameters can be saved to a .csv file 
and reloaded at a later point in time. 

.. jupyter-execute::

    cals.save(file_type="csv", overwrite=True, file_prefix="PulseBackend")

After saving the values of the parameters you may restart your kernel. If you do so, 
you will only need to run the following cell to recover the state of your calibrations. 
Since the schedules are currently not stored we need to call our ``setup_cals`` function 
or use a library to populate an instance of Calibrations with the template schedules. 
By contrast, the value of the parameters will be recovered from the file.

.. jupyter-execute::

    cals = Calibrations.from_backend(backend, library)
    cals.load_parameter_values(file_name="PulseBackendparameter_values.csv")

.. jupyter-execute::

    pd.DataFrame(**cals.parameters_table(qubit_list=[qubit, ()], parameters="amp"))[columns_to_show]

.. _DRAG Calibration:

Calibrating the value of the DRAG coefficient
---------------------------------------------

A Derivative Removal by Adiabatic Gate (DRAG) pulse is designed to minimize leakage 
and phase errors to a neighbouring transition. It is a standard pulse with an additional 
derivative component. It is designed to reduce the frequency spectrum of a 
normal pulse near the  :math:`|1> - |2>` transition, 
reducing the chance of leakage to the :math:`|2>` state. 
The optimal value of the DRAG parameter is chosen to minimize both 
leakage and phase errors resulting from the AC Stark shift. 
The pulse envelope is :math:`f(t)=\Omega_x(t)+j\beta\frac{\rm d}{{\rm d}t}\Omega_x(t)`.
Here, :math:`\Omega_x(t)` is the envelop of the in-phase component 
of the pulse and :math:`\beta` is the strength of the quadrature 
which we refer to as the DRAG parameter and seek to calibrate 
in this experiment. The DRAG calibration will run several 
series of circuits. In a given circuit a Rp(β) - Rm(β) block
is repeated :math:`N` times. Here, Rp is a rotation 
with a positive angle and Rm is the same rotation with a 
negative amplitude.

.. jupyter-execute::

    from qiskit_experiments.library import RoughDragCal
    cal_drag = RoughDragCal(qubit, cals, backend=backend, betas=np.linspace(-20, 20, 25))
    cal_drag.set_experiment_options(reps=[3, 5, 7])
    cal_drag.circuits()[5].draw(output='mpl')

.. jupyter-execute::

    drag_data = cal_drag.run().block_for_results()
    drag_data.figure(0) 

.. jupyter-execute::

    print(drag_data.analysis_results("beta"))

.. jupyter-execute::

    pd.DataFrame(**cals.parameters_table(qubit_list=[qubit, ()], parameters="β"))[columns_to_show]

.. _fine-amplitude-cal:

Fine amplitude calibration
--------------------------

The :class:`.FineAmplitude` experiment and its subclass experiments repeats 
a gate :math:`N` times with a pulse to amplify the under or over-rotations 
in the gate to determine the optimal amplitude.

.. jupyter-execute::
    
    from qiskit_experiments.library.calibration.fine_amplitude import FineXAmplitudeCal
    amp_x_cal = FineXAmplitudeCal(qubit, cals, backend=backend, schedule_name="x")
    amp_x_cal.circuits()[5].draw(output="mpl")

.. jupyter-execute::

    data_fine = amp_x_cal.run().block_for_results()
    data_fine.figure(0)

.. jupyter-execute::

    print(data_fine.analysis_results("d_theta"))

The cell below shows how the amplitude is updated based on the error in the rotation angle measured by the FineXAmplitude experiment. Note that this calculation is automatically done by the Amplitude.update function.

.. jupyter-execute::

    dtheta = data_fine.analysis_results("d_theta").value.nominal_value
    target_angle = np.pi
    scale = target_angle / (target_angle + dtheta)
    pulse_amp = cals.get_parameter_value("amp", qubit, "x")
    print(f"The ideal angle is {target_angle:.2f} rad. We measured a deviation of {dtheta:.3f} rad.")
    print(f"Thus, scale the {pulse_amp:.4f} pulse amplitude by {scale:.3f} to obtain {pulse_amp*scale:.5f}.")

Observe, once again, that the calibrations have automatically been updated.

.. jupyter-execute::

    pd.DataFrame(**cals.parameters_table(qubit_list=[qubit, ()], parameters="amp"))[columns_to_show]

To check that we have managed to reduce the error in the rotation angle we will run the fine amplitude calibration experiment once again.

.. jupyter-execute::

    data_fine2 = amp_x_cal.run().block_for_results()
    data_fine2.figure(0)

.. jupyter-execute::

    print(data_fine2.analysis_results("d_theta"))

As can be seen from the data above and the analysis result below 
we have managed to reduce the error in the rotation angle dtheta.

Fine amplitude calibration of the :math:`\pi`/2 rotation
--------------------------------------------------------

We now wish to calibrate the amplitude of the :math:`\pi/2` rotation.

.. jupyter-execute::

    from qiskit_experiments.library.calibration.fine_amplitude import FineSXAmplitudeCal

    amp_sx_cal = FineSXAmplitudeCal(qubit, cals, backend=backend, schedule_name="sx")
    amp_sx_cal.circuits()[5].draw(output="mpl")

.. jupyter-execute::

    data_fine_sx = amp_sx_cal.run().block_for_results()
    data_fine_sx.figure(0)

.. jupyter-execute::

    print(data_fine_sx.analysis_results(0))

.. jupyter-execute::

    print(data_fine_sx.analysis_results("d_theta"))

.. jupyter-execute::

    pd.DataFrame(**cals.parameters_table(qubit_list=[qubit, ()], parameters="amp"))[columns_to_show]


Fine calibrations of a pulse amplitude
======================================

The amplitude of a pulse can be precisely calibrated using
error amplifying gate sequences. These gate sequences apply 
the same gate a variable number of times. Therefore, if each gate
has a small error :math:`d\theta` in the rotation angle then 
a sequence of :math:`n` gates will have a rotation error of :math:`n` * :math:`d\theta`.

.. jupyter-execute:: 

    import numpy as np
    from qiskit.pulse import InstructionScheduleMap
    import qiskit.pulse as pulse
    from qiskit_experiments.library import FineXAmplitude, FineSXAmplitude
    from qiskit_experiments.test.pulse_backend import SingleTransmonTestBackend

.. jupyter-execute::

    backend = SingleTransmonTestBackend()
    qubit = 0

Fine `X` gate amplitude calibration
-----------------------------------

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

Analyzing a :math:`\pi`/2 pulse
-------------------------------

The amplitude of the `SX` gate is calibrated with the FineSXAmplitude experiment.
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







