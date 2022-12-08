=============================================================
Calibrating single-qubit gates on a pulse backend
=============================================================
In this tutorial we demonstrate how to calibrate single-qubit gates 
on ``SingleTransmonTestBackend`` using the calibration framework in qiskit-experiments. 
We will run experiments to find the qubit frequency, calibrate the amplitude 
of DRAG pulses and chose the value of the DRAG parameter that minimizes leakage.
The calibration framework requires the user to

- setup an instance of Calibrations,

- run calibration experiments which can be found in ``qiskit_experiments.library.calibration``.

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

When setting up the calibrations we add three pulses: a :math:`pi`-rotation, 
with a schedule named ``xp``, a schedule ``xm`` identical to ``xp`` 
but with a nagative amplitude, and a :math:`pi/2`-rotation, with a schedule 
named ``x90p``. Here, we have linked the amplitude of the ``xp`` and ``xm`` pulses. 
Therefore, calibrating the parameters of ``xp`` will also calibrate 
the parameters of ``xm``.

.. jupyter-execute::

    cals = setup_cals(backend)
    add_parameter_guesses(cals)

A samilar setup is achieved by using a pre-built library of gates. 
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
one after another and update the parameters after every experiment. 
We will keep track of the parameter values after every experiment.

====================================
1. Finding qubits with spectroscopy
====================================
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

    columns_to_show = ["parameter","qubits","schedule","value","date_time"]    
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


The instance of ``calibrations`` has automatically been updated with the measured
frequency, as shown below.
In addition to the columns shown below, the calibrations also store the group to which a value belongs, 
whether a values is valid or not and the experiment id that produce a value.

.. jupyter-execute::

    pd.DataFrame(**cals.parameters_table(qubit_list=[qubit]))[columns_to_show]
    
    
=================================================================
2. Calibrating the pulse amplitudes with a Rabi experiment
=================================================================
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

The table above shows that we have now updated the amplitude of our :math:`pi` pulse 
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


=====================================
3. Saving and loading calibrations
=====================================
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

===========================================================
 4. Calibrating the value of the DRAG coefficient
===========================================================

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

==========================================================
5. Fine amplitude calibration
==========================================================
The ``FineAmplitude`` calibration experiment repeats :math:`N` times 
a gate with a pulse to amplify the under or over-rotations 
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
we have managed to reduce the error in the rotation angle :math:`{\rm d}\theta`.










