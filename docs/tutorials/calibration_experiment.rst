#########################################
Run a Single-Qubit Calibration Experiment
#########################################

To produce high fidelity quantum operations, we want to be able to run good gates. The calibration module in qiskit-experiments allows users to run experiments to find the pulse shapes and parameter values that maximizes the fidelity of the resulting quantum operations. Calibrations experiments encapsulates the internal processes and allow experimenters do calibration operations in a quicker way. Without the experiments module, we would need to define pulse schedules and plot the resulting measurement data manually (see also `Qiskit textbook <https://qiskit.org/textbook/ch-quantum-hardware/calibrating-qubits-pulse.html>`_ for calibrating qubits with Qiskit Terra). 

.. jupyter-execute::

	import numpy as np

	import qiskit.pulse as pulse
	from qiskit.circuit import Parameter

	from qiskit_experiments.calibration_management import BackendCalibrations

	from qiskit.pulse import InstructionScheduleMap

On our own environment, we may use one of the pulse-enabled real backends like below.

.. jupyter-execute::

	# IBMQ.load_account()
	# provider = IBMQ.get_provider(hub='ibm-q', group='open', project='main')
	# backend = provider.get_backend('ibmq_armonk')

We can use a mock backend in case no IBM Quantum Experience credentials found. For this tutorial, we will use mock backends suited to each experiment.

.. jupyter-execute::

	from qiskit_experiments.test.mock_iq_backend import RabiBackend
	backend = RabiBackend()

===================================
1. Finding qubits with spectroscopy
===================================

========================================================
2. Calibrating the pulse amplitudes with Rabi experiment
========================================================
We are going to run a sample Rabi experiment to calibrate rotations between the ground-state \|0\⟩ and the excited state \|1\⟩. We can think of this as a rotation by π radians around the x-axis of the Bloch sphere. Our goal is to seek the amplitude of the pulse needed to achieve this rotation.

We first need to define template schedule to calibrate for `x` pulse.

.. jupyter-execute::

	def setup_cals(backend) -> BackendCalibrations:
		"""A function to instantiate calibrations and add a couple of template schedules."""
		cals = BackendCalibrations(backend)

		dur = Parameter("dur")
		amp = Parameter("amp")
		sigma = Parameter("σ")
		beta = Parameter("β")
		drive = pulse.DriveChannel(Parameter("ch0"))

		# Define and add template schedules.
		with pulse.build(name="x") as x:
			pulse.play(pulse.Drag(dur, amp, sigma, beta), drive)

		cals.add_schedule(x, num_qubits=1)
		
		return cals

	def add_parameter_guesses(cals: BackendCalibrations):
		
		"""Add guesses for the parameter values to the calibrations."""
		for sched in ["x"]:
			print(sched)
			cals.add_parameter_value(80, "σ", schedule=sched)
			cals.add_parameter_value(0.5, "β", schedule=sched)
			cals.add_parameter_value(320, "dur", schedule=sched)
			cals.add_parameter_value(0.5, "amp", schedule=sched)

	cals = setup_cals(backend)
	add_parameter_guesses(cals)

We create a new Rabi experiment instance by providing the qubit index to be calibrated. In the Rabi experiment we apply a pulse at the frequency of the qubit and scan its amplitude to find the amplitude that creates a rotation of a desired angle.

We do this with the calibration experiment `RoughAmplitudeCal`. This is a calibration version of the Rabi experiment that will update the calibrations for the X pulse automatically.

If we do not set any experiment options using `set_experiment_options()` method, experiment will use the default values. Default values can be seen `here <https://qiskit.org/documentation/experiments/stubs/qiskit_experiments.library.calibration.Rabi.html#qiskit_experiments.library.calibration.Rabi>`__ under `Experiment Options`.

.. jupyter-execute::

	from qiskit_experiments.library.calibration import RoughAmplitudeCal

	qubit = 0

	rabi = RoughAmplitudeCal(qubit, cals)

The rough amplitude calibration is therefore a Rabi experiment in which each circuit contains a pulse with a gate. Different circuits correspond to pulses with different amplitudes.

.. jupyter-execute::

	rabi.circuits()[0].draw()

After the experiment completes the value of the amplitudes in the calibrations will automatically be updated. This behaviour can be controlled using the `auto_update` argument given to the calibration experiment at initialization.

.. jupyter-execute::
	
	rabi_data = rabi.run(backend)
	rabi_data.block_for_results() # Block until our job and its post processing finish.
	print(rabi_data)

.. jupyter-execute::

	rabi_data.figure(0)

In the analysis results, ``rabi_rate`` is the unit of frequency which our qubit completes a full cycle by 2π radians around the x-axis of the Bloch sphere. Using this information we calculate one period. However our goal was to seek the amplitude of the pulse needed to achieve a rotation by π radians which will take our qubit from ground-state \|0\⟩ to the excited state \|1\⟩. So we need to divide it by 2.

.. jupyter-execute::
	
	pi_pulse_amplitude = (1/rabi_data.analysis_results("rabi_rate").value.value) / 2
	print(pi_pulse_amplitude)

==================================
3. Saving and loading calibrations
==================================

The values of the calibrated parameters can be saved to a .csv file and reloaded at a later point in time.

.. code-block:: python

	cals.save(file_type="csv", overwrite=True, file_prefix="RabiBackend")

After saving the values of the parameters we may restart our kernel. If we do so, we will only need to run the following cell to recover the state of the calibrations. Since the schedules are currently not stored we need to call our `setup_cals` function to populate an instance of `Calibrations` with the template schedules. By contrast, the value of the parameters will be recovered from the file.

.. code-block:: python

	from qiskit_experiments.test.mock_iq_backend import RabiBackend
	rabi_backend = RabiBackend()
	cals = BackendCalibrations(rabi_backend)
	cals.load_parameter_values(file_name="RabiBackendparameter_values.csv")

=======================================================
4. Using the Calibrated Amplitude in Another Experiment
=======================================================
------------------------------------------------------
4.1. Calibrating the value of the DRAG coefficient
------------------------------------------------------

A Derivative Removal by Adiabatic Gate (DRAG) pulse is designed to minimize leakage
to a neighbouring transition. It is a standard pulse with an additional derivative
component. It is designed to reduce the frequency spectrum of a normal pulse near
the $|1\rangle$ - $|2\rangle$ transition, reducing the chance of leakage
to the $|2\rangle$ state. The optimal value of the DRAG parameter is chosen to
minimize both leakage and phase errors resulting from the AC Stark shift.
The pulse envelope is $f(t) = \Omega_x(t) + j \beta \frac{\rm d}{{\rm d }t} \Omega_x(t)$.
Here, $\Omega_x$ is the envelop of the in-phase component of the pulse and
$\beta$ is the strength of the quadrature which we refer to as the DRAG
parameter and seek to calibrate in this experiment. 
The DRAG calibration will run
several series of circuits. In a given circuit a Rp(β) - Rm(β) block is repeated
$N$ times. Here, Rp is a rotation with a positive angle and Rm is the same rotation
with a negative amplitude.

We use a mock backend in case no IBM credentials found.

.. jupyter-execute::

	from qiskit_experiments.test.mock_iq_backend import DragBackend
	drag_backend = DragBackend(gate_name="Drag(x)")

We define the template schedule for `x` pulse using previous methods.

Note that, if we run the experiments on real backends, we wouldn't need to define template schedules again.

.. jupyter-execute::

	cals = setup_cals(drag_backend)
	add_parameter_guesses(cals)

We create a calibration version of Drag experiment instance by providing the qubit index to be calibrated. We use the calibration version of Drag experiment `RoughDragCal`. This is a calibration version of the Rabi experiment that will update the calibrations for the X pulse automatically.

If we do not set any experiment options using `set_experiment_options()` method, experiment will use the default values. Default values can be seen `here <https://qiskit.org/documentation/experiments/stubs/qiskit_experiments.library.calibration.DragCal.html#qiskit_experiments.library.calibration.DragCal>`__ under `Experiment Options`.

.. jupyter-execute::

	from qiskit_experiments.library import RoughDragCal
	drag = RoughDragCal(qubit, cals)

.. jupyter-execute::

	drag_data = drag.run(drag_backend)
	drag_data.block_for_results()

.. jupyter-execute::

	drag_data.figure(0)

===============
5. Failure Mode
===============