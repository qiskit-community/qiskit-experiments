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

	from qiskit import IBMQ, schedules

On our own environment, we may use one of the pulse-enabled real backends like below.

.. jupyter-execute::

	# IBMQ.load_account()
	# provider = IBMQ.get_provider(hub='ibm-q', group='open', project='main')
	# backend = provider.get_backend('ibmq_armonk')

We can use a mock backend in case no IBM Quantum Experience credentials found.

.. jupyter-execute::

	from qiskit_experiments.test.mock_iq_backend import RabiBackend
	backend = RabiBackend()

========================================================
1. Calibrating the pulse amplitudes with Rabi experiment
========================================================
We are going to run a sample Rabi experiment to calibrate rotations between the ground-state \|0\⟩ and the excited state \|1\⟩. We can think of this as a rotation by π radians around the x-axis of the Bloch sphere. Our goal is to seek the amplitude of the pulse needed to achieve this rotation.

We create a new Rabi experiment instance by providing the qubit index to be calibrated. In the Rabi experiment we apply a pulse at the frequency of the qubit and scan its amplitude to find the amplitude that creates a rotation of a desired angle.

We do this with the calibration experiment RoughXAmplitudeCal. This is a specialization of the Rabi experiment that will update the calibrations for the X pulse automatically.

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

.. jupyter-execute::

	from qiskit_experiments.library.calibration import RoughAmplitudeCal

	qubit = 0

	rabi = RoughAmplitudeCal(qubit, cals)

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
2. Saving and loading calibrations
==================================

The values of the calibrated parameters can be saved to a .csv file and reloaded at a later point in time.

.. jupyter-execute::

	cals.save(file_type="csv", overwrite=True, file_prefix="Armonk")

After saving the values of the parameters you may restart your kernel. If you do so, you will only need to run the following cell to recover the state of your calibrations. Since the schedules are currently not stored we need to call our `setup_cals` function to populate an instance of `Calibrations` with the template schedules. By contrast, the value of the parameters will be recovered from the file.

.. jupyter-execute::

	cals = BackendCalibrations(backend)
	cals.load_parameter_values(file_name="Armonkparameter_values.csv")