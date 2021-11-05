#########################################
Run a Single-Qubit Calibration Experiment
#########################################

The calibration module in qiskit-experiments allows users to run calibration experiments to find the pulse shapes and parameter values that maximizes the fidelity of the resulting quantum operations. To produce high fidelity quantum operations, we want to be able to run good gates. Calibrations experiments encapsulates the internal processes and allow experimenters do calibration operations in a quicker way. Without the experiments module, we need to define pulse schedules and plot the resulting measurement data manually (see also `Qiskit textbook <https://qiskit.org/textbook/ch-quantum-hardware/calibrating-qubits-pulse.html>`_ for calibrating qubits with Qiskit Terra). 

.. jupyter-execute::

	import numpy as np

	import qiskit.pulse as pulse
	from qiskit.circuit import Parameter

	from qiskit_experiments.calibration_management.backend_calibrations import BackendCalibrations
	from qiskit_experiments.library.calibration import Rabi

	from qiskit import IBMQ, schedule

For this guide, we choose one of the publicly available and pulse-enabled backends.

.. jupyter-execute::

	IBMQ.load_account()
	provider = IBMQ.get_provider(hub='ibm-q', group='open', project='main')
	backend = provider.get_backend('ibmq_armonk')

========================================================
1. Calibrating the pulse amplitudes with Rabi experiment
========================================================
We are going to run a sample Rabi experiment to calibrate rotations between the ground-state \|0\⟩ and the excited state \|1\⟩. We can think of this as a rotation by π radians around the x-axis of the Bloch sphere. Our goal is to seek the amplitude of the pulse needed to achieve this rotation.

We create a new Rabi experiment instance by providing the qubit index to be calibrated. In the Rabi experiment we apply a pulse at the frequency of the qubit and scan its amplitude to find the amplitude that creates a rotation of a desired angle.

.. jupyter-execute::

	qubit = 0

	rabi = Rabi(qubit)

We can give custom amplitude values by providing a list to the ``amplitude`` parameter of ``set_experiment_options()`` method or run the experiment with default values. See `API reference <https://qiskit.org/documentation/experiments/stubs/qiskit_experiments.library.calibration.Rabi.html#qiskit_experiments.library.calibration.Rabi>`_ for the current default amplitude values.

.. jupyter-execute::

	rabi.set_experiment_options(
		amplitudes=np.linspace(-0.95, 0.95, 51)
	)

.. jupyter-execute::
	
	#rabi_data = rabi.run(backend)
	#rabi_data.block_for_results() # Block until our job and its post processing finish.
	#print(rabi_data)

In the analysis results, ``rabi_rate`` is the unit of frequency which our qubit completes a full cycle from ground-state \|0\⟩ to the excited state \|1\⟩ and back to ground-state \|0\⟩. Using this information we calculate one period, which is a full cycle by 2π radians around the x-axis of the Bloch sphere. However our goal was to seek the amplitude of the pulse needed to achieve a rotation by π radians which will take our qubit from ground-state \|0\⟩ to the excited state \|1\⟩. So we need to divide it by 2.

.. jupyter-execute::
	
	#pi_pulse_amplitude = (1/rabi_data.analysis_results("rabi_rate").value.value) / 2
	#print(pi_pulse_amplitude)

Optionally we can save our experiment to load the analysis results at a later time. See `Saving Experiment Data to the Cloud <https://qiskit.org/documentation/experiments/tutorials/experiment_cloud_service.html>`_ guide for further details on saving experiments.

.. jupyter-execute::

	#rabi_data.save()
