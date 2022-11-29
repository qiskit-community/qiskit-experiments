##########################
The Calibrations Module
##########################

To produce high fidelity quantum operations, we want to be able to run good gates. The calibration module in qiskit-experiments allows users to run experiments to find the pulse shapes and parameter values that maximizes the fidelity of the resulting quantum operations. Calibrations experiments encapsulates the internal processes and allow experimenters do calibration operations in a quicker way. Without the experiments module, we would need to define pulse schedules and plot the resulting measurement data manually (see also `Qiskit textbook <https://qiskit.org/textbook/ch-quantum-hardware/calibrating-qubits-pulse.html>`_ for calibrating qubits with Qiskit Terra). 

Each experiment usually provides additional information about the system used in subsequent experiments.

.. jupyter-execute::

	import numpy as np

	import qiskit.pulse as pulse
	from qiskit.circuit import Parameter

	from qiskit_experiments.calibration_management import BackendCalibrations

On our own environment, we may use one of the pulse-enabled real backends for all the experiments like below.

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

We can verify whether the backend supports Pulse features by checking the backend configuration.

.. jupyter-execute::	
	
	backend_config = backend.configuration()
	assert backend_config.open_pulse, "Backend doesn't support Pulse"

On the other hand we can also use a mock backend in case no IBM Quantum Experience credentials found. For this tutorial, we will use mock backends prepared for each experiment.

To use in the experiments we first need to define template schedule to calibrate for `x` pulse. 

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

===================================
1. Finding qubits with spectroscopy
===================================
Typically, the first experiment we do is to search for the qubit frequency,  which is the difference between the ground and excited states. This frequency will be crucial for creating pulses which enact particular quantum operators on the qubit.

We start with a mock backend.

.. jupyter-execute::

	from qiskit_experiments.test.test_qubit_spectroscopy import SpectroscopyBackend
	spec_backend = SpectroscopyBackend()

We then setup calibrations for the backend.

.. jupyter-execute::

	cals = setup_cals(spec_backend) # Block until our job and its post processing finish.
	add_parameter_guesses(cals)

We define the qubit we will work with and prepare the experiment using `RoughFrequencyCal`.

.. jupyter-execute::

	from qiskit_experiments.library.calibration.rough_frequency import RoughFrequencyCal

	qubit = 0
	freq01_estimate = spec_backend.defaults().qubit_freq_est[qubit]
	frequencies = np.linspace(freq01_estimate -15e6, freq01_estimate + 15e6, 51)
	spec = RoughFrequencyCal(qubit, cals, frequencies, backend=spec_backend)

.. jupyter-execute::

	circuit = spec.circuits()[0]
	circuit.draw()

We run the experiment. After the experiment completes the value of the amplitudes in the calibrations will automatically be updated. This behaviour can be controlled using the `auto_update` argument given to the calibration experiment at initialization.

.. jupyter-execute::

	spec_data = spec.run().block_for_results() 
	spec_data.figure(0)

We can see the analysis results

.. jupyter-execute::

	print(spec_data.analysis_results("f01"))

========================================================
2. Calibrating the pulse amplitudes with Rabi experiment
========================================================
We are going to run a sample Rabi experiment to calibrate rotations between the ground-state \|0\⟩ and the excited state \|1\⟩. We can think of this as a rotation by π radians around the x-axis of the Bloch sphere. Our goal is to seek the amplitude of the pulse needed to achieve this rotation.

First we define the mock backend.

.. jupyter-execute::

	from qiskit_experiments.test.mock_iq_backend import RabiBackend
	rabi_backend = RabiBackend()

We then setup calibrations for the backend.

.. jupyter-execute::

	cals = setup_cals(rabi_backend)
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
	
	rabi_data = rabi.run(rabi_backend)
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
the :math:`|1\rangle - |2\rangle` transition, reducing the chance of leakage
to the :math:`|2\rangle` state. The optimal value of the DRAG parameter is chosen to
minimize both leakage and phase errors resulting from the AC Stark shift.
The pulse envelope is :math:`f(t) = \Omega_x(t) + j \beta \frac{\rm d}{{\rm d }t} \Omega_x(t)`.
Here, :math:`\Omega_x` is the envelop of the in-phase component of the pulse and
$\beta$ is the strength of the quadrature which we refer to as the DRAG
parameter and seek to calibrate in this experiment. 
The DRAG calibration will run
several series of circuits. In a given circuit a :math:`Rp(β) - Rm(β)` block is repeated
:math:`N` times. Here, Rp is a rotation with a positive angle and Rm is the same rotation
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

==================
1. Miscalibrations
==================

In this section, we will see what if we run a miscalibrated `X` gate - with a false amplitude - on a qubit. After that, we will use the amplitude value we get from the Rabi experiment above to see the difference.

Note that, the following lines are for demonstration purposes and should be run on a real backend to see the actual difference.

We first define a simple circuit that contains an X gate and measurement.

.. jupyter-execute::
	
	from qiskit import QuantumCircuit

	circ = QuantumCircuit(1, 1)
	circ.x(0)
	circ.measure(0, 0)
	circ.draw()

Then we define a calibration for the `X` gate on qubit 0. For the `amp` parameter we use a default wrong value.

.. jupyter-execute::

	from qiskit import pulse, transpile
	from qiskit.test.mock import FakeArmonk
	from qiskit.pulse.library import Constant
	backend = FakeArmonk()

	# build a simple circuit that only contain one x gate and measurement
	circ = QuantumCircuit(1, 1)
	circ.x(0)
	circ.measure(0, 0)
	with pulse.build(backend) as my_schedule:
		pulse.play(Constant(duration=10, amp=0.1), pulse.drive_channel(0)) # build the constant pulse

	circ.add_calibration('x', [0], my_schedule) # map x gate in qubit 0 to my_schedule
	circ = transpile(circ, backend)
	circ.draw(idle_wires=False)

Execute our circuit:

.. jupyter-execute::

	result = backend.run(transpile(circ, backend), shots=1000).result()
	counts  = result.get_counts(circ)
	print(counts)