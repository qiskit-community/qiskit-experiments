T2* Ramsey Characterization
===========================

The purpose of the :math:`T_2^*` Ramsey experiment is to determine two of the qubit's
properties: *Ramsey* or *detuning frequency* and :math:`T_2^\ast`. In this experiment,
we would like to get a more precise estimate of the qubit's frequency given a rough
estimate. The difference between the frequency used for the control rotation pulses and
the qubit transition frequency is called the *detuning frequency*. This part of the
experiment is called a *Ramsey Experiment*. :math:`T_2^\ast` represents the rate of
decay toward a mixed state, when the qubit is initialized to the
:math:`\left|1\right\rangle` state. It is the dephasing time or the transverse
relaxation time of the qubit on the Bloch sphere as a result of both energy relaxation
and pure dephasing in the transverse plane. Unlike :math:`T_2`, which is measured by
:class:`.T2Hahn`, :math:`T_2^*` is sensitive to inhomogenous broadening.

Since the detuning frequency is relatively small, we add a phase gate to the circuit to
enable better measurement. The actual frequency measured is the sum of the detuning
frequency and the user induced *oscillation frequency* (``osc_freq`` parameter).

.. jupyter-execute::

    import numpy as np
    import qiskit
    from qiskit_experiments.library import T2Ramsey

The circuits used for the experiment comprise the following steps:

#. Hadamard gate
#. Delay
#. RZ gate that rotates the qubit in the x-y plane 
#. Hadamard gate
#. Measurement

The user provides as input a series of delays (in seconds) and the
oscillation frequency (in Hz). During the delay, we expect the qubit to
precess about the z-axis. If the p gate and the precession offset each
other perfectly, then the qubit will arrive at the
:math:`\left|0\right\rangle` state (after the second Hadamard gate). By
varying the extension of the delays, we get a series of oscillations of
the qubit state between the :math:`\left|0\right\rangle` and
:math:`\left|1\right\rangle` states. We can draw the graph of the
resulting function, and can analytically extract the desired values.

.. jupyter-execute::

    qubit = 0
    # set the desired delays
    delays = list(np.arange(1e-6, 50e-6, 2e-6))

.. jupyter-execute::

    # Create a T2Ramsey experiment. Print the first circuit as an example
    exp1 = T2Ramsey((qubit,), delays, osc_freq=1e5)
    
    print(exp1.circuits()[0])

We run the experiment on a simulated backend using Qiskit Aer with a
pure T1/T2 relaxation noise model.

.. note::
    This tutorial requires the :external+qiskit_aer:doc:`qiskit-aer <index>` and :external+qiskit_ibm_runtime:doc:`qiskit-ibm-runtime <index>`
    packages to run simulations.  You can install them with ``python -m pip
    install qiskit-aer qiskit-ibm-runtime``.

.. jupyter-execute::

    # A T1 simulator
    from qiskit_ibm_runtime.fake_provider import FakePerth
    from qiskit_aer import AerSimulator
    from qiskit_aer.noise import NoiseModel
    
    # Create a pure relaxation noise model for AerSimulator
    noise_model = NoiseModel.from_backend(
        FakePerth(), thermal_relaxation=True, gate_error=False, readout_error=False
    )
    
    # Create a fake backend simulator
    backend = AerSimulator.from_backend(FakePerth(), noise_model=noise_model)

The resulting graph will have the form:
:math:`f(t) = a e^{-t/T_2*} \cdot \cos(2 \pi f t + \phi) + b` where *t* is
the delay, :math:`T_2^\ast` is the decay factor, and *f* is the detuning
frequency.

.. jupyter-execute::

    # Set scheduling method so circuit is scheduled for delay noise simulation
    exp1.set_transpile_options(scheduling_method='asap')
    
    # Run experiment
    expdata1 = exp1.run(backend=backend, shots=2000, seed_simulator=101)
    expdata1.block_for_results()  # Wait for job/analysis to finish.
    
    # Display the figure
    display(expdata1.figure(0))


.. jupyter-execute::

    # Print results
    for result in expdata1.analysis_results():
        print(result)


Providing initial user estimates
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The user can provide initial estimates for the parameters to help the
analysis process. Because the curve is expected to decay toward
:math:`0.5`, the natural choice for parameters :math:`A` and :math:`B`
is :math:`0.5`. Varying the value of :math:`\phi` will shift the graph
along the x-axis. Since this is not of interest to us, we can safely
initialize :math:`\phi` to 0. In this experiment, ``t2ramsey`` and ``f``
are the parameters of interest. Good estimates for them are values
computed in previous experiments on this qubit or a similar values
computed for other qubits.

.. jupyter-execute::

    user_p0={
        "A": 0.5,
        "T2star": 20e-6,
        "f": 110000,
        "phi": 0,
        "B": 0.5
            }
    exp_with_p0 = T2Ramsey((qubit,), delays, osc_freq=1e5)
    exp_with_p0.analysis.set_options(p0=user_p0)
    exp_with_p0.set_transpile_options(scheduling_method='asap')
    expdata_with_p0 = exp_with_p0.run(backend=backend, shots=2000, seed_simulator=101)
    expdata_with_p0.block_for_results()
    
    # Display fit figure
    display(expdata_with_p0.figure(0))


.. jupyter-execute::

    # Print results
    for result in expdata_with_p0.analysis_results():
        print(result)


See also
--------

* API documentation: :mod:`~qiskit_experiments.library.characterization.T2Ramsey`
