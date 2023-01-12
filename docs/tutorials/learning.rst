Getting Started
===============

Installation
============

Official Qiskit Experiments releases can be installed via the python package manager 
``pip``.

.. code-block:: console

    python -m pip install qiskit-experiments

If you want to install the most up-to-date version instead (may not be stable), you can
install the latest main branch:

.. code-block:: console

    python -m pip install git+https://github.com/Qiskit/qiskit-experiments.git

If you want to develop the package, you can install Qiskit Experiments from source by 
cloning the repository:

.. code-block:: console

    git clone https://github.com/Qiskit/qiskit-experiments.git
    python -m pip install -e qiskit-experiments

The ``-e`` option will keep your installed package up to date as you make or pull new 
changes.

Running your first experiment
=============================

Let's run a T1 experiment. First, we have to import the T1 experiment from the 
Qiskit Experiments library:

.. jupyter-execute::

    from qiskit_experiments.library import T1

Experiments must be run on a backend. We're going to use a simulator, 
:class:`qiskit.providers.fake_provider.FakeVigo`, for 
this example, but you can use any IBM backend that you can access through Qiskit.

.. jupyter-execute::

    from qiskit.providers.fake_provider import FakeVigo
    from qiskit_aer import AerSimulator
    from qiskit.providers.aer.noise import NoiseModel
    import numpy as np

    # Create a pure relaxation noise model for AerSimulator
    noise_model = NoiseModel.from_backend(
        FakeVigo(), thermal_relaxation=True, gate_error=False, readout_error=False
    )

    backend = AerSimulator.from_backend(FakeVigo(), noise_model=noise_model)
    qubit0_t1 = backend.properties().t1(0)

    delays = np.arange(1e-6, 3 * qubit0_t1, 3e-5)
    exp = T1(qubit=0, delays=delays)

Run and display results:

.. jupyter-execute::

    exp_data = exp.run(backend=backend, seed_simulator=101).block_for_results()

    # Print the result
    display(exp_data.figure(0))
    for result in exp_data.analysis_results():
        print(result)

The `block_for_results` method is optional here, and is used to block execution
of subsequent code until the experiment has fully completed execution and analysis.

Setting experiment options
==========================

Often it's insufficient to run an experiment with only the default options. 
There are four types of options one can set for an experiment:

Run options
-----------

These options are passed to the experiment's ``run()`` method and then to the ``run()``
method of your specified backend. Any run option that your backend supports can be set:

.. code-block::

  exp.set_run_options(shots=1000,
                      meas_level=MeasLevel.CLASSIFIED,
                      meas_return="avg")

Consult the documentation of :meth:`qiskit.providers.ibmq.IBMQBackend.run` or 
your specific backend type for valid options.

Transpile options
-----------------
These options are passed to the Terra transpiler to transpile the experiment circuits
before execution:

.. code-block::

  exp.set_transpile_options(scheduling_method='asap',
                            optimization_level=3,
                            basis_gates=["x", "sx", "rz"])

Consult the documentation of :func:`qiskit.compiler.transpile` for valid options.

Experiment options
------------------
These options are unique to each experiment class. Many experiment options can be set
upon experiment instantiation, but can also be explicitly set via :meth:`set_experiment_options`:

.. code-block:: 

  exp = T1(qubit=i, delays=delays)
  exp.set_experiment_options(delays=new_delays)

Consult the :doc:`API documentation <apidocs/index>` for the options of each experiment class.

Analysis options
----------------
These options are unique to each analysis class. Unlike the other options, this one is not set via the experiment object
but via analysis instead:

.. code-block::

  exp = rb.StandardRB(qubits=(0,),
                      lengths=list(range(1, 300, 30)),
                      seed=123,
                      backend=backend)
  exp.analysis.set_options(gate_error_ratio=None)

Consult the :doc:`API documentation <apidocs/index>` for the options of each 
experiment's analysis class. 

Running experiments on multiple qubits
======================================

To run experiments across many qubits of the same device, we use **composite experiments**.
There are two core types of composite experiments:

* **Parallel experiments** run across qubits simultaneously. The circuits cannot overlap in 
  qubits used.
* **Batch experiments** run consecutively in time. These circuits can overlap in qubits.

Here's an example of measuring :math:`T_1` of two qubits on the same device simultaneously
in a parallel experiment:

.. jupyter-execute::

    from qiskit_experiments.framework import ParallelExperiment

    # Create a parallel T1 experiment
    parallel_exp = ParallelExperiment([T1(qubit=i, delays=delays) for i in range(2)])
    parallel_exp.set_transpile_options(scheduling_method='asap')
    parallel_data = parallel_exp.run(backend, seed_simulator=101).block_for_results()
    
    # View result data
    for result in parallel_data.analysis_results():
        print(result)

Parallel and batch experiments can be nested arbitrarily to make

Viewing sub experiment data
===========================

The experiment data returned from a composite experiment also contains
individual experiment data for each sub experiment, which can be accessed
using ``child_data``.

.. jupyter-execute::

    # Print sub-experiment data
    for i, sub_data in enumerate(parallel_data.child_data()):
        print("Component experiment",i)
        display(sub_data.figure(0))
        for result in sub_data.analysis_results():
            print(result)

You can use the `flatten_results` flag to flatten the results of all 
component experiments into one level.

