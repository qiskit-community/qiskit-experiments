.. _qiskit-experiments:

================================
Qiskit Experiments API Reference
================================

To execute useful algorithms with quantum computing systems,
you need to start from scrutinizing the physical parameters of a quantum processor (:ref:`characterization`)
and carefully design stimulus pulses tailored for each qubit (:ref:`calibration`)
to maximize the fidelity of your quantum state operation constituting
the instruction set architecture, or `basis gates` in the Qiskit terminology.
Once this process has completed, you need to evaluate different levels of performance metrics
from a single qubit gate error to the quantum volume for a group of qubits (:ref:`verification`).

During this workflow, experimentalists need to deal with many different experiments and
need to manage massive amount of analysis data.
Furthermore, physical properties of qubits may change from time to time,
and this operation is often repeated on a regular basis while keeping the past data available for the time series analysis.
This is where the Qiskit Experiments comes into play.
This project aims at lightening the burden of experimentalists with the circuit generation and
data analysis framework implemented upon the Qiskit core package with preset experiment libraries.


Overview
========

.. rubric:: Running experiment

A workflow of the experiment can be roughly divided into two operations without loss of generality, namely,
generation of the experiment program and analysis of data returned from the quantum system.
These operations are embodied as a python class which we call `experiment class` and `analysis class`.

This illustrates a typical workflow in Qiskit Experiments.

.. code-block:: python
    :linenos:
    :emphasize-lines: 3,6,11

    from qiskit_experiments.library import MyExperiment

    exp = MyExperiment(qubits=[0, 1])
    exp.set_experiment_options(**options)

    exp_data = exp.run(backend)

    # wait for backend queue
    exp_data.block_for_results()

    exp_data.save()


An experiment class contains the default analysis class, thus you just need to import the experiment class
from the experiment library and initialize a class instance with the set of options.
Once all options are set, you can call :py:meth:`run` method to execute the experiment.
This internally generates experimental circuits according to the options you set and submits a job to the specified ``backend``.
Then, it initializes the preset analysis class and performs data analysis immediately after you receive the experiment result.
The method returns formatted data object that conforms to the remote database specification.
This object is stored in the database by calling the :py:meth:`save` method.
The stored data can be retrieved at anytime even if you shutdown your python kernel without saving.
You can refer to :doc:`/tutorials/resultsdb_tutorial` for the use case of the remote database.

.. rubric:: Composite experiment

In the policy of Qiskit Experiments, an experiment class is defined for the minimum unit of device component,
i.e. if an experiment runs on a single qubit, this experiment always takes one qubit index.
This allows us to remove ambiguity from the expression of experiment.
For example, ``qubits=[0, 1]`` may represent a two-qubit experiment runs on a pair of qubit 0 and 1,
or it can be read as a single-qubit experiment simultaneously runs on a group of qubit.
According to our policy, this always specifies the two-qubit experiment.

In practice, there are many use cases experimentalits want to run the same, or even different experiments on different
qubits at the same time to save the total execution time.
To cope with this situation, Qiskit Experiments provides a feature called `composite experiment`.
This is a python class that collects initialized experiment instances and run them as if a single composite experiment.
See :ref:`composite-experiment` for more details.

.. rubric:: Extending experiment library

If you need an experiment that is not available in our experiment library,
you can still write your own experiment class inheriting from the base experiment class.
Because complicated job and data management mechanism are already implemented in the base class,
you can enjoy creating own experiment by just providing a circuit generation logic.
You can reuse existing analysis class or you can define your own one in the same way.
If you are an experiment developer, you might be interested in :ref:`create-experiment` for more details.
Since Qiskit Experiments is distributed under the Apache License, Version 2.0, you can also release
your experiment library to the public at will.

.. note::

    A framework to define a sequence of experiment, e.g. closed-loop experiment, is not yet provided with this version.
    You can write own wrapper code until we officially support it.


Package Modules
===============

.. toctree::
    :maxdepth: 1

    main
    framework
    library
    data_processing
    curve_analysis
    calibration_management
    database_service

Experiment Modules
==================

.. toctree::
    :maxdepth: 1
    
    mod_calibration
    mod_characterization
    mod_randomized_benchmarking
    mod_tomography
    mod_quantum_volume