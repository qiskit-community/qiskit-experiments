===============
Getting Started
===============

Installation
============

Qiskit Experiments is built on top of Qiskit, so we recommend that you first install
Qiskit following its :external+qiskit:doc:`installation guide <getting_started>`. Qiskit
Experiments supports the same platforms and Python versions (currently 3.7+) as Qiskit
itself.

Qiskit Experiments releases can be installed via the Python package manager ``pip``:

.. jupyter-input::

    python -m pip install qiskit-experiments

If you want to run the most up-to-date version instead (may not be stable), you can
install the latest main branch:

.. jupyter-input::

    python -m pip install git+https://github.com/Qiskit/qiskit-experiments.git

If you want to develop the package, you can install Qiskit Experiments from source by
cloning the repository:

.. jupyter-input::

    git clone https://github.com/Qiskit/qiskit-experiments.git
    python -m pip install -e qiskit-experiments

The ``-e`` option will keep your installed package up to date as you make or pull new
changes.

Running your first experiment
=============================

Let's run a :class:`.T1` Experiment, which estimates the characteristic relaxation time
of a qubit from the excited state to the ground state, also known as :math:`T_1`, by
measuring the excited state population after varying delays. First, we have to import
the experiment from the Qiskit Experiments library:

.. jupyter-execute::

    from qiskit_experiments.library import T1

Experiments must be run on a backend. We're going to use a simulator,
:class:`~qiskit.providers.fake_provider.FakeVigo`, for this example, but you can use any
IBM backend, real or simulated, that you can access through Qiskit.

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

All experiments require a ``physical_qubits`` parameter as input that specifies which
physical qubit or qubits the circuits will be executed on. The qubits must be given as a
Python sequence (usually a tuple or a list).

.. note::
    Since 0.5.0, using ``qubits`` instead of ``physical_qubits`` or specifying an 
    integer qubit index instead of a one-element sequence for a single-qubit experiment
    is deprecated.

In addition, the :math:`T_1` experiment has
a second required parameter, ``delays``, which is a list of times in seconds at which to
measure the excited state population. In this example, we'll run the :math:`T_1`
experiment on qubit 0, and use the ``t1`` backend property of this qubit to give us a
good estimate for the sweep range of the delays.

.. jupyter-execute::

    qubit0_t1 = backend.properties().t1(0)

    delays = np.arange(1e-6, 3 * qubit0_t1, 3e-5)
    exp = T1(physical_qubits=(0,), delays=delays)

The circuits encapsulated by the experiment can be accessed using the experiment's
:meth:`~.BaseExperiment.circuits` method, which returns a list of circuits that can be
run on a backend. Let's print the range of delay times we're sweeping over and draw the
first and last circuits for our :math:`T_1` experiment:

.. jupyter-execute::

    print(delays)
    exp.circuits()[0].draw(output='mpl')

.. jupyter-execute::

    exp.circuits()[-1].draw(output='mpl')

As expected, the delay block spans the full range of time values that we specified.

The :class:`.ExperimentData` class
==================================

After instantiating the experiment, we run the experiment by calling
:meth:`~.BaseExperiment.run` with our backend of choice. This transpiles our experiment
circuits then packages them into jobs that are run on the backend.

.. note::
    See the how-tos for :doc:`customizing job splitting </howtos/job_splitting>` when
    running an experiment. 

This statement returns the :class:`.ExperimentData` class containing the results of the
experiment, so it's crucial that we assign the output to a data variable. We could have
also provided the backend at the instantiation of the experiment, but specifying the
backend at run time allows us to run the same exact experiment on different backends
should we choose to do so.

.. jupyter-execute::

    exp_data = exp.run(backend=backend).block_for_results()

The :meth:`~.ExperimentData.block_for_results` method is optional and is used to block
execution of subsequent code until the experiment has fully completed execution and
analysis. If

.. jupyter-input::
    
    exp_data = exp.run(backend=backend)

is executed instead, the statement will finish running as soon as the jobs are
submitted, but the analysis callback won't populate ``exp_data`` with results until the
entire process has finished. In this case, there are two useful methods in the
:class:`.ExperimentData`, :meth:`~.ExperimentData.job_status` and
:meth:`~.ExperimentData.analysis_status`, that return the current status of the job and
analysis, respectively:

.. jupyter-execute::

    print(exp_data.job_status())
    print(exp_data.analysis_status())

Once the analysis is complete, figures are retrieved using the
:meth:`~.ExperimentData.figure` method. See the :doc:`visualization module
<visualization>` tutorial on how to customize figures for an experiment. For our
:math:`T_1` experiment, we have a single figure showing the raw data and fit to the
exponential decay model of the :math:`T_1` experiment:

.. jupyter-execute::

    display(exp_data.figure(0))

The fit results and associated parameters are accessed with
:meth:`~.ExperimentData.analysis_results`:

.. jupyter-execute::

    for result in exp_data.analysis_results():
        print(result)

Results can be indexed numerically (starting from 0) or using their name.

.. note::
    See the :meth:`~.ExperimentData.analysis_results` API documentation for more 
    advanced usage patterns to access subsets of analysis results.

Each analysis
result value is a ``UFloat`` object from the ``uncertainties`` package. The nominal
value and standard deviation of each value can be accessed as follows:

.. jupyter-execute::

    print(exp_data.analysis_results("T1").value.nominal_value)
    print(exp_data.analysis_results("T1").value.std_dev)

For further documentation on how to work with UFloats, consult the ``uncertainties``
:external+uncertainties:doc:`user_guide`.

Raw circuit output data and its associated metadata can be accessed with the
:meth:`~.ExperimentData.data` property. Data is indexed by the circuit it corresponds
to. Depending on the measurement level set in the experiment, the raw data will either
be in the key ``counts`` (level 2) or ``memory`` (level 1 IQ data).

.. note::
    See the :doc:`data processor tutorial <data_processor>` for more 
    information on level 1 and level 2 data.

Circuit metadata contains information set by the experiment on a circuit-by-circuit
basis; ``xval`` is used by the analysis to extract the x value for each circuit when
fitting the data.

.. jupyter-execute::

    print(exp_data.data(0))

Experiments also have global associated metadata accessed by the
:meth:`~.ExperimentData.metadata` property.

.. jupyter-execute::

    print(exp_data.metadata)

The actual backend jobs that were executed for the experiment can be accessed with the
:meth:`~.ExperimentData.jobs` method.

.. note::
    See the how-tos for :doc:`rerunning the analysis </howtos/rerun_analysis>`
    for an existing experiment that finished execution.

Setting options for your experiment
===================================

It's often insufficient to run an experiment with only its default options. There are
four types of options one can set for an experiment:

Run options
-----------

These options are passed to the experiment's :meth:`~.BaseExperiment.run` method and
then to the ``run()`` method of your specified backend. Any run option that your backend
supports can be set:

.. jupyter-input::

  exp.set_run_options(shots=1000,
                      meas_level=MeasLevel.CLASSIFIED,
                      meas_return="avg")

Consult the documentation of :func:`qiskit.execute_function.execute` or the run method
of your specific backend type for valid options.

Transpile options
-----------------
These options are passed to the Terra transpiler to transpile the experiment circuits
before execution:

.. jupyter-input::

  exp.set_transpile_options(scheduling_method='asap',
                            optimization_level=3,
                            basis_gates=["x", "sx", "rz"])

Consult the documentation of :func:`qiskit.compiler.transpile` for valid options.

Experiment options
------------------
These options are unique to each experiment class. Many experiment options can be set
upon experiment instantiation, but can also be explicitly set via
:meth:`~.BaseExperiment.set_experiment_options`:

.. jupyter-input::

    exp = T1(physical_qubits=(i,), delays=delays)
    exp.set_experiment_options(delays=new_delays)

Consult the :doc:`API documentation </apidocs/index>` for the options of each experiment
class.

Analysis options
----------------

These options are unique to each analysis class. Unlike the other options, analyis
options are not directly set via the experiment object but use instead a method of the
associated ``analysis``:

.. jupyter-execute::

    from qiskit_experiments.library import StandardRB

    exp = StandardRB(physical_qubits=(0,),
                    lengths=list(range(1, 300, 30)),
                    seed=123,
                    backend=backend)
    exp.analysis.set_options(gate_error_ratio=None)

Consult the :doc:`API documentation </apidocs/index>` for the options of each
experiment's analysis class.

Running experiments on multiple qubits
======================================

To run experiments across many qubits of the same device, we use **composite
experiments**. A composite experiment is a parent object that contains one or more child
experiments, which may themselves be composite. There are two core types of composite
experiments:

* **Parallel experiments** run across qubits simultaneously as set by the user. The
  circuits of child experiments are combined into new circuits that map circuit gates
  onto qubits in parallel. Therefore, the circuits in child experiments *cannot* overlap
  in the ``physical_qubits`` parameter. The marginalization of measurement data for
  analysis of each child experiment is handled automatically. 
* **Batch experiments** run consecutively in time. These child circuits *can* overlap in
  qubits used.

Using parallel experiments, we can measure the :math:`T_1` of one qubit while doing a
standard Randomized Benchmarking :class:`.StandardRB` experiment on other qubits
simultaneously on the same device:

.. jupyter-execute::

    from qiskit_experiments.framework import ParallelExperiment

    child_exp1 = T1(physical_qubits=(2,), delays=delays)
    child_exp2 = StandardRB(physical_qubits=(3,1), lengths=np.arange(1,100,10), num_samples=2)
    parallel_exp = ParallelExperiment([child_exp1, child_exp2])

Note that when the transpile and run options are set for a composite experiment, the
child experiments's options are also set to the same options recursively. Let's examine
how the parallel experiment is constructed by visualizing child and parent circuits. The
child experiments can be accessed via the
:meth:`~.ParallelExperiment.component_experiment` method, which indexes from zero:

.. jupyter-execute::

    parallel_exp.component_experiment(0).circuits()[0].draw(output='mpl')

.. jupyter-execute::

    parallel_exp.component_experiment(1).circuits()[0].draw(output='mpl')

The circuits of all experiments assume they're acting on virtual qubits starting from
index 0. In the case of a parallel experiment, the child experiment 
circuits are composed together and then reassigned virtual qubit indices:

.. jupyter-execute::

    parallel_exp.circuits()[0].draw(output='mpl')

During experiment transpilation, a mapping is performed to place these circuits on the
physical layout. We can see its effects by looking at the transpiled
circuit, which is accessed via the internal method ``_transpiled_circuits()``. After
transpilation, the :class:`.T1` experiment is correctly placed on physical qubit 2 
and the :class:`.StandardRB` experiment's gates are on physical qubits 3 and 1.

.. jupyter-execute::

    parallel_exp._transpiled_circuits()[0].draw(output='mpl')

:class:`.ParallelExperiment` and :class:`.BatchExperiment` classes can also be nested
arbitrarily to make complex composite experiments.

.. figure:: ./images/compositeexperiments.png
    :align: center

Viewing child experiment data
-----------------------------

The experiment data returned from a composite experiment contains individual analysis
results for each child experiment that can be accessed using
:meth:`~.ExperimentData.child_data`. By default, the parent data object does not contain
analysis results.

.. jupyter-execute::

    parallel_data = parallel_exp.run(backend, seed_simulator=101).block_for_results()

    for i, sub_data in enumerate(parallel_data.child_data()):
        print("Component experiment",i)
        display(sub_data.figure(0))
        for result in sub_data.analysis_results():
            print(result)

If you want the parent data object to contain the analysis results instead, you can set
the ``flatten_results`` flag to true to flatten the results of all component experiments
into one level:

.. jupyter-execute::

    parallel_exp = ParallelExperiment(
        [T1(physical_qubits=(i,), delays=delays) for i in range(2)], flatten_results=True
    )
    parallel_data = parallel_exp.run(backend, seed_simulator=101).block_for_results()

    for result in parallel_data.analysis_results():
        print(result)