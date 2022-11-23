Installation
=============

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
    exp_data = exp.run(backend=backend, seed_simulator=101).block_for_results()

Run and display results:

.. jupyter-execute::

    exp_data = exp.run(backend=backend, seed_simulator=101).block_for_results()

    # Print the result
    display(exp_data.figure(0))
    for result in exp_data.analysis_results():
        print(result)


Setting experiment options
==========================

Often it's insufficient to run an experiment with only the default options. 
There are four types of options one can set for an experiment:

* Run options, for passing to the experiment's ``run()`` method
* Transpile options, for passing to the transpiler
* Experiment options, for the experiment class
* Analysis options, for the analysis class

Setting up a calibrations instance
==================================

Parallel :math:`T_1` experiments on multiple qubits
============================================

To measure :math:`T_1` of multiple qubits in the same experiment, we
create a parallel experiment:

.. jupyter-execute::

    # Create a parallel T1 experiment
    parallel_exp = ParallelExperiment([T1(qubit=i, delays=delays) for i in range(2)])
    parallel_exp.set_transpile_options(scheduling_method='asap')
    parallel_data = parallel_exp.run(backend, seed_simulator=101).block_for_results()
    
    # View result data
    for result in parallel_data.analysis_results():
        print(result)


Viewing sub experiment data
======================

The experiment data returned from a batched experiment also contains
individual experiment data for each sub experiment which can be accessed
using ``child_data``

.. jupyter-execute::

    # Print sub-experiment data
    for i, sub_data in enumerate(parallel_data.child_data()):
        print("Component experiment",i)
        display(sub_data.figure(0))
        for result in sub_data.analysis_results():
            print(result)

Writing a custom experiment
===========================

Qiskit-Experiments is designed to be easily customizable. To create an experiment subclass
based on either the :class:`.BaseExperiment` class or an existing experiment, you should:

- Implement the abstract :meth:`.BaseExperiment.circuits` method.
  This should return a list of ``QuantumCircuit`` objects defining
  the experiment payload.

- Call the :meth:`BaseExperiment.__init__` method during the subclass
  constructor with a list of physical qubits. The length of this list must
  be equal to the number of qubits in each circuit and is used to map these
  circuits to this layout during execution.
  Arguments in the constructor can be overridden so that a subclass can
  be initialized with some experiment configuration.

Optionally the following methods can also be overridden in the subclass to
allow configuring various experiment and execution options

- :meth:`.BaseExperiment._default_experiment_options`
  to set default values for configurable option parameters for the experiment.

- :meth:`.BaseExperiment._default_transpile_options`
  to set custom default values for the ``qiskit.transpile`` used to
  transpile the generated circuits before execution.

- :meth:`.BaseExperiment._default_run_options`
  to set default backend options for running the transpiled circuits on a backend.

- :meth:`.BaseExperiment._default_analysis_options`
  to set default values for configurable options for the experiments analysis class.
  Note that these should generally be set by overriding the :class:`.BaseAnalysis`
  method :meth:`.BaseAnalysis._default_options` instead of this method except in the
  case where the experiment requires different defaults to the used analysis class.

- :meth:`.BaseExperiment._transpiled_circuits`
  to override the default transpilation of circuits before execution.

- :meth:`.BaseExperiment._metadata`
  to add any experiment metadata to the result data.

Furthermore, some characterization and calibration experiments can be run with restless
measurements, i.e. measurements where the qubits are not reset and circuits are executed
immediately after the previous measurement. Here, the :class:`.RestlessMixin` can help
to set the appropriate run options and data processing chain.

Analysis Subclasses
~~~~~~~~~~~~~~~~~~~

To create an analysis subclass, one only needs to implement the abstract
:meth:`.BaseAnalysis._run_analysis` method. This method takes a
:class:`.ExperimentData` container and kwarg analysis options. If any
kwargs are used the :meth:`.BaseAnalysis._default_options` method should be
overriden to define default values for these options.

The :meth:`.BaseAnalysis._run_analysis` method should return a pair
``(results, figures)`` where ``results`` is a list of
:class:`.AnalysisResultData` and ``figures`` is a list of
:class:`matplotlib.figure.Figure`.

The :mod:`qiskit_experiments.data_processing` module contains classes for
building data processor workflows to help with advanced analysis of
experiment data.

==================================
Subclassing an Existing Experiment
==================================

This document will take you step-by-step through the process of subclassing an existing experiment in the Qiskit Experiment module.
The example in this guide focuses on adjusting the FineAmplitude experiment to calibrate on higher order transitions.
However, a similar process can be followed for other experiments.

The FineAmplitude Experiment
============================

The ``FineAmplitude`` calibration experiment repeats N times per gate with a pulse to amplify the under-/over-rotations in the gate to determine the optimal amplitude.
This experiment can be performed for a variety of rotations and subclasses are provided for the :math:`\pi` and :math:`\frac{\pi}{2}` rotations as ``FineXAmplitude`` and ``FineSXAmplitude`` respectively.
These provided subclasses focus on the 0 <-> 1 transition, however this experiment can also be performed for higher order transitions.

Subclassing the Experiment
==========================

Our objective is to create a new class, ``HigherOrderFineXAmplitude``, which calibrates schedules on transitions other than the 0 <-> 1 transition for the :math:`\pi` rotation.
In order to do this, we need to create a subclass, shown below.

.. code-block::
   
    class HigherOrderFineXAmplitude(FineXAmplitude):
        def _pre_circuit(self) -> QuantumCircuit:
            """Return a preparation circuit.
            
            This method can be overridden by subclasses e.g. to calibrate schedules on
            transitions other than the 0 <-> 1 transition.
            """
            circuit = QuantumCircuit(1)

            circuit.x(0)

            if self.experiment_options.add_sx:
                circuit.sx(0)

            if self.experiment_options.sx_schedule is not None:
                sx_schedule = self.experiment_options.sx_schedule
                circuit.add_calibration("sx", (self.physical_qubits[0],), sx_schedule, params=[])
                circuit.barrier()

            return circuit

In this subclass we have overridden the ``_pre_circuit`` method in order to calibrate on higher energy transitions by using an initial X gate to populate the first excited state.

Using the Subclass
==================

Now, we can use our new subclass as we would the original parent class.
Pictured below are the results from following the Fine amplitude calibration tutorial for detecting an over-rotated pulse using our new ``HigherOrderFineXAmplitude`` class in place of the original ``FineXAmplitude`` class.
You can try this for yourself and verify that your results are similar.

.. code-block::
   
   DbAnalysisResultV1
   - name: d_theta
   - value: -0.020710672666906425 ± 0.0012903658449026907
   - χ²: 0.7819653845899581
   - quality: good
   - device_components: ['Q0']
   - verified: False