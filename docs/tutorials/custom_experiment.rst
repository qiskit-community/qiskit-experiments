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

.. Writing a custom experiment
.. ===========================

.. In this tutorial, we'll use what we've learned so far to make a full experiment from
.. the :class:`.BaseExperiment` template.

.. A randomized measurement experiment
.. ===================================


.. This experiment creates a list of copies of an input circuit
.. and randomly samples an N-qubit Paulis to apply to each one before
.. a final N-qubit Z-basis measurement to randomized the expected
.. ideal output bitstring in the measured.

.. The analysis uses the applied Pauli frame of a randomized
.. measurement experiment to de-randomize the measured counts
.. and combine across samples to return a single counts dictionary
.. the original circuit.

.. This has the effect of Pauli-twirling and symmetrizing the
.. measurement readout error. 