Writing a custom experiment
===========================

Qiskit Experiments is designed to be easily customizable. To create an experiment subclass
based on either the :class:`.BaseExperiment` class or an existing experiment, you should:

- Implement the abstract :meth:`.BaseExperiment.circuits` method.
  This should return a list of :class:`~qiskit.QuantumCircuit` objects defining
  the experiment payload.

- Call the :meth:`.BaseExperiment.__init__` method during the subclass
  constructor with a list of physical qubits. The length of this list must
  be equal to the number of qubits in each circuit and is used to map these
  circuits to this layout during execution.
  Arguments in the constructor can be overridden so that a subclass can
  be initialized with some experiment configuration.

Optionally, to allow configuring experiment and execution options, you can override:

- :meth:`.BaseExperiment._default_experiment_options`
  to set default values for configurable option parameters for the experiment.

- :meth:`.BaseExperiment._default_transpile_options`
  to set custom default values for the :func:`qiskit.compiler.transpile` method used to
  transpile the generated circuits before execution.

- :meth:`.BaseExperiment._default_run_options`
  to set default backend options for running the transpiled circuits on a backend.

- :meth:`.BaseAnalysis._default_options`
  to set default values for configurable options for the experiment's analysis class.

- :meth:`.BaseExperiment._transpiled_circuits`
  to override the default transpilation of circuits before execution.

- :meth:`.BaseExperiment._metadata`
  to add any experiment metadata to the result data.

Furthermore, some characterization and calibration experiments can be run with restless
measurements, i.e. measurements where the qubits are not reset and circuits are executed
immediately after the previous measurement. Here, the :class:`.RestlessMixin` class
can help
to set the appropriate run options and data processing chain.

Analysis Subclasses
-------------------

To create an analysis subclass, one only needs to implement the abstract
:meth:`.BaseAnalysis._run_analysis` method. This method takes an
:class:`.ExperimentData` container and kwarg analysis options. If any
kwargs are used, the :meth:`.BaseAnalysis._default_options` method should be
overriden to define default values for these options.

The :meth:`.BaseAnalysis._run_analysis` method should return a pair
``(results, figures)``, where ``results`` is a list of
:class:`.AnalysisResultData` and ``figures`` is a list of
:class:`matplotlib.figure.Figure`.

The :mod:`~qiskit_experiments.data_processing` module contains classes for
building data processor workflows to help with advanced analysis of
experiment data.

Subclassing an Existing Experiment
----------------------------------

Let's walk through the process of subclassing an existing experiment in the Qiskit
Experiments library.

The FineAmplitude Experiment
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The ``FineAmplitude`` calibration experiment optimizes gate amplitude by repeating the 
gate pulse N times, hence amplifying the under- or over-rotations.
This experiment can be performed for a variety of rotations, and subclasses are 
provided for the :math:`\pi` and :math:`\frac{\pi}{2}` rotations as ``FineXAmplitude`` and ``FineSXAmplitude`` respectively.
These provided subclasses focus on the 0 <-> 1 transition, however, this experiment can also be performed for higher order transitions.

Our objective is to create a new class, ``HigherOrderFineXAmplitude``, which calibrates 
schedules on transitions other than the 0 <-> 1 transition for the :math:`\pi` rotation.
In order to do this, we need to create a subclass as shown below.

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

In this subclass we have overridden the ``_pre_circuit`` method in order to calibrate 
on higher energy transitions by using an initial X gate to populate the first excited state.

Using the Subclass
------------------

Now, we can use our new subclass as we would the original parent class.
Shown below are results from following the :ref:`fine-amplitude-cal` tutorial
for detecting an over-rotated pulse using our new 
``HigherOrderFineXAmplitude`` class in place of the original 
:class:`.FineXAmplitude` class.
You can try this for yourself and verify that your results are similar.

.. code-block::
   
   DbAnalysisResultV1
   - name: d_theta
   - value: -0.020710672666906425 ± 0.0012903658449026907
   - χ²: 0.7819653845899581
   - quality: good
   - device_components: ['Q0']
   - verified: False

Writing a new experiment
------------------------

Now we'll use what we've learned so far to make an entirely new experiment using
the :class:`.BaseExperiment` template.

A randomized measurement experiment
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Our goal is to write an experiment that symmetrizes the measurement readout error
of an input circuit, which is especially useful in systems where readout has an unknown
and potentially large bias. To do so, our experiment should create a list of copies of an input circuit
and randomly sample an :math:`N`-qubit Pauli to apply to each one before
a final :math:`N`-qubit :math:`Z`-basis measurement to randomize the expected
ideal output bitstring in the measurement.

The analysis uses the applied Pauli frame of a randomized
measurement experiment to de-randomize the measured counts
and combine across samples to return a single counts dictionary for
the original circuit. This has the effect of Pauli-twirling and symmetrizing the
measurement readout error.

To start, we must write our own ``__init__()`` method to take as input the circuit that
we want to twirl on. We also want to give the user the option to specify which
physical qubits to run the circuit over, and which qubits to measure over. If the user
doesn't specify these options, we default to the list of qubits starting with 0 with
the length of the number of qubits in the circuit for both.

.. code-block:: python

  from qiskit import QuantumCircuit
  from qiskit_experiments.framework import BaseExperiment

  class RandomizedMeasurement(BaseExperiment):
    """Randomized measurement experiment.
    """

    def __init__(
        self,
        circuit,
        measured_qubits=None,
        physical_qubits=None,
        backend=None,
        **experiment_options
    ):
        """Basic randomized Z-basis measurement experiment via a Pauli frame transformation
        
        Note this will just append a new set of measurements at the end of a circuit.
        A more advanced version of this experiment would be use a transpiler pass to
        replace all existing measurements in a circuit with randomized measurements.
        """
        if physical_qubits is None:
            physical_qubits = tuple(range(circuit.num_qubits))
        if measured_qubits is None:
            measured_qubits = tuple(range(circuit.num_qubits))
        
        # Initialize BaseExperiment
        analysis = RandomizedMeasurementAnalysis()
        super().__init__(physical_qubits, analysis=analysis, backend=backend)
        
        # Add experiment properties
        self._circuit = circuit        
        self._measured_qubits = measured_qubits
        
        # Set any init optinos
        self.set_experiment_options(**experiment_options)

Notice that when we called ``super().__init__``, we provided the list of physical qubits,
the name of our analysis class, and the backend, which is optionally specified by the
user at this stage.

Now we consider default experiment options. Because randomness is involved,
it is good practice to allow the user to set a seed. We would also like the user to 
be able to set how many repetitions of the circuit to run:

.. code-block:: python

    @classmethod
    def _default_experiment_options(cls):
        options = super()._default_experiment_options()
        options.num_samples = "default"
        options.seed = None
        return options

Now we write the ``circuits()`` method. We need to take the input circuit in ``self._circuit``
and add our random Paulis as well as measurement at the end.

.. code-block:: python

    from qiskit.quantum_info import random_pauli_list
    from numpy.random import default_rng, Generator

    def circuits(self):
        # Number of classical bits for original circuit and added measurements
        circ_nc = self._circuit.num_clbits
        meas_nc = len(self._measured_qubits)
        circ_qubits = list(range(self.num_qubits))
        circ_clbits = list(range(circ_nc))
        meas_qubits = self._measured_qubits
        meas_clbits = list(range(circ_nc, circ_nc + meas_nc))

        # Get number of samples from options
        num_samples = self.experiment_options.num_samples
        if num_samples == "default":
            num_samples = 2 ** self.num_qubits
        
        # Get rng seed
        seed = self.experiment_options.seed
        if isinstance(seed, Generator):
            rng = seed
        else:
            rng = default_rng(seed)

We use the :func:`~qiskit.quantum_info.random_pauli_list` function from the quantum 
info module to generate random Paulis. This returns ``num_samples`` Paulis, each 
across ``meas_nc`` qubits.

.. code-block:: python

        # Sample Paulis this might have duplicates, but we don't really
        # have any easy way of running different number of shots per circuit
        # so we just run repeat circuits multiple times
        paulis = random_pauli_list(meas_nc, size=num_samples, phase=False, seed=rng)

        # Construct circuits
        circuits = []
        orig_metadata = self._circuit.metadata or {}
        for pauli in paulis:
            name = f"{self._circuit.name}_{str(pauli)}"
            circ = QuantumCircuit(
                self.num_qubits, circ_nc + meas_nc,
                name=name
            )
            # Append original circuit
            circ.compose(
                self._circuit, circ_qubits, circ_clbits, inplace=True
            )

            # Add Pauli frame
            circ.compose(pauli, meas_qubits, inplace=True)

            # Add final Measurement
            circ.measure(meas_qubits, meas_clbits)

Let's look at what the :meth:`~.qiskit.circuit.QuantumCircuit.compose` does here.

We need to tell our analysis class how to restore the results of the original circuit.
To do so, we add metadata to each of our circuits.

.. code-block:: python

            circ.metadata = orig_metadata.copy()
            circ.metadata["rm_bits"] = meas_clbits
            circ.metadata["rm_frame"] = str(pauli)
            circ.metadata["rm_sig"] = pauli.x.astype(int).tolist()

            circuits.append(circ)

        return circuits

And the corresponding analysis class:

.. code-block:: python

  from qiskit_experiments.framework import BaseAnalysis, AnalysisResultData

  class RandomizedMeasurementAnalysis(BaseAnalysis):
      """Analysis for randomized measurement experiment."""

      # Helper dict to swap a clbit value
      _swap_bit = {"0": "1", "1": "0"}

      def _run_analysis(self, experiment_data):
          
          combined_counts = {}
          for datum in experiment_data.data():
              # Get counts
              counts = datum["counts"]
              num_bits = len(next(iter(counts)))
              
              # Get metadata
              metadata = datum["metadata"]
              clbits = metadata["rm_bits"]
              sig = metadata["rm_sig"]

              # Construct full signature
              full_sig = num_bits * [0]
              for bit, val in zip(clbits, sig):
                  full_sig[bit] = val
              
              # Combine dicts
              for key, val in counts.items():
                  bitstring = self._swap_bitstring(key, full_sig)
                  if bitstring in combined_counts:
                      combined_counts[bitstring] += val
                  else:
                      combined_counts[bitstring] = val
                      
          
          result = AnalysisResultData("counts", combined_counts)
          return [result], []

      @classmethod
      def _swap_bitstring(cls, bitstring, sig):
          """Swap a bitstring based signature to flip bits at."""
          # This is very inefficient but demonstrates the basic idea
          # Really should do with bitwise operations of integer counts rep
          return "".join(reversed(
              [cls._swap_bit[b] if sig[- 1 - i] else b for i, b in enumerate(bitstring)]
          ))

.. jupyter-execute::
  :hide-code:
  :hide-output:

  # this is the actual code that defines the experiment so the code below can work

  from numpy.random import default_rng, Generator
  from qiskit import QuantumCircuit
  from qiskit_experiments.framework import BaseExperiment
  from qiskit.quantum_info import random_pauli_list

  class RandomizedMeasurement(BaseExperiment):
    """Randomized measurement experiment.
    """

    def __init__(
        self,
        circuit,
        measured_qubits=None,
        physical_qubits=None,
        backend=None,
        **experiment_options
    ):
        """Basic randomize Z-basis measurement via a Pauli frame transformation

        Note this will just append a new set of measurment at the end of a circuit.
        A more advanced version of this experiment would be use a transpiler pass to
        replace all exisiting measurements in a circuit with randomized measurements.
        """
        if physical_qubits is None:
            physical_qubits = tuple(range(circuit.num_qubits))
        if measured_qubits is None:
            measured_qubits = tuple(range(circuit.num_qubits))

        # Initialize BaseExperiment
        analysis = RandomizedMeasurementAnalysis()
        super().__init__(physical_qubits, analysis=analysis, backend=backend)

        # Add experiment properties
        self._circuit = circuit
        self._measured_qubits = measured_qubits

        # Set any init optinos
        self.set_experiment_options(**experiment_options)

    @classmethod
    def _default_experiment_options(cls):
        options = super()._default_experiment_options()
        options.num_samples = "default"
        options.seed = None
        return options

    def circuits(self):
        # Number of classical bits for original circuit and added measurements
        circ_nc = self._circuit.num_clbits
        meas_nc = len(self._measured_qubits)
        circ_qubits = list(range(self.num_qubits))
        circ_clbits = list(range(circ_nc))
        meas_qubits = self._measured_qubits
        meas_clbits = list(range(circ_nc, circ_nc + meas_nc))

        # Get number of samples from options
        num_samples = self.experiment_options.num_samples
        if num_samples == "default":
            num_samples = 2 ** self.num_qubits

        # Get rng seed
        seed = self.experiment_options.seed
        if isinstance(seed, Generator):
            rng = seed
        else:
            rng = default_rng(seed)

        # Sample Paulis this might have duplicates, but we don't really
        # have any easy way of running different number of shots per circuit
        # so we just run repeat circuits multiple times
        paulis = random_pauli_list(meas_nc, size=num_samples, phase=False, seed=rng)

        # Construct circuits
        circuits = []
        orig_metadata = self._circuit.metadata or {}
        for pauli in paulis:
            name = f"{self._circuit.name}_{str(pauli)}"
            circ = QuantumCircuit(
                self.num_qubits, circ_nc + meas_nc,
                name=name
            )
            # Append original circuit
            circ.compose(
                self._circuit, circ_qubits, circ_clbits, inplace=True
            )

            # Add Pauli frame
            circ.compose(pauli, meas_qubits, inplace=True)

            # Add final Measurement
            circ.measure(meas_qubits, meas_clbits)

            # Add metadata
            circ.metadata = orig_metadata.copy()
            circ.metadata["rm_bits"] = meas_clbits
            circ.metadata["rm_frame"] = str(pauli)
            circ.metadata["rm_sig"] = pauli.x.astype(int).tolist()

            circuits.append(circ)

        return circuits




  from qiskit_experiments.framework import BaseAnalysis, AnalysisResultData

  class RandomizedMeasurementAnalysis(BaseAnalysis):
      """Analysis for randomized measurement experiment."""

      # Helper dict to swap a clbit value
      _swap_bit = {"0": "1", "1": "0"}

      def _run_analysis(self, experiment_data):
          
          combined_counts = {}
          for datum in experiment_data.data():
              # Get counts
              counts = datum["counts"]
              num_bits = len(next(iter(counts)))
              
              # Get metadata
              metadata = datum["metadata"]
              clbits = metadata["rm_bits"]
              sig = metadata["rm_sig"]

              # Construct full signature
              full_sig = num_bits * [0]
              for bit, val in zip(clbits, sig):
                  full_sig[bit] = val
              
              # Combine dicts
              for key, val in counts.items():
                  bitstring = self._swap_bitstring(key, full_sig)
                  if bitstring in combined_counts:
                      combined_counts[bitstring] += val
                  else:
                      combined_counts[bitstring] = val
                      
          
          result = AnalysisResultData("counts", combined_counts)
          return [result], []

      @classmethod
      def _swap_bitstring(cls, bitstring, sig):
          """Swap a bitstring based signature to flip bits at."""
          # This is very inefficient but demonstrates the basic idea
          # Really should do with bitwise operations of integer counts rep
          return "".join(reversed(
              [cls._swap_bit[b] if sig[- 1 - i] else b for i, b in enumerate(bitstring)]
          ))


To test our code, we first simulate a noisy backend with asymmetric readout error in Aer:

.. jupyter-execute::

  from qiskit.providers.aer import AerSimulator, noise

  backend_ideal = AerSimulator()

  # Backend with asymetric readout error
  p0g1 = 0.3
  p1g0 = 0.05
  noise_model = noise.NoiseModel()
  noise_model.add_all_qubit_readout_error([[1 - p1g0, p1g0], [p0g1, 1 - p0g1]])
  noise_backend = AerSimulator(noise_model=noise_model)

We now run the experiment with a GHZ circuit on an ideal backend:

.. jupyter-execute::

  # GHZ Circuit
  nq = 4
  qc = QuantumCircuit(nq)
  qc.h(0)
  for i in range(1, nq):
      qc.cx(i-1, i)

  # Experiment parameters
  total_shots = 100000
  num_samples = 50
  shots = total_shots // num_samples

  # Run ideal randomized meas experiment
  exp = RandomizedMeasurement(qc, num_samples=num_samples)
  expdata_ideal = exp.run(AerSimulator(), shots=shots)
  counts_ideal = expdata_ideal.analysis_results("counts").value
  print(counts_ideal)

Now we repeat the experiment on the backend with readout error:

.. jupyter-execute::

  # Run noisy randomized meas experiment with readout error
  expdata_noise = exp.run(noise_backend, shots=shots)
  counts_noise = expdata_noise.analysis_results("counts").value

  # Run noisy direct simulation of original circuit without randomization
  meas_circ = qc.copy()
  meas_circ.measure_all()
  result = noise_backend.run(meas_circ, shots=total_shots).result()
  counts_direct = result.get_counts(0)

  from qiskit.visualization import plot_histogram

  # Plot counts, ideally randomized one should be more symmetric in noise
  # than direct one with asymmetric readout error
  plot_histogram([counts_ideal, counts_direct, counts_noise],
                legend=["Ideal",
                        "Asymmetric meas error (Direct)",
                        "Asymmetric meas error (Randomized)"])

We see that the direct asymmetric measurement is symmetrized by the application of randomized measurement.