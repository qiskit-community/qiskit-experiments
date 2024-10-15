Quantum Volume
==============

Quantum Volume (QV) is a single-number metric that can be measured using
a concrete protocol on near-term quantum computers of modest size. The
QV method quantifies the largest random circuit of equal width and depth
that the computer successfully implements. Quantum computing systems
with high-fidelity operations, high connectivity, large calibrated gate
sets, and circuit rewriting toolchains are expected to have higher
quantum volumes. See the `Qiskit
Textbook <https://github.com/Qiskit/textbook/blob/main/notebooks/quantum-hardware/measuring-quantum-volume.ipynb>`__
for an explanation on the QV method, which is described in Refs. [1]_ [2]_.

The Quantum Volume is determined by the largest successful circuit depth
:math:`d_{max}`, and equals to :math:`2^{d_{max}}`. In this experiment,
we generate QV circuits using the :class:`qiskit.circuit.library.QuantumVolume` class
on :math:`d` qubits, which contain :math:`d` layers, where each layer
consists of random 2-qubit unitary gates from :math:`SU(4)`, followed by
a random permutation on the :math:`d` qubit. Then these circuits run on
the quantum backend and on an ideal simulator (either :class:`qiskit_aer.AerSimulator`
or :class:`qiskit.quantum_info.Statevector`).

A depth :math:`d` QV circuit is successful if it has `mean heavy-output
probability` > 2/3 with confidence level > 0.977 (corresponding to
z_value = 2), and at least 100 trials have been ran.

.. note::
    This tutorial requires the :external+qiskit_aer:doc:`qiskit-aer <index>` and :external+qiskit_ibm_runtime:doc:`qiskit-ibm-runtime <index>`
    packages to run simulations.  You can install them with ``python -m pip
    install qiskit-aer qiskit-ibm-runtime``.

.. jupyter-execute::
    :hide-code:

    # Temporary workaround for missing support in Qiskit and qiskit-ibm-runtime
    from qiskit_experiments.test.patching import patch_sampler_test_support
    patch_sampler_test_support()

.. jupyter-execute::

    from qiskit_experiments.framework import BatchExperiment
    from qiskit_experiments.library import QuantumVolume
    
    # For simulation
    from qiskit_aer import AerSimulator
    from qiskit_ibm_runtime.fake_provider import FakeSydneyV2
    
    backend = AerSimulator.from_backend(FakeSydneyV2())

QV experiment
-------------

To run the QV experiment we need need to provide the following QV
parameters, in order to generate the QV circuits and run them on a
backend and on an ideal simulator:

-  ``qubits``: The number of qubits or list of physical qubits for the
   experiment.

-  ``trials``: The number of trials to run the quantum volume circuit
   (the default is 100).

-  ``seed``: Seed or generator object for random number generation. If
   ``None`` then ``default_rng`` will be used.

-  ``simulation_backend``: The simulator backend to use to generate the expected
   results. the simulator must have a ``save_probabilities`` method. If None,
   :class:`~qiskit_aer.AerSimulator` will be used (in case
   :class:`~qiskit_aer.AerSimulator` is not installed,
   :class:`~qiskit.quantum_info.Statevector` will be used).

**Note:** In some cases, 100 trials are not enough to obtain a QV
greater than 1 for the specified number of qubits. In this case, adding
more trials may reduce the error bars to allow passing the threshold.

The analysis results of the QV Experiment are:

-  The mean heavy-output probabilities (HOP) and standard deviation

-  The calculated quantum volume, which will be None if the experiment
   does not pass the threshold

Extra data included in the analysis results includes

-  The heavy HOPs for each trial

-  Confidence level (should be greater than 0.977)

-  The number of trials and depth of the QV circuits

-  Whether the QV circuit was successful

.. jupyter-execute::

    qubits = tuple(range(4)) # Can use specific qubits. for example [2, 4, 7, 10]
    
    qv_exp = QuantumVolume(qubits, seed=42)
    # Transpile options like optimization_level affect only the real device run and not the simulation run
    # Run options affect both simulation and real device runs
    qv_exp.set_transpile_options(optimization_level=3)
    
    # Run experiment
    expdata = qv_exp.run(backend).block_for_results()

.. jupyter-execute::

    # View result data
    display(expdata.figure(0))
    
    for result in expdata.analysis_results():
        print(result)


.. jupyter-execute::

    # Print extra data
    for result in expdata.analysis_results():
        print(f"\n{result.name} extra:")
        for key, val in result.extra.items():
            print(f"- {key}: {val}")


Adding trials
-------------

Adding more trials may reduce the error bars to allow passing the
threshold (beside the error bars - QV experiment must have at least 100
trials to be considered successful). In case you want to add less than
100 additional trials, you can modify the amount of trials added before
re-running the experiment.

.. jupyter-execute::

    qv_exp.set_experiment_options(trials=60)
    expdata2 = qv_exp.run(backend, analysis=None).block_for_results()
    expdata2.add_data(expdata.data())
    qv_exp.analysis.run(expdata2).block_for_results()
    
    # View result data
    display(expdata2.figure(0))
    for result in expdata2.analysis_results():
        print(result)


Calculating Quantum Volume using a batch experiment
---------------------------------------------------

Run the QV experiment with an increasing number of qubits to check what
is the maximum Quantum Volume for the specific device. To reach the real
system’s Quantum Volume, one must run more trials and additional
enhancements might be required (See Ref. [2]_ for details).

.. jupyter-execute::

    exps = [QuantumVolume(tuple(range(i)), trials=200) for i in range(3, 6)]

    batch_exp = BatchExperiment(exps)
    batch_exp.set_transpile_options(optimization_level=3)
    
    # Run
    batch_expdata = batch_exp.run(backend).block_for_results()

Extracting the maximum Quantum Volume.

.. jupyter-execute::

    qv_values = [
        batch_expdata.analysis_results("quantum_volume")[i].value
        for i in range(batch_exp.num_experiments)
    ]
    
    print(f"Max quantum volume is: {max(qv_values)}")


.. jupyter-execute::

    for i in range(batch_exp.num_experiments):
        print(f"\nComponent experiment {i}")
        display(batch_expdata.figure(i))
    for result in batch_expdata.analysis_results():
        print(result)

References
----------

.. [1] Andrew W. Cross, Lev S. Bishop, Sarah Sheldon, Paul D. Nation, and
    Jay M. Gambetta, Validating quantum computers using randomized model
    circuits, Phys. Rev. A 100, 032328 (2019).
    https://arxiv.org/pdf/1811.12926

.. [2] Petar Jurcevic et. al. Demonstration of quantum volume 64 on
    a superconducting quantum computing system,
    https://arxiv.org/pdf/2008.08571

See also
--------

* API documentation: :mod:`~qiskit_experiments.library.quantum_volume`
* Qiskit Textbook: `Measuring Quantum Volume <https://github.com/Qiskit/textbook/blob/main/notebooks/quantum-hardware/measuring-quantum-volume.ipynb>`__

