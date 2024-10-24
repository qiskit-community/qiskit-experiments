Use Experiments with Sampler
=============================

Problem
-------

You want to run experiments with a custom :class:`qiskit.primitives.BaseSamplerV2` service. 
A sampler can be instantiated with a backend, session or batch, which allows one to 
run an experiment in different execution modes.

.. note::
    All jobs, by default, run using the :class:`qiskit_ibm_runtime.SamplerV2` class. When calling ``exp.run`` a 
    :class:`qiskit_ibm_runtime.SamplerV2` object will be automatically generated to wrap the specified backend.

Solution
--------

In this example, we will pass in a :class:`qiskit_ibm_runtime.SamplerV2` object to a tomography experiment.

.. note::
    If a sampler object is passed to :meth:`qiskit_experiments.framework.BaseExperiment.run` then the `run options 
    <https://docs.quantum.ibm.com/api/qiskit-ibm-runtime/qiskit_ibm_runtime.options.SamplerExecutionOptionsV2>`_ of the 
    sampler object are used. The execution options set by the experiment are ignored.

.. jupyter-input::

    from qiskit_ibm_runtime import SamplerV2 as Sampler
    from qiskit_experiments.library.tomography import ProcessTomography
    from qiskit import QuantumCircuit

    service = QiskitRuntimeService(channel="ibm_quantum")
    backend = service.backend("ibm_osaka")
    qc = QuantumCircuit(1)
    qc.x(0)

    sampler = Sampler(backed)
    # set the shots in the sampler object
    sampler.options.default_shots = 300
    exp = ProcessTomography(qc)
    # Artificially lower circuits per job, adjust value for your own application
    exp.set_experiment_options(max_circuits=3)
    # pass the sampler into the experiment
    exp_data = exp.run(sampler=sampler)



