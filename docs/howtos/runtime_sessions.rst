Use Experiments with Runtime sessions and sampler
=================================================

Problem
-------

You want to run experiments with a custom `SamplerV2
<https://docs.quantum.ibm.com/api/qiskit-ibm-runtime/qiskit_ibm_runtime.SamplerV2>`_ service. 

.. note::
    All jobs, by default, run using the ``SamplerV2`` service. When calling ``exp.run`` a 
    ``SamplerV2`` object will be automatically generated from the specified backend.

Solution
--------

In this example, we will pass in a ``SamplerV2`` object to a tomography experiment.

.. note::
    If a sampler object is passed to ``exp.run`` then the `run options 
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
    exp_data = exp.run(sampler)

Problem
-------

You want to run experiments in a `Runtime session
<https://docs.quantum.ibm.com/run/sessions>`_ so that jobs can run in close temporal proximity.

Solution
--------

.. note::
    This guide requires :external+qiskit_ibm_runtime:doc:`qiskit-ibm-runtime <index>` version 0.15 and up, which can be installed with ``python -m pip install qiskit-ibm-runtime``.
    For how to migrate from the older ``qiskit-ibm-provider`` to :external+qiskit_ibm_runtime:doc:`qiskit-ibm-runtime <index>`,
    consult the `migration guide <https://docs.quantum.ibm.com/api/migration-guides/qiskit-runtime-from-provider>`_.\

Use the :class:`~qiskit_ibm_runtime.IBMBackend` object in :external+qiskit_ibm_runtime:doc:`index`, which supports sessions.

In this example, we will set the ``max_circuits`` property to an artificially low value so that the experiment will be
split into multiple jobs that run sequentially in a single session. When running real experiments with a
large number of circuits that can't fit in a single job, it may be helpful to follow this usage pattern:

.. jupyter-input::

    from qiskit_ibm_runtime import QiskitRuntimeService
    from qiskit_experiments.library.tomography import ProcessTomography
    from qiskit import QuantumCircuit

    service = QiskitRuntimeService(channel="ibm_quantum")
    backend = service.backend("ibm_osaka")
    qc = QuantumCircuit(1)
    qc.x(0)

    backend.open_session()
    exp = ProcessTomography(qc)
    # Artificially lower circuits per job, adjust value for your own application
    exp.set_experiment_options(max_circuits=3)
    exp_data = exp.run(backend)
    # This will prevent further jobs from being submitted without terminating current jobs
    backend.close_session()


