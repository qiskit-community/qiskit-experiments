Use Experiments with Runtime sessions
=====================================

Problem
-------

You want to run experiments in a `Runtime session
<https://qiskit.org/ecosystem/ibm-runtime/sessions.html>`_ so that jobs can run in close temporal proximity.

Solution
--------

Use the :class:`~qiskit_ibm_provider.IBMBackend` object in ``qiskit-ibm-provider``, which supports sessions.

In this example, we will set the ``max_circuits`` property to an artificially low value so that the experiment will be
split into multiple jobs that run sequentially in a single session. When running real experiments with a
large number of circuits that can't fit in a single job, it may be helpful to follow this usage pattern:

.. jupyter-input::

    from qiskit_ibm_provider import IBMProvider
    from qiskit_experiments.library.tomography import ProcessTomography
    from qiskit import QuantumCircuit

    provider = IBMProvider()
    backend = provider.get_backend("ibm_nairobi")
    qc = QuantumCircuit(1)
    qc.x(0)

    with backend.open_session() as session:
        exp = ProcessTomography(qc)
        exp.set_experiment_options(max_circuits=3)
        exp_data = exp.run(backend)
        exp_data.block_for_results()
        # Calling cancel because session.close() is not available for qiskit-ibm-provider<=0.7.2.
        # It is safe to call cancel since block_for_results() ensures there are no outstanding jobs 
        # still running that would be canceled.
        session.cancel()

Note that runtime primitives are not currently supported natively in Qiskit Experiments, so  
the ``backend.run()`` path is required to run experiments.
