Use Experiments with Runtime sessions
=====================================

Problem
-------

You want to run experiments in a `Runtime session
<https://docs.quantum.ibm.com/run/sessions>`_ so that jobs can run in close temporal proximity.

Solution
--------

Use the :class:`~qiskit_ibm_runtime.IBMBackend` object in :mod:`qiskit-ibm-runtime`, which supports sessions.

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
    exp.set_experiment_options(max_circuits=3)
    exp_data = exp.run(backend)
    # this will prevent further jobs from being submitted without terminating current jobs
    backend.close_session()

Note that runtime primitives are not currently supported natively in Qiskit Experiments, so  
the ``backend.run()`` path is required to run experiments.
