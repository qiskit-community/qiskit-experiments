Use Experiments with Runtime sessions
=====================================

Problem
-------

You want to run experiments in a `Runtime session
<https://qiskit.org/ecosystem/ibm-runtime/sessions.html>`_ so that jobs can run in close temporal proximity.

Solution
--------

There are two pathways currently supported:

1. Use the :class:`~qiskit_ibm_provider.IBMBackend` object in ``qiskit-ibm-provider``, which supports sessions.

.. jupyter-input::

    from qiskit_ibm_provider import IBMProvider
    from qiskit_experiments.library.tomography import ProcessTomography
    from qiskit import QuantumCircuit

    provider = IBMProvider()
    backend = provider.get_backend("ibm_nairobi")
    qc = QuantumCircuit(1)
    qc.x(0)

    with backend.open_session() as session:
        tomography = ProcessTomography(qc)
        tomography_result = tomography.run(backend)
        tomography_result.block_for_results()
        session.cancel()

2. Use the ``qiskit-ibm-runtime`` provider. This requires extracting the circuits from the
   experiment and running them using :meth:`qiskit_ibm_runtime.Session.run`:

.. jupyter-input::

    from qiskit_experiments.library import StandardRB
    from qiskit_ibm_runtime import Session, QiskitRuntimeService
    import numpy as np

    exp = StandardRB([0], np.arange(1,800,200))
    backend = "ibm_nairobi"

    # all run options must be set before execution
    exp.set_run_options(shots=100)

    def run_jobs(session, job_circuits, run_options = None):
        runtime_inputs={'circuits': job_circuits,
                        'skip_transpilation': True, 
                        **run_options}
        jobs = session.run(program_id="circuit-runner", inputs=runtime_inputs)
        return jobs

    service = QiskitRuntimeService()

    with Session(service=service, backend=backend) as session:
        exp.backend = service.get_backend(session.backend())
        jobs = run_jobs(session, exp._transpiled_circuits(), exp.run_options)
        session.close()

    # exp_data will be the usual experiment data object
    exp_data = exp._initialize_experiment_data()
    exp_data.add_jobs(jobs)
    exp_data = exp.analysis.run(exp_data).block_for_results()

Runtime primitives are not currently supported natively in Qiskit Experiments, so running jobs
with the Runtime provider must be done with the ``circuit-runner`` program. We also turn off
transpilation with ``skip_transpilation`` since Qiskit Experiments already transpiles the circuits.