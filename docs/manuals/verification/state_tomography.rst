Quantum State Tomography
========================

Quantum tomography is an experimental procedure to reconstruct a description of
part of a quantum system from the measurement outcomes of a specific set of
experiments. In particular, quantum state tomography reconstructs the density matrix
of a quantum state by preparing the state many times and measuring them in a tomographically 
complete basis of measurement operators.

.. note::
    This tutorial requires the :external+qiskit_aer:doc:`qiskit-aer <index>` and :external+qiskit_ibm_runtime:doc:`qiskit-ibm-runtime <index>`
    packages to run simulations.  You can install them with ``python -m pip
    install qiskit-aer qiskit-ibm-runtime``.

We first initialize a simulator to run the experiments on.

.. jupyter-execute::

    from qiskit_aer import AerSimulator
    from qiskit_ibm_runtime.fake_provider import FakePerth
    
    backend = AerSimulator.from_backend(FakePerth())

To run a state tomography experiment, we initialize the experiment with a circuit to
prepare the state to be measured. We can also pass in an
:class:`~qiskit.quantum_info.Operator` or a :class:`~qiskit.quantum_info.Statevector`
to describe the preparation circuit.

.. jupyter-execute::

    import qiskit
    from qiskit_experiments.framework import ParallelExperiment
    from qiskit_experiments.library import StateTomography

    # GHZ State preparation circuit
    nq = 2
    qc_ghz = qiskit.QuantumCircuit(nq)
    qc_ghz.h(0)
    qc_ghz.s(0)
    for i in range(1, nq):
        qc_ghz.cx(0, i)
    
    # QST Experiment
    qstexp1 = StateTomography(qc_ghz)
    qstdata1 = qstexp1.run(backend, seed_simulation=100).block_for_results()
    
    # Print results
    display(qstdata1.analysis_results(dataframe=True))


Tomography Results
~~~~~~~~~~~~~~~~~~

The main result for tomography is the fitted state, which is stored as a
``DensityMatrix`` object:

.. jupyter-execute::

    state_result = qstdata1.analysis_results("state", dataframe=True).iloc[0]
    print(state_result.value)

We can also visualize the density matrix:

.. jupyter-execute::

    from qiskit.visualization import plot_state_city
    state = qstdata1.analysis_results("state", dataframe=True).iloc[0].value
    plot_state_city(state, title='Density Matrix')

The state fidelity of the fitted state with the ideal state prepared by
the input circuit is stored in the ``"state_fidelity"`` result field.
Note that if the input circuit contained any measurements the ideal
state cannot be automatically generated and this field will be set to
``None``.

.. jupyter-execute::

    fid_result = qstdata1.analysis_results("state_fidelity", dataframe=True).iloc[0]
    print("State Fidelity = {:.5f}".format(fid_result.value))



Additional state metadata
^^^^^^^^^^^^^^^^^^^^^^^^^

Additional data is stored in the tomography under additional fields. This includes

- ``eigvals``: the eigenvalues of the fitted state 
- ``trace``: the trace of the fitted state 
- ``positive``: Whether the eigenvalues are all non-negative 

If trace rescaling was performed this dictionary will also contain a ``raw_trace`` field
containing the trace before rescaling. Futhermore, if the state was rescaled to be
positive or trace 1 an additional field ``raw_eigvals`` will contain the state
eigenvalues before rescaling was performed.

.. jupyter-execute::

    for col in ["eigvals", "trace", "positive"]:
        print(f"{col}: {state_result[col]}")

To see the effect of rescaling, we can perform a “bad” fit with very low
counts:

.. jupyter-execute::

    # QST Experiment
    bad_data = qstexp1.run(backend, shots=10, seed_simulation=100).block_for_results()
    bad_state_result = bad_data.analysis_results("state", dataframe=True).iloc[0]
    
    # Print result
    for key, val in bad_state_result.items():
        print(f"{key}: {val}")


Tomography Fitters
------------------

The default fitters is ``linear_inversion``, which reconstructs the
state using *dual basis* of the tomography basis. This will typically
result in a non-positive reconstructed state. This state is rescaled to
be positive-semidefinite (PSD) by computing its eigen-decomposition and
rescaling its eigenvalues using the approach from Ref. [1]_.

There are several other fitters are included (See API documentation for
details). For example, if ``cvxpy`` is installed we can use the
:func:`~.cvxpy_gaussian_lstsq` fitter, which allows constraining the fit to be
PSD without requiring rescaling.

.. jupyter-execute::
    :hide-code:

    import warnings
    # Hide warning from scs from output; is there a way to avoid the warning?
    # Full warning line:
    #   /.tox/docs/lib/python3.13/site-packages/scs/__init__.py:83: UserWarning: 
    #   Converting A to a CSC (compressed sparse column) matrix; may take a while.
    warnings.filterwarnings("ignore", module=r"scs", message="Converting A to a CSC")

.. jupyter-execute::

    try:
        import cvxpy
        
        # Set analysis option for cvxpy fitter
        qstexp1.analysis.set_options(fitter='cvxpy_gaussian_lstsq')
        
        # Re-run experiment
        qstdata2 = qstexp1.run(backend, seed_simulation=100).block_for_results()
    
        state_result2 = qstdata2.analysis_results("state", dataframe=True).iloc[0]
        for key, val in state_result2.items():
            print(f"{key}: {val}")
    
    except ModuleNotFoundError:
        print("CVXPY is not installed")

Parallel Tomography Experiment
------------------------------

We can also use the :class:`.ParallelExperiment` class to
run subsystem tomography on multiple qubits in parallel.

For example if we want to perform 1-qubit QST on several qubits at once:

.. jupyter-execute::

    from math import pi
    num_qubits = 5
    gates = [qiskit.circuit.library.RXGate(i * pi / (num_qubits - 1))
             for i in range(num_qubits)]
    
    subexps = [
        StateTomography(gate, physical_qubits=(i,))
        for i, gate in enumerate(gates)
    ]
    parexp = ParallelExperiment(subexps)
    pardata = parexp.run(backend, seed_simulation=100).block_for_results()
    
    display(pardata.analysis_results(dataframe=True))

View experiment analysis results for one component:

.. jupyter-execute::

    results = pardata.analysis_results(dataframe=True)
    display(results[results.components.apply(lambda x: x == ["Q0"])])

References
----------

.. [1] J Smolin, JM Gambetta, G Smith, Phys. Rev. Lett. 108, 070502 (2012), 
    `open access <https://arxiv.org/abs/arXiv:1106.5458>`__.

See also
--------

* API documentation: :mod:`~qiskit_experiments.library.tomography.StateTomography`
