Randomized Benchmarking (RB) - Tutorial
=======================================


Standard RB experiment
----------------------
We begin by creating a standard RB experiment for 1-qubit on a fake backend:

.. jupyter-execute::

    import numpy as np
    from qiskit_experiments.library import StandardRB, InterleavedRB
    from qiskit_experiments.framework import ParallelExperiment, BatchExperiment
    import qiskit.circuit.library as circuits
    
    # For simulation
    from qiskit_aer import AerSimulator
    from qiskit.providers.fake_provider import FakeVigo
    
    backend = AerSimulator.from_backend(FakeVigo())

.. jupyter-execute::

    lengths = np.arange(1, 800, 200)
    num_samples = 10
    seed = 1010
    qubits = [0]
    
    # Run an RB experiment on qubit 0
    exp1 = StandardRB(qubits, lengths, num_samples=num_samples, seed=seed)
    expdata1 = exp1.run(backend).block_for_results()
    results1 = expdata1.analysis_results()
    
    # View result data
    # note that the basis gates for FakeVigo are ["id", "rz", "sx", "x", "cx"]
    display(expdata1.figure(0))
    for result in results1:
        print(result)

Similarly, we create a standard RB experiment for 2 qubits. Here, we first compute the error on the
single-qubit operations for the two qubits:

.. jupyter-execute::

    lengths_2_qubit = np.arange(1, 200, 30)
    lengths_1_qubit = np.arange(1, 800, 200)
    num_samples = 10
    seed = 1010
    qubits = (1, 4)

    # Run a 1-qubit RB experiment on qubits 1, 4 to determine the error-per-gate of 1-qubit gates
    single_exps = BatchExperiment(
        [
            StandardRB([qubit], lengths_1_qubit, num_samples=num_samples, seed=seed)
            for qubit in qubits
        ],
        flatten_results=True,
    )
    expdata_1q = single_exps.run(backend).block_for_results()
    result_1q = expdata_1q.analysis_results()
    EPG_sx_q1 = expdata_1q.analysis_results("EPG_sx")[0].value.n
    EPG_rz_q1 = expdata_1q.analysis_results("EPG_rz")[0].value.n
    EPG_x_q1 = expdata_1q.analysis_results("EPG_x")[0].value.n
    EPG_sx_q4 = expdata_1q.analysis_results("EPG_sx")[1].value.n
    EPG_rz_q4 = expdata_1q.analysis_results("EPG_rz")[1].value.n
    EPG_x_q4 = expdata_1q.analysis_results("EPG_x")[1].value.n
    print("For q1, EPG_sx = {:.3e}, EPG_rz = {:.3e}, EPG_x = {:.3e}".format(EPG_sx_q1, EPG_rz_q1, EPG_x_q1))
    print("For q4, EPG_sx = {:.3e}, EPG_rz = {:.3e}, EPG_x = {:.3e}".format(EPG_sx_q4, EPG_rz_q4, EPG_x_q4))

.. jupyter-execute::

    # Run an RB experiment on qubits 1, 4
    exp_2q = StandardRB(qubits, lengths_2_qubit, num_samples=num_samples, seed=seed)
    
    # Use the EPG data of the 1-qubit runs to ensure correct 2-qubit EPG computation
    exp_2q.analysis.set_options(epg_1_qubit=result_1q)
    
    # Run the 2-qubit experiment
    expdata_2q = exp_2q.run(backend).block_for_results()

    # View result data
    display(expdata_2q.figure(0))
    for result in expdata_2q.analysis_results():
        print(result)

Note that ``EPC_corrected`` value is smaller than one of raw ``EPC``, which indicates the
contribution of depolarization from single-qubit error channels.

Displaying the RB circuits
~~~~~~~~~~~~~~~~~~~~~~~~~~

Generating an sample RB circuit:


.. jupyter-execute::

    # Run an RB experiment on qubit 0
    exp = StandardRB(qubits=[0], lengths=[10], num_samples=1, seed=seed)
    c = exp.circuits()[0]

We transpile the circuit into the backend’s basis gate set:

.. jupyter-execute::

    from qiskit import transpile
    basis_gates = backend.configuration().basis_gates
    print(transpile(c, basis_gates=basis_gates))

Setting the gate error ratio
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Next, we use the Aer simulator with a noise model and user-defined basis gates.

.. jupyter-execute::

    from qiskit_aer import AerSimulator
    from qiskit_aer.noise import NoiseModel, depolarizing_error

    x_error = depolarizing_error(0.04, 1)
    h_error = depolarizing_error(0.02, 1)
    s_error = depolarizing_error(0.00, 1)
    noise_model = NoiseModel()
    noise_model.add_all_qubit_quantum_error(x_error, "x")
    noise_model.add_all_qubit_quantum_error(h_error, "h")
    noise_model.add_all_qubit_quantum_error(s_error, "s")

    # Aer simulator
    backend = AerSimulator(noise_model=noise_model, seed_simulator=123)
    transpiler_options = {
                "basis_gates": ["x", "h", "s", "cx"],
            }
    # Prepare experiment data without analysis
    exp_1qrb = StandardRB(
        qubits=(0,),
        lengths=np.arange(1, 500, 50),
        seed=123,
        backend=backend,
    )
    exp_1qrb.set_transpile_options(**transpiler_options)
    expdata_1qrb = exp_1qrb.run(analysis=None).block_for_results(timeout=300)

    from qiskit_experiments.library.randomized_benchmarking import RBAnalysis
    # Run analysis with default options
    analysis = RBAnalysis()
    result = analysis.run(expdata_1qrb, replace_results=False)

    EPG_x = result.analysis_results("EPG_x").value.n
    EPG_s = result.analysis_results("EPG_s").value.n
    EPG_h = result.analysis_results("EPG_h").value.n
    print("EPG_x = {:.3e}, EPG_h = {:.3e}, EPG_s = {:.3e}".format(EPG_x, EPG_h, EPG_s))

We can define the gate error ratio, as in the following example:

.. jupyter-execute::

    analysis = RBAnalysis()
    # Run analysis with user-defined gate error ratio
    analysis.set_options(gate_error_ratio={"x": 2, "h": 1, "s": 0})
    result = analysis.run(expdata_1qrb)

    EPG_x = result.analysis_results("EPG_x").value.n
    EPG_s = result.analysis_results("EPG_s").value.n
    EPG_h = result.analysis_results("EPG_h").value.n
    print("EPG_x = {:.3e}, EPG_h = {:.3e}, EPG_s = {:.3e}".format(EPG_x, EPG_h, EPG_s))

Interleaved RB experiment
-------------------------

Running a 1-qubit interleaved RB experiment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. jupyter-execute::

    lengths = np.arange(1, 800, 200)
    num_samples = 10
    seed = 1010
    qubits = [0]
    
    # Run an Interleaved RB experiment on qubit 0
    # The interleaved gate is the x gate
    int_exp1 = InterleavedRB(
        circuits.XGate(), qubits, lengths, num_samples=num_samples, seed=seed)
    
    # Run
    int_expdata1 = int_exp1.run(backend).block_for_results()
    int_results1 = int_expdata1.analysis_results()

.. jupyter-execute::

    # View result data
    display(int_expdata1.figure(0))
    for result in int_results1:
        print(result)


Running a 2-qubit interleaved RB experiment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. jupyter-execute::

    lengths = np.arange(1, 200, 30)
    num_samples = 10
    seed = 1010
    qubits = [1,4]
    
    # Run an Interleaved RB experiment on qubits 1, 4
    # The interleaved gate is the cx gate
    int_exp2 = InterleavedRB(
        circuits.CXGate(), qubits, lengths, num_samples=num_samples, seed=seed)
    
    # Run
    int_expdata2 = int_exp2.run(backend).block_for_results()
    int_results2 = int_expdata2.analysis_results()

.. jupyter-execute::

    # View result data
    display(int_expdata2.figure(0))
    for result in int_results2:
        print(result)



Running a simultaneous RB experiment
------------------------------------

We use ``ParallelExperiment`` to run the RB experiment simultaneously on
different qubits (see Ref. [5])

.. jupyter-execute::

    lengths = np.arange(1, 800, 200)
    num_samples = 10
    seed = 1010
    qubits = range(3)
    
    # Run a parallel 1-qubit RB experiment on qubits 0, 1, 2
    exps = [StandardRB([i], lengths, num_samples=num_samples, seed=seed + i)
            for i in qubits]
    par_exp = ParallelExperiment(exps)
    par_expdata = par_exp.run(backend).block_for_results()
    par_results = par_expdata.analysis_results()


Viewing sub experiment data
~~~~~~~~~~~~~~~~~~~~~~~~~~~

The experiment data returned from a batch (or parallel) experiment also contains
individual experiment data for each sub experiment which can be accessed
using ``child_data``

.. jupyter-execute::

    # Print sub-experiment data
    for i in qubits:
        print(f"Component experiment {i}")
        display(par_expdata.child_data(i).figure(0))
        for result in par_expdata.child_data(i).analysis_results():
            print(result)

.. jupyter-execute::

    import qiskit.tools.jupyter
    %qiskit_copyright
