Randomized Benchmarking
=======================

A randomized benchmarking (RB) experiment consists of the generation of
random Clifford circuits on the given qubits such that the unitary
computed by the circuits is the identity. After running the circuits,
the number of shots resulting in an error (i.e. an output different than
the ground state) are counted, and from this data one can infer error
estimates for the quantum device, by calculating the Error Per Clifford.
See `Qiskit
Textbook <https://qiskit.org/textbook/ch-quantum-hardware/randomized-benchmarking.html>`__
for an explanation on the RB method, which is based on Ref. [1, 2].

.. jupyter-execute::

    import numpy as np
    from qiskit_experiments.library import StandardRB, InterleavedRB
    from qiskit_experiments.framework import ParallelExperiment
    from qiskit_experiments.library.randomized_benchmarking import RBUtils
    import qiskit.circuit.library as circuits
    
    # For simulation
    from qiskit.providers.aer import AerSimulator
    from qiskit.test.mock import FakeParis
    
    backend = AerSimulator.from_backend(FakeParis())

Standard RB experiment
----------------------

To run the RB experiment we need to provide the following RB parameters,
in order to generate the RB circuits and run them on a backend:

-  ``qubits``: The number of qubits or list of physical qubits for the
   experiment

-  ``lengths``: A list of RB sequences lengths

-  ``num_samples``: Number of samples to generate for each sequence
   length

-  ``seed``: Seed or generator object for random number generation. If
   ``None`` then ``default_rng`` will be used

-  ``full_sampling``: If ``True`` all Cliffords are independently
   sampled for all lengths. If ``False`` for sample of lengths longer
   sequences are constructed by appending additional Clifford samples to
   shorter sequences. The default is ``False``

The analysis results of the RB Experiment includes:

-  ``EPC``: The estimated Error Per Clifford

-  ``alpha``: The depolarizing parameter. The fitting function is
   :math:`a \cdot \alpha^m + b`, where :math:`m` is the Clifford length

-  ``EPG``: The Error Per Gate calculated from the EPC, only for 1-qubit
   or 2-qubit quantum gates (see Ref. [3])

Running a 1-qubit RB experiment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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
    display(expdata1.figure(0))
    for result in results1:
        print(result)



Running a 2-qubit RB experiment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Running a 1-qubit RB experiment and a 2-qubit RB experiment, in order to
calculate the gate error (EPG) of the ``cx`` gate:

.. jupyter-execute::

    lengths = np.arange(1, 200, 30)
    num_samples = 10
    seed = 1010
    qubits = (1,4)
    
    # Run a 1-qubit RB expriment on qubits 1, 4 to determine the error-per-gate of 1-qubit gates
    expdata_1q = {}
    epg_1q = []
    lengths_1_qubit = np.arange(1, 800, 200)
    for qubit in qubits:
        exp = StandardRB([qubit], lengths_1_qubit, num_samples=num_samples, seed=seed)
        expdata = exp.run(backend).block_for_results()
        expdata_1q[qubit] = expdata
        epg_1q += expdata.analysis_results()

.. jupyter-execute::

    # Run an RB experiment on qubits 1, 4
    exp2 = StandardRB(qubits, lengths, num_samples=num_samples, seed=seed)
    
    # Use the EPG data of the 1-qubit runs to ensure correct 2-qubit EPG computation
    exp2.analysis.set_options(epg_1_qubit=epg_1q)
    
    # Run the 2-qubit experiment
    expdata2 = exp2.run(backend).block_for_results()
    
    # View result data
    results2 = expdata2.analysis_results()

.. jupyter-execute::

    # View result data
    display(expdata2.figure(0))
    for result in results2:
        print(result)

.. jupyter-execute::

    # Compare the computed EPG of the cx gate with the backend's recorded cx gate error:
    expected_epg = RBUtils.get_error_dict_from_backend(backend, qubits)[(qubits, 'cx')]
    exp2_epg = expdata2.analysis_results("EPG_cx").value
    
    print("Backend's reported EPG of the cx gate:", expected_epg)
    print("Experiment computed EPG of the cx gate:", exp2_epg)


Displaying the RB circuits
~~~~~~~~~~~~~~~~~~~~~~~~~~

Generating an example RB circuit:

.. jupyter-execute::

    # Run an RB experiment on qubit 0
    exp = StandardRB(qubits=[0], lengths=[10], num_samples=1, seed=seed)
    c = exp.circuits()[0]

We transpile the circuit into the backend’s basis gate set:

.. jupyter-execute::

    from qiskit import transpile
    basis_gates = backend.configuration().basis_gates
    print(transpile(c, basis_gates=basis_gates))


Interleaved RB experiment
-------------------------

Interleaved RB experiment is used to estimate the gate error of the
interleaved gate (see Ref. [4]).

In addition to the usual RB parameters, we also need to provide:

-  ``interleaved_element``: the element to interleave, given either as a
   group element or as an instruction/circuit

The analysis results of the RB Experiment includes the following:

-  ``EPC``: The estimated error of the interleaved gate

-  ``alpha`` and ``alpha_c``: The depolarizing parameters of the
   original and interleaved RB sequences respectively

Extra analysis results include

-  ``EPC_systematic_err``: The systematic error of the interleaved gate
   error (see Ref. [4])

-  ``EPC_systematic_bounds``: The systematic error bounds of the
   interleaved gate error (see Ref. [4])

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
    
    # View result data
    for result in par_results:
        print(result)
        print("\nextra:")
        print(result.extra)


Viewing sub experiment data
~~~~~~~~~~~~~~~~~~~~~~~~~~~

The experiment data returned from a batched experiment also contains
individual experiment data for each sub experiment which can be accessed
using ``child_data``

.. jupyter-execute::

    # Print sub-experiment data
    for i in qubits:
        print(f"Component experiment {i}")
        display(par_expdata.child_data(i).figure(0))
        for result in par_expdata.child_data(i).analysis_results():
            print(result)

References
----------

[1] Easwar Magesan, J. M. Gambetta, and Joseph Emerson, *Robust
randomized benchmarking of quantum processes*,
https://arxiv.org/pdf/1009.3639

[2] Easwar Magesan, Jay M. Gambetta, and Joseph Emerson, *Characterizing
Quantum Gates via Randomized Benchmarking*,
https://arxiv.org/pdf/1109.6887

[3] David C. McKay, Sarah Sheldon, John A. Smolin, Jerry M. Chow, and
Jay M. Gambetta, *Three Qubit Randomized Benchmarking*,
https://arxiv.org/pdf/1712.06550

[4] Easwar Magesan, Jay M. Gambetta, B. R. Johnson, Colm A. Ryan, Jerry
M. Chow, Seth T. Merkel, Marcus P. da Silva, George A. Keefe, Mary B.
Rothwell, Thomas A. Ohki, Mark B. Ketchen, M. Steffen, *Efficient
measurement of quantum gate error by interleaved randomized
benchmarking*, https://arxiv.org/pdf/1203.4550

[5] Jay M. Gambetta, A. D. C´orcoles, S. T. Merkel, B. R. Johnson, John
A. Smolin, Jerry M. Chow, Colm A. Ryan, Chad Rigetti, S. Poletto, Thomas
A. Ohki, Mark B. Ketchen, and M. Steffen, *Characterization of
addressability by simultaneous randomized benchmarking*,
https://arxiv.org/pdf/1204.6308

.. jupyter-execute::

    import qiskit.tools.jupyter
    %qiskit_copyright



.. raw:: html

    <div style='width: 100%; background-color:#d5d9e0;padding-left: 10px; padding-bottom: 10px; padding-right: 10px; padding-top: 5px'><h3>This code is a part of Qiskit</h3><p>&copy; Copyright IBM 2017, 2021.</p><p>This code is licensed under the Apache License, Version 2.0. You may<br>obtain a copy of this license in the LICENSE.txt file in the root directory<br> of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.<p>Any modifications or derivative works of this code must retain this<br>copyright notice, and modified files need to carry a notice indicating<br>that they have been altered from the originals.</p></div>

