Installation
=============

Official Qiskit Experiments releases can be installed via the python package manager 
`pip`.

.. code-block:: python

    python -m pip install qiskit-experiments

If you want to install the most up-to-date version instead (may not be stable), you can
install the latest main branch:

.. code-block:: python

    python -m pip install git+https://github.com/Qiskit/qiskit-experiments.git

If you want to develop the package, you can install Qiskit Experiments from source by 
cloning the repository:

.. code-block:: python

    git clone https://github.com/Qiskit/qiskit-experiments.git
    python -m pip install -e qiskit-experiments

The `-e` option will keep your installed package up to date as you make or pull new 
changes.

Running Your First Experiment
=============================

Let's run a T1 experiment. FIrst, we have to import the T1 experiment from the 
Qiskit Experiments library:

.. code-block:: python

    from qiskit_experiments.library import T1
    from qiskit_aer import AerSimulator
    import numpy as np

Instantiate the backend and the experiment:

.. code-block:: python

    backend = AerSimulator.from_backend(FakeVigo(), noise_model=noise_model)
    qubit0_t1 = backend.properties().t1(0)

    delays = np.arange(1e-6, 3 * qubit0_t1, 3e-5)
    exp = T1(qubit=0, delays=delays)exp_data = exp.run(backend=backend, seed_simulator=101).block_for_results()

Run and display results:

.. code-block:: python

    exp_data = exp.run(backend=backend, seed_simulator=101).block_for_results()

    # Print the result
    display(exp_data.figure(0))
    for result in exp_data.analysis_results():
        print(result)