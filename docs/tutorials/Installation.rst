Installation
=============

Official Qiskit Experiments releases can be installed via the python package manager 
`pip`.

.. code-block::
    python -m pip install qiskit-experiments

If you want to install the most up-to-date version instead (may not be stable), you can
install the latest main branch:

.. code-block::
    python -m pip install git+https://github.com/Qiskit/qiskit-experiments.git

If you want to develop the package, you can install Qiskit Experiments from source by 
cloning the repository:

.. code-block::
    git clone https://github.com/Qiskit/qiskit-experiments.git
    python -m pip install -e qiskit-experiments

The `-e` option will keep your installed package up to date as you make or pull new 
changes.

Running Your First Experiment
=============================

Let's run a T1 experiment: