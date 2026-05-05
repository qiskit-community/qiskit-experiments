Control the splitting of experiment circuits into jobs
======================================================

Problem
-------

You want to manually control how an experiment is split into jobs when running on 
a backend.

Solution
--------

There are two experiment options relevant to custom job splitting.
You can set the ``max_circuits`` option manually when running an experiment:

.. jupyter-input::

    exp = Experiment((0,))
    exp.set_experiment_options(max_circuits=100)

The experiment class will split its circuits into jobs such that no job has more than
``max_circuits`` number of jobs.

Furthermore, the :class:`.BatchExperiment` class has the experiment option
``separate_jobs`` which will run circuits of different sub-experiments in different
jobs:

.. jupyter-input::

    batch_exp = BatchExperiment([exp, exp])
    batch_exp.set_experiment_options(separate_jobs=True)

Note that this option is only available to :class:`.BatchExperiment` objects. To manage
job splitting when using :class:`.ParallelExperiment`, you can make a nested batch
experiment of parallel experiments.

Discussion
----------

Qiskit Experiments will automatically split circuits across jobs for you for backends
that have a maximum circuit number per job, which is given by the ``max_experiments`` 
property of :meth:`~qiskit.providers.BackendV1.configuration` for V1 backends and 
the :attr:`~qiskit.providers.BackendV2.max_circuits` attribute for V2 backends. This should
work automatically in most cases, but there may be some backends where other limits
exist. When the ``max_circuits`` experiment option is provided, the experiment class
will split the experiment circuits as dictated by the smaller of the backend property
and the experiment option.