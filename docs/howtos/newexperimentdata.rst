How to instantiate a new experiment data object
===============================================

Problem
-------

Sometimes due to events like a lost connection, the :class:`.ExperimentData` class that is returned 
upon completion of an experiment may not contain correct results. There 
are also times when you would like to rerun the analysis for existing experiments with
previously run jobs.

Solution
--------

If you want to instantiate a new experiment data object from an existing experiment and
jobs that finished execution successfully, you need to know the exact experiment you
ran and its options, as well as the IDs of the jobs that were executed.

.. code-block:: python

    from qiskit_experiments.framework import ExperimentData

    # The same experiment that you ran
    experiment = Experiment(**opts)

    # List of job IDs for the experiment
    data = ExperimentData(job_ids=job_ids)
    experiment.analysis.run(data)
    data.block_for_results()

``data`` will be the new experiment data class.

Discussion
----------

The job IDs can be retrieved from the original experiment data object using the 
``job_ids`` attribute.

See Also
--------

* `Saving and loading experiment data locally <local_service.html>`_
* `Saving and loading experiment data with the cloud service <cloud_service.html>`_
