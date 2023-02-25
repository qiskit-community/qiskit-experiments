How to instantiate a new data object for an existing experiment
===============================================================

Problem
-------

You want to instantiate a new :class:`.ExperimentData` object from an existing
experiment whose jobs have finished execution successfully.

Solution
--------

Use the code template below. You need to know the exact experiment you
ran and its options, as well as the IDs of the jobs that were executed.

.. code-block:: python

    from qiskit_experiments.framework import ExperimentData

    # The experiment you ran
    experiment = Experiment(**opts)

    # List of job IDs for the experiment
    job_ids= [job1, job2, ...]

    data = ExperimentData(job_ids=job_ids)
    experiment.analysis.run(data)
    # Blocks execution of subsequent code until analysis is complete
    data.block_for_results()

``data`` will be the new experiment data object.

Discussion
----------

This recipe is helpful for cases such as a lost connection during experiment execution, 
where the jobs may have finished running on the remote backends but the 
:class:`.ExperimentData` class returned upon completion of an experiment does not 
contain correct results.

There are also times when you may want to rerun the analysis of a previously-run 
experiment. You can instantiate this new :class:`.ExperimentData` object 
with different options. Here's an example where we take an existing T1 experiment
and rerun it with a new analysis:

exp = T1(qubit=0, delays=t1_delays)
...

See Also
--------

* `Saving and loading experiment data with the cloud service <cloud_service.html>`_
