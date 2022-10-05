How to instantiate a new experiment data object
============

If you want to instantiate a new experiment data object from an existing experiment and
jobs that finished execution successfully:

.. code-block::
    from qiskit_experiments.framework import ExperimentData

    # The same experiment that you ran
    experiment = Experiment(**opts)

    # List of job IDs for the experiment
    data = ExperimentData(job_ids=job_ids)
    experiment.analysis.run(data)
    data.block_for_results()

`data` will be the new experiment data class.