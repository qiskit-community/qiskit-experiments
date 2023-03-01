Instantiate a new data object for an existing experiment
========================================================

Problem
-------

You want to instantiate a new :class:`.ExperimentData` object from an existing
experiment whose jobs have finished execution successfully.

Solution
--------

.. note::
    This recipe requires :mod:`qiskit-ibm-provider`. For how to migrate from the deprecated :mod:`qiskit-ibmq-provider` to :mod:`qiskit-ibm-provider`,
    consult the `migration guide <https://qiskit.org/documentation/partners/qiskit_ibm_provider/tutorials/Migration_Guide_from_qiskit-ibmq-provider.html>`_.\

Use the code template below. You need to know the exact experiment you
ran and its options, as well as the IDs of the jobs that were executed.

.. jupyter-input::

    from qiskit_experiments.framework import ExperimentData
    from qiskit_ibm_provider import IBMProvider

    # The experiment you ran
    experiment = Experiment(**opts)

    # List of job IDs for the experiment
    job_ids= [job1, job2, ...]

    provider = IBMProvider()

    data = ExperimentData(experiment = experiment)
    data.add_jobs([provider.retrieve_job(job_id) for job_id in job_ids])
    experiment.analysis.run(data)

    # Block execution of subsequent code until analysis is complete
    data.block_for_results()

``data`` will be the new experiment data object.

Discussion
----------

This recipe is helpful for cases such as a lost connection during experiment execution, 
where the jobs may have finished running on the remote backends but the 
:class:`.ExperimentData` class returned upon completion of an experiment does not 
contain correct results.

You may also want to rerun the analysis of a previously-run experiment with different 
options when you instantiate this new :class:`.ExperimentData` object.
Here's a code snippet where we reconstruct a parallel experiment
consisting of randomized benchmarking experiments, then change the gate error ratio
as well as the line plot color of the first component experiment.

.. jupyter-input::

    pexp = ParallelExperiment([
        StandardRB((i,), np.arange(1, 800, 200), num_samples=10) for i in range(2)])

    pexp.analysis.component_analysis(0).options.gate_error_ratio = {
        "x": 10, "sx": 1, "rz": 0
    }
    pexp.analysis.component_analysis(0).plotter.figure_options.series_params.update(
        {
            "rb_decay": {"color": "r"}
        }
    )

    data = ExperimentData(experiment=pexp)
    data.add_jobs([provider.retrieve_job(job_id) for job_id in job_ids])
    pexp.analysis.run(data)

See Also
--------

* `Saving and loading experiment data with the cloud service <cloud_service.html>`_
