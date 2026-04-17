Save and load experiment data with an experiment service
========================================================

.. note::
    The cloud service at https://quantum.ibm.com/experiments has been
    sunset in the move to the new IBM Quantum cloud platform. Saving and loading
    to the cloud will not work.

Problem
-------

You want to save and retrieve experiment data from an experiment service.

Solution
--------

The :class:`~.ExperimentData` supports saving and loading data with an
experiment service class satisfying the :class:`~.ExperimentService` protocol.
Here we demonstrate with the :class:`~.LocalExperimentService` class in Qiskit
Experiments.

Saving
~~~~~~

.. note::

   In the examples below, the service is instantiated with
   ``LocalExperimentService()`` which only saves results to
   memory. You might want to use ``LocalExperimentService(db_dir=db_dir)``
   instead specifying some local file path ``db_dir`` to save results to. Keep
   in mind that :class:`~.LocalExperimentService` was not designed to scale to
   saving a large amount of data.

Saving results is done by calling :meth:`.ExperimentData.save`:

.. jupyter-execute::
    :hide-code:

    # backend
    from qiskit_ibm_runtime.fake_provider import FakeManilaV2
    from qiskit_aer import AerSimulator
    backend = AerSimulator.from_backend(FakeManilaV2())

.. jupyter-execute::

    import numpy as np
    from qiskit_experiments.library import T1
    from qiskit_experiments.database_service import LocalExperimentService

    delays = np.arange(1.e-6, 300.e-6, 30.e-6)
    exp = T1(physical_qubits=(0, ), delays=delays, backend=backend)

    exp_data = exp.run().block_for_results()
    service = LocalExperimentService()
    exp_data.service = service
    exp_data.save()

Loading
~~~~~~~

Let's load the previous experiment again from the service. First, we create a
:class:`~qiskit_experiments.framework.Provider` object that has a
``job(job_id)`` method that can return a
:class:`~qiskit_experiments.framework.Job` instance. Since this is a local
test, a fake provider class that just returns jobs it has been given is used.
Another provider like :class:`~qiskit_ibm_runtime.QiskitRuntimeService` could
be used instead. Also, the provider is only needed for reloading the raw job
data for rerunning analysis. If only the experiment results and figures are
needed, the ``provider`` argument to :meth:`.ExperimentData.load` can omitted.
A warning about not being able to access the job data will be emitted in this
case.

.. jupyter-execute::

    from qiskit_experiments.test.utils import FakeProvider

    provider = FakeProvider()
    for job in exp_data.jobs():
        provider.add_job(job)

Now the experiment data can be reloaded:

.. jupyter-execute::

    from qiskit_experiments.framework import ExperimentData
    load_expdata = ExperimentData.load(exp_data.experiment_id, service=service, provider=provider)

Now we can display the figure from the loaded experiment data:

.. jupyter-execute::

    load_expdata.figure(0)

The analysis results have been retrieved as well and can be accessed normally.

.. jupyter-execute::

    load_expdata.analysis_results(dataframe=True)

Discussion
----------

Note that calling :meth:`~.ExperimentData.save` before the experiment is complete will
instantiate an experiment entry in the database, but it will not have
complete data. To fix this, you can call :meth:`~.ExperimentData.save` again once the
experiment is done running.

Sometimes the metadata of an experiment can be very large and cannot be stored directly in the database.
In this case, a separate ``metadata.json`` file will be stored along with the experiment. Saving and loading
this file is done automatically in :meth:`~.ExperimentData.save` and :meth:`~.ExperimentData.load`.

Auto-saving an experiment
~~~~~~~~~~~~~~~~~~~~~~~~~

The `auto_save` feature automatically saves changes to the
:class:`.ExperimentData` object to the experiment service whenever it's updated.

.. jupyter-execute::

    delays = np.arange(1.e-6, 300.e-6, 30.e-6)
    exp = T1(physical_qubits=(0, ), delays=delays, backend=backend)

    exp_data = exp.run()
    service = LocalExperimentService()
    exp_data.service = service
    exp_data.auto_save = True
    exp_data.block_for_results()


Setting ``auto_save = True`` works by triggering :meth:`.ExperimentData.save`
once the experiment's analysis completes.

When working with composite experiments, setting ``auto_save`` will propagate this
setting to the child experiments.

Deleting an experiment
~~~~~~~~~~~~~~~~~~~~~~

Both figures and analysis results can be deleted. Note that unless you
have auto save on, the update has to be manually saved to the
database by calling :meth:`~.ExperimentData.save`. Because there are two analysis
results, one for the T1 parameter and one for the curve fitting results, we must
delete twice to fully remove the analysis results.

.. jupyter-input::

    t1_expdata.delete_figure(0)
    t1_expdata.delete_analysis_result(0)
    t1_expdata.delete_analysis_result(0)

Tagging experiments
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Tags and notes can be added to experiments to help identify specific experiments in the interface.
For example, an experiment can be tagged with the following code.

.. jupyter-input::

   t1_expdata.tags = ['tag1', 'tag2']
   t1_expdata.notes = "Example note."
