Get experiment timing information 
=================================

Problem
-------

You want to know when an experiment started and finished running.

Solution
--------

The :class:`.ExperimentData` class contains timing information in the following attributes, which
are all of type ``datetime.datetime`` and in your local timezone:

- :attr:`.ExperimentData.start_datetime` is when the :class:`.ExperimentData` was instantiated,
  which is also when the experiment began running in the typical workflow by calling
  ``experiment_data = exp.run()``.

- :attr:`.ExperimentData.end_datetime` is the time the most recently completed job was successfully
  added to the :class:`.ExperimentData` object. If a job was canceled, did not run successfully, or
  could not be added to the :class:`.ExperimentData` object for some other reason,
  :attr:`.ExperimentData.end_datetime` will not update.

.. note::
    The below attributes are only relevant for those who have access to the cloud service. You can 
    check whether you do by logging into the IBM Quantum interface 
    and seeing if you can see the `database <https://quantum.ibm.com/experiments>`__.

- :attr:`.ExperimentData.creation_datetime` is the time when the experiment data was saved via the
  service. This defaults to ``None`` if experiment data has not yet been saved.

- :attr:`.ExperimentData.updated_datetime` is the time the experiment data entry in the service was
  last updated. This defaults to ``None`` if experiment data has not yet been saved.

Discussion
----------

:attr:`.ExperimentData.start_datetime` can also be set to a custom timestamp when instantiating an
:class:`.ExperimentData` object by passing a value to the ``start_datetime`` field.
