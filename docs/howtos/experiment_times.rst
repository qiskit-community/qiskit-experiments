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

Discussion
----------

:attr:`.ExperimentData.start_datetime` can also be set to a custom timestamp when instantiating an
:class:`.ExperimentData` object by passing a value to the ``start_datetime`` field.
