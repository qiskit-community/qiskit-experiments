How to control the splitting of experiment circuits into jobs
=============================================================

Problem
-------

You want to control how many jobs an experiment is split into when running on a backend.

Solution
--------

Discussion
----------

Qiskit Experiments will automatically split circuits across jobs for you for backends
that have a maximum circuit number per circuit, which is given by `max_circuits` property in :meth:`qiskit.providers.BackendV1.configuration` for V1 backends and :meth:`qiskit.providers.BackendV2.max_circuits` for V2. This should
work automatically in most cases, but there may be some backends where other limits
exist.