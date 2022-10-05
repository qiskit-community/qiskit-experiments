################################
Qiskit Experiments Documentation
################################

About Qiskit Experiments
========================

Qiskit Experiments provides both a :doc:`library <apidocs/library>` of standard
quantum characterization, calibration, and verification experiments, and a
general :doc:`framework <apidocs/framework>` for implementing custom experiments
which can be run on quantum devices through Qiskit.

Experiments run on `IBMQ Provider <https://github.com/Qiskit/qiskit-ibmq-provider>`_
backends can be stored and retrieved from an online experiment
:doc:`database <apidocs/database_service>`.

.. warning::

   This package is still under active development and it is very likely
   that there will be breaking API changes in future releases.
   If you encounter any bugs please open an issue on
   `Github <https://github.com/Qiskit/qiskit-experiments/issues>`_


Table of Contents
=================

We've divided up the documentation into areas by purpose. The tutorials are 
learning-based documentation suitable for getting started. The how-to recipes
are short and direct instructions for solving specific problems. The experiment
guides have in-depth explanations for key experiments in the package, and discuss
advanced usage and options that would be of interest to experimentalists and 
researchers.

.. toctree::
  :maxdepth: 2

  Tutorials <tutorials/index>
  How-To Recipes <howtos/index>
  Experiment Guides <guides/index>
  API References <apidocs/index>
  Experiment Library <apidocs/library>
  Release Notes <release_notes>

.. Hiding - Indices and tables
   :ref:`genindex`
   :ref:`modindex`
   :ref:`search`
