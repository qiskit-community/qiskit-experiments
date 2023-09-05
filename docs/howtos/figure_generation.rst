Control figure generation 
=========================

Problem
-------

You want to change the default behavior where figures are generated with every experiment.

Solution
--------

For a single non-composite experiment, figure generation can be switched off by setting the analysis
option ``plot`` to ``False``:

.. jupyter-input::

    experiment.analysis.set_options(plot = False)    

For composite experiments, there is a ``generate_figures`` parameter which controls how child figures are
generated. There are three options:

- ``always``: The default behavior, generate figures for each child experiment.
- ``never``: Never generate figures for any child experiment.
- ``selective``: Only generate figures for analysis results where ``quality`` is ``bad``. This is useful
  for large composite experiments where you only want to examine qubits with problems.

Discussion
----------

This guide is helpful for large composite experiments, where generating all figures incurs a significant
overhead.

See Also
--------

* The `Visualization tutorial <visualization.html>`_ discusses how to customize figures
