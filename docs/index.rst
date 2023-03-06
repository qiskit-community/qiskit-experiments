################################
Qiskit Experiments documentation
################################

.. warning::

    This package is still under active development and it is very likely
    that there will be breaking API changes in future releases.
    If you encounter any bugs, please open an issue on
    `Github <https://github.com/Qiskit/qiskit-experiments/issues>`_.

Qiskit Experiments provides both a :doc:`library <apidocs/library>` of standard
quantum characterization, calibration, and verification experiments, and a
general :doc:`framework <apidocs/framework>` for implementing custom experiments
which can be run on quantum devices through Qiskit.

We've divided up the documentation into four sections with different purposes:

.. grid:: 2
    :gutter: 5

    .. grid-item-card:: Tutorials

        .. image:: _static/tutorials.png
            :target: tutorials/index.html

        These step-by-step tutorials teach the fundamentals of the package and 
        are suitable for getting started. You'll find in these tutorials:

        * An overview of the :ref:`package structure <primer>`
        * How to :doc:`install the package and run your first experiment </tutorials/getting_started>`
        * How to :doc:`write your own experiment </tutorials/custom_experiment>`

        +++

        .. button-ref:: tutorials/index
            :expand:
            :color: secondary

            To the learning tutorials

    .. grid-item-card:: How-Tos

        .. image:: _static/howtos.png
            :target: howtos/index.html
    
        These individual how-to recipes provide short and direct solutions to some commonly
        asked questions for Qiskit Experiments users. You'll find in these recipes:

        * How to :doc:`re-instantiate experiment data for an existing experiment <howtos/new_experimentdata>`
        * How to :doc:`customize the splitting of circuits into jobs <howtos/job_splitting>`

        +++

        .. button-ref:: howtos/index
            :expand:
            :color: secondary

            To the how-to recipes

    .. grid-item-card:: Experiment Guides

        .. image:: _static/guides.png
            :target: guides/index.html

        These are in-depth guides to key experiments in the package, describing
        their background, principle, and how to run them in Qiskit Experiments. You'll find in these guides:
        
        * How to analyze 1- and 2-qubit errors in :doc:`randomized benchmarking </guides/randomized_benchmarking>`
        * How to calculate the speedup from using :doc:`restless measurements </guides/restless_measurements>`

        +++

        .. button-ref:: guides/index
            :expand:
            :color: secondary

            To the experiment guides


    .. grid-item-card:: API Reference

        .. image:: _static/api.png
            :target: apidocs/index.html

        This is a detailed description of every module, method, and function in 
        Qiskit Experiments and how to use them, suitable for those working closely
        with specific parts of the package or writing your custom code. You'll find in these references:
        
        * Parameters, attributes, and methods of the :class:`.BaseExperiment` class
        * Default experiment, transpile, and run options for the :class:`.T1` experiment
        +++

        .. button-ref:: apidocs/index
            :expand:
            :color: secondary

            To the API reference

.. toctree::
    :hidden:

    tutorials/index
    howtos/index
    guides/index
    apidocs/index
    release_notes
    GitHub <https://github.com/Qiskit/qiskit-experiments>

.. Hiding - Indices and tables
   :ref:`genindex`
   :ref:`modindex`
   :ref:`search`

