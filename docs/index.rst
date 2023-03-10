.. _qiskit-experiments:

Qiskit Experiments Documentation
================================

.. warning::

    This package is still under active development and it is very likely
    that there will be breaking API changes in future releases.
    If you encounter any bugs, please open an issue on
    `GitHub <https://github.com/Qiskit/qiskit-experiments/issues>`_.

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

    .. grid-item-card:: How-To Guides

        .. image:: _static/howtos.png
            :target: howtos/index.html
    
        These standalone how-to guides provide short and direct solutions to some commonly
        asked questions for Qiskit Experiments users. You'll find in these guides:

        * How to :doc:`re-instantiate experiment data for an existing experiment <howtos/new_experimentdata>`
        * How to :doc:`customize the splitting of circuits into jobs <howtos/job_splitting>`

        +++

        .. button-ref:: howtos/index
            :expand:
            :color: secondary

            To the how-to guides

    .. grid-item-card:: Experiment Manuals

        .. image:: _static/manuals.png
            :target: manuals/index.html

        These are in-depth manuals to key experiments in the package, describing their
        background, principle, and how to run them in Qiskit Experiments. You'll find in
        these manuals:
        
        * How to analyze 1- and 2-qubit errors in :doc:`randomized benchmarking </manuals/benchmarking/randomized_benchmarking>`
        * How to calculate the speedup from using :doc:`restless measurements </manuals/measurement/restless_measurements>`

        +++

        .. button-ref:: manuals/index
            :expand:
            :color: secondary

            To the experiment manuals


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
    :caption: Tutorials

    All Tutorials <tutorials/index>
    tutorials/intro
    tutorials/getting_started
    Calibrations <tutorials/calibrations>
    Data Processor <tutorials/data_processor>
    Curve Analysis <tutorials/curve_analysis>
    Visualization <tutorials/visualization>
    Custom Experiments <tutorials/custom_experiment>

.. toctree::
    :hidden:

    howtos/index
    manuals/index
    apidocs/index
    release_notes
    Development Branch Docs <https://qiskit.org/documentation/experiments/dev>
    GitHub <https://github.com/Qiskit/qiskit-experiments>

|

.. Hiding - Indices and tables
   :ref:`genindex`
   :ref:`modindex`
   :ref:`search`

