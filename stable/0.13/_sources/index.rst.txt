.. _qiskit-experiments:

Qiskit Experiments Documentation
================================

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

        * How to :doc:`rerun analysis for an existing experiment <howtos/rerun_analysis>`
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
        
        * How to analyze 1- and 2-qubit errors in :doc:`randomized benchmarking </manuals/verification/randomized_benchmarking>`
        * How to characterize a quantum circuit using :doc:`state tomography </manuals/verification/state_tomography>`

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

    Qiskit Experiments Home <self>
    Getting Started <tutorials/getting_started>
    tutorials/index
    howtos/index
    manuals/index
    apidocs/index
    release_notes
    GitHub <https://github.com/Qiskit-Community/qiskit-experiments>
    Development Branch Docs <https://qiskit-community.github.io/qiskit-experiments/dev/>

|

.. Hiding - Indices and tables
   :ref:`genindex`
   :ref:`modindex`
   :ref:`search`

