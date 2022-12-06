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

    .. grid-item-card::
        :img-top: _static/tutorials.png

        Tutorials
        ^^^^^^^^^

        These step-by-step tutorials teach the fundamentals of the package and 
        are suitable for getting started.

        You'll find in these tutorials:

        * An overview of the package structure
        * What constitutes an experiment
        * How to write your own experiment

        +++

        .. button-ref:: tutorials/index
            :expand:
            :color: secondary
            :click-parent:

            To the learning tutorials

    .. grid-item-card::
        :img-top: _static/howtos.png

        How-Tos
        ^^^^^^^

        These individual how-to recipes provide short and direct solutions to some commonly
        asked questions for Qiskit Experiments users.

        You'll find in these recipes:

        * How to save and retrieve experiment data
        * How to customize the appearance of your figures

        +++

        .. button-ref:: howtos/index
            :expand:
            :color: secondary
            :click-parent:

            To the how-to recipes

    .. grid-item-card::
        :img-top: _static/guides.png

        Experiment Guides
        ^^^^^^^^^^^^^^^^^

        These are in-depth guides to key experiments in the package, describing
        their background, principle, and how to run them in Qiskit Experiments.

        You'll find in these guides:
        
        * What is randomized benchmarking and when is it useful

        +++

        .. button-ref:: guides/index
            :expand:
            :color: secondary
            :click-parent:

            To the experiment guides


    .. grid-item-card::
        :img-top: _static/api.png

        API Reference
        ^^^^^^^^^^^^^

        This is a detailed description of every module, method, and function in 
        Qiskit Experiments and how to use them, suitable for those working closely
        with specific parts of the package or writing your custom code.

        You'll find in these references:
        
        * What are all possible input parameters to CurveAnalysis

        +++

        .. button-ref:: apidocs/index
            :expand:
            :color: secondary
            :click-parent:

            To the API reference

.. toctree::
    :hidden:

    tutorials/index
    howtos/index
    guides/index
    apidocs/index
    release_notes

.. Hiding - Indices and tables
   :ref:`genindex`
   :ref:`modindex`
   :ref:`search`

