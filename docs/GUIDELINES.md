# Guidelines for writing documentation

Read the [contributing guidelines](CONTRIBUTING.md) before proceeding.

Contents:
- [Guidelines for writing documentation](#guidelines-for-writing-documentation)
  - [Introduction](#introduction)
    - [General formatting guidelines](#general-formatting-guidelines)
    - [Writing code](#writing-code)
    - [Referencing objects](#referencing-objects)
    - [Tutorials](#tutorials)
    - [How-to guides](#how-to-guides)
    - [Experiment manuals](#experiment-manuals)
    - [API documentation](#api-documentation)
      - [Experiment class documentation](#experiment-class-documentation)
      - [Analysis class documentation](#analysis-class-documentation)
      - [Populating the table of contents](#populating-the-table-of-contents)

## Introduction

Qiskit Experiments documentation is split into four sections:

- Tutorials for learning the package from the ground up
- How-to guides for solving specific problems
- Experiment manuals for information on specific experiments
- API reference for technical documentation

### General formatting guidelines 

* For experiments, the documentation title should be just the name of the experiment. Use
  regular capitalization
* Use headers, subheaders, subsubheaders etc. for hierarchical text organization. No
  need to number the headers
* Use present progressive for subtitles, such as "Saving experiment data to the
  database" instead of "Save experiment data to the database"
* Use math notation as much as possible (e.g. use $\frac{\pi}{2}$ instead of pi-half or
  pi/2)
* Use device names as shown in the IBM Quantum Services dashboard, e.g. `ibmq_lima`
  instead of IBMQ Lima
* put identifier names (e.g. osc_freq) in code blocks using double backticks, i.e. `osc_freq`

### Writing code

All documentation is written in reStructuredText format and then built into formatted
text by Sphinx. Code cells can be written using `jupyter-execute` blocks, which will be
automatically executed, with both code and output shown to the user:

    .. jupyter-execute::

        # write Python code here

To display a block without actually executing the code, use the `.. jupyter-input::` and
`.. jupyter-output::` directives. To ignore an error from a Jupyter cell block, use the
`:raises:` directive. To see more options, consult the [Jupyter Sphinx documentation](https://jupyter-sphinx.readthedocs.io/en/latest/).

### Referencing objects

Modules, classes, methods, functions, and attributes mentioned in the documentation
should link to their API documentation whenever possible using the `:mod:`, `:class:`,
`:meth:`, `:func:`, and `:attr:` directives followed by the name of the object in single
backticks. Here are some common usage patterns:

- `` :class:`.CurveAnalysis` ``: This will render a link to the curve analysis class
  `CurveAnalysis` if its name is unique.
- `` :class:`qiskit_experiments.curve_analysis.CurveAnalysis` ``: This will render the 
  full path to the object with a link as long as the path is correct.
- `` :class:`~qiskit_experiments.curve_analysis.CurveAnalysis` ``: This will render only
  the object name itself instead of the full path. It's simpler to use the first pattern
  instead if the name is unique.

Consult the [Sphinx documentation](https://www.sphinx-doc.org/en/master/usage/restructuredtext/domains.html) for more detailed syntax.


Below are templates and guidelines for each of these types of documentation.

### Tutorials

The learning tutorials are for users who are familiar with Python and Qiskit and new to
the Qiskit Experiments package. Here are what to keep in mind when writing and updating
tutorials:

- The tutorials should be suitable for progressive learning, starting with simple
  instructions and gradually adding complexity. For example, T1 is a much better
  starting experiment than cross resonance hamiltonian tomography. Each new bit of
  added complexity that the user hasn't seen before should be explained.
- Whenever possible, external resources should be linked to. For example, classes and
  methods in Qiskit should be linked.
- If you make changes to the basic API shown in the tutorials, it's important to update
  the corresponding part in the tutorials. Consider adding a special note for major
  recent changes to inform users who may be used to the old usage pattern.


### How-to guides

The title of a how-to should clearly describe what problem it's solving. It should be an
action that follows "How to". The text itself has up to four sections, but only the
first two are required:

- Problem: This section should describe the user problem that your guide is providing a
  direct solution for in second person. This should ideally be a one-liner so that users
  can quickly scan it and see if it’s relevant to what they’re trying to do.

- Solution: This section should describe possible solutions for the problem with code
  snippets and text before and after that describe what is needed to run the code, as
  well as what it generates and how this solves the problem.

- Discussion: This section can go into detail on when this kind of problem can arise,
  caveats to running the code, and any related.

- See also: Links to other relevant documentation or resources.

Here is a template for how-to guides:

```
Write a how-to guide
====================

Problem
-------

You want to write a how-to guide.

Solution
--------

First, you need to have a specific problem in mind that you want to solve with your 
how-to. This might be a problem you encountered when using Qiskit Experiments yourself, 
for example. You then need to have a solution that you can describe with words and code
examples.

Discussion
----------

Not every type of information is suitable for a how-to. For example, if it's essential 
information that newcomers to the package should know, then it should go in the tutorials
section.

Subsection
~~~~~~~~~~

You can add subsections whenever appropriate.

See also
--------

* `The Qiskit Docs Guide <https://qiskit.github.io/qiskit_sphinx_theme>`__ 
```


### Experiment manuals

The main goal of `qiskit-experiment` experiment manuals is to serve as user manuals for
the various package components such as the characterization and calibration experiments.
To this end, each document should introduce the cover the main (if not all) use cases of
the experiment functionality, including code examples and expected outputs. Another
objective of the documentation is to provide the user with basic background on each
experiment method. The start of the manual should have a short background explanation
for what the experiment does, preferably 1 or 2 paragraphs long, which includes the main
literature references as well as a link to the relevant chapter in the Qiskit textbook,
if available. The common use cases of the experiment should be covered with a code
example and example outputs by printing relevant analysis results and plot figures.
Required and common parameters, such as experiment and analysis options, should be
covered.

See the [Randomized Benchmarking](https://qiskit-extensions.github.io/qiskit-experiments/manuals/verification/randomized_benchmarking.html)
guide and its [source code](manuals/verification/randomized_benchmarking.rst) for an
example. Here is a simple template for a manual:

```
New Experiment
==============

Here the experiment is introduced, and any background info needed to understand it is 
ideally provided to the level of someone who has taken a background course in quantum 
computing. References are provided to the original paper where the experiment was 
described, if relevant, and to good resources for understanding it.

Running the experiment
----------------------

Here caveats about the specific implementation of the experiment in this package are 
discussed and sample code is provided. Because information on the general inputs and 
outputs of an experiment will be covered in the tutorials, there’s no need to repeat 
information that applies to all experiments.

.. jupyter-execute::

    # Sample code that runs the experiment is shown here.

Choosing good parameters
~~~~~~~~~~~~~~~~~~~~~~~~

If there are specific considerations when running the experiment that you want to
highlight, this is a good place to discuss them.

Advanced usage
--------------

You may want to highlight advanced usage or ways to improve performance that will be of 
interest to experimentalists and researchers. For example, for the T1 experiment, one 
such section might discuss the optimal way of choosing delay lengths to obtain the 
most information about T1, in scenarios where T1 is roughly known versus scenarios 
where nearly nothing is known. Papers should be cited where relevant.

See also
--------

Links to relevant experiment classes in the API docs should be provided here.

```

### API documentation

API documentation is automatically generated from docstrings. If you implement a new
experiment or analysis or update how an existing one functions, you should use following
style so that the documentation is formatted in the same manner throughout our
experiment library. You can use standard
[reStructuredText](https://www.sphinx-doc.org/en/master/usage/restructuredtext/index.html)
directives along with our syntax.

#### Experiment class documentation

There are several predefined sections for the class docstring.

```buildoutcfg
   """One line simple summary of this experiment in the format of "An experiment that
   measures [parameter]".
   
   You can add more information after line feed. The first line will be shown in an 
   automatically generated table of contents on the module's top page. 
   This text block is not shown so you can keep the table clean.
   
   You can use following sections. The text within a section should be indented.
   
   # section: overview

       Overview of the experiment. This information SHOULD be provided for every experiment. 
       This section covers technical aspect of experiment and explains how the experiment works.
       
       A diagram of typical quantum circuit that the experiment generates may help readers 
       to grasp the behavior of this experiment.
   
   # section: analysis_ref

       You MUST provide a reference to the default analysis class in the base class. 
       This section is recursively referred by child classes if not explicitly given there.
       The format should be a Sphinx cross-reference to the class, such as
       
       :class:`~qiskit_experiments.framework.BaseAnalysis`
   
   # section: warning
       If user must take special care when using the experiment (e.g. API is not stabilized) 
       you should clarify in this section. 
   
   # section: note
       Optional. This comment is shown in a box so that the message is stood out.
   
   # section: example
       Optional. You can write code example here. For example,
       
       .. code-block:: python
       
           exp = MyExperiment(qubits=[0, 1], backend=backend)
           exp.run()
       
       This is effective especially when your experiment has complicated options.
   
   # section: reference
       Optional. You can write reference to article or external website.
       To write a reference to an arXiv work, you can use convenient macro.
       
       .. ref_arxiv:: Auth2020a 21xx.01xxx
       
       This collects the latest article information from web and automatically 
       generates a nicely formatted citation from the arXiv ID.
       
       For referring to the website,
       
       .. ref_website:: Qiskit Experiment GitHub, https://github.com/Qiskit-Extensions/qiskit-experiments
       
       you can use the above macro, where you can provide a string for the hyperlink and 
       the destination location separated by single comma.
   
   # section: manual
       Optional. Link to manuals of this experiment if one exists.
   
   # section: see_also
       Optional. You can list relevant experiment or module.
       Here you cannot write any comments. 
       You just need to list absolute paths to relevant API documents, i.e.
       
       qiskit_experiments.framework.BaseExperiment
       qiskit_experiments.framework.BaseAnalysis
   """
```

You also need to provide the experiment option description in the
`_default_experiment_options` method if you add new options. This description will be
automatically propagated through child classes, so you don't need to manually copy
documentation. Of course, you can override documentation in the child class if it
behaves differently there.

```buildoutcfg
    """Default experiment options.
    
    Experiment Options:
        opt1 (int): Description of opt1.
        opt2 (float): Description of opt2.
        opt3 (List[SomeClass]): Description of opt3.
    """
```

Note that you should use the [Google docstring
style](https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html).
Numpy or other docstring styles cannot be parsed by our Sphinx extension, and the
section header should be named `Experiment Options` (NOT `Args`). Since this is a
private method, any other documentation besides option descriptions are not rendered in
the HTML documentation. Documentation for options are automatically formatted and
inserted into the class documentation.

#### Analysis class documentation

You can use the same syntax and section headers for the analysis class documentation. In
addition, you can use extra sections, `fit_model` and `fit_parameters`, if needed.

```buildoutcfg
   """One line simple summary of this analysis.
   
   # section: overview
       Overview of this analysis.
   
   # section: fit_model
       Optional. If this analysis fits something, probably it is worth describing 
       the fit model. You can use math mode where latex commands are available.
       
       .. math::
       
           F(x) = a\exp(x) + b
       
       It is recommended to omit `*` symbols for multiplication (looks ugly in math mode), 
       and you should carefully choose the parameter name so that symbols matches with
       variable names shown in analysis results. You can write symbol :math:`a` here too.
   
   # section: fit_parameters
       Optional. Description for fit parameters in the model.
       You can also write how initial guess is generated and how fit bound is determined.
       
       defpar a:
           desc: Amplitude.
           init_guess: This is how :math:`a` is generated. No line feed.
           bounds: [-1, 1]
       
       defpar b:
           desc: Offset.
           init_guess: This is how :math:`b` is generated. No line feed.
           bounds: (0, 1]
        
       The defpar syntax is parsed and formatted nicely.
   """
```

You also need to provide a description for analysis class options in the
`_default_options` method.

```buildoutcfg
    """Default analysis options.
    
    Analysis Options:
        opt1 (int): Description of opt1.
        opt2 (float): Description of opt2.
        opt3 (List[SomeClass]): Description of opt3.
    """
```

This is the same syntax with experiment options in the experiment class. Note that
header should be named `Analysis Options` to be parsed correctly.

#### Populating the table of contents

After you complete documentation of your classes, you must add documentation to the
toctree so that it can be rendered as the API documentation. In Qiskit Experiments, we
have a separate tables of contents for each experiment module (e.g. [characterization
experiments](https://qiskit-extensions.github.io/qiskit-experiments/apidocs/mod_characterization.html))
and for the [entire
library](https://qiskit-extensions.github.io/qiskit-experiments/apidocs/library.html). Thus we
should add document to the tree of a particular module and then reference it to the
entire module.

As an example, when writing the characterization experiment and analysis, first add your
documentation to the table of contents of the module:

```buildoutcfg
qiskit_experiments/library/characterization/__init__.py
    """
   .. currentmodule:: qiskit_experiments.library.characterization
   
   Experiments
   ===========
   .. autosummary::
       :toctree: ../stubs/
       :template: autosummary/experiment.rst
       
       MyExperiment1
       MyExperiment2
    
   Analysis
   ========
   
   .. autosummary::
       :toctree: ../stubs/
       :template: autosummary/analysis.rst

   ...
   """
   
   from my_experiment import MyExperiment1, MyExperiment2
   from my_analysis import MyAnalysis
```

Note that there are different stylesheets, `experiment.rst` and `analysis.rst`, for the
experiment class and analysis class, respectively. Take care to place your documentation
under the correct stylesheet, otherwise it may not be rendered properly. Then the table
for the entire library should be written like this:

```buildoutcfg
qiskit_experiments/library/__init__.py

    """
    .. currentmodule:: qiskit_experiments.library
    
    Characterization Experiments
    ============================
   .. autosummary::
       :toctree: ../stubs/
       :template: autosummary/experiment.rst
   
       ~characterization.MyExperiment1    
       ~characterization.MyExperiment2    
    """
    
    from .characterization import MyExperiment1, MyExperiment2
    from . import characterization
```

Here the reference start with `~`. We only add experiment classes to the table of the
entire library.
