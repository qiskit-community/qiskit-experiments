# Guidelines for writing documentation

Read the [contributing guidelines](CONTRIBUTING.md) before proceeding.
## Introduction

Qiskit Experiments Documentation is split into four sections:

- tutorials for learning the package from the ground up
- how-tos for solving specific problems
- guides for information on specific experiments
- API reference for technical documentation on 

Below we provide templates and guidelines for each of these types of documentation.

### How-to recipe

The title of a how-to should clearly describe what problem it's solving. It should be an action that follows "How to". The text itself has up to four sections, but only the first two are required:

- Problem: This section should describe the user problem that your recipe is providing a direct solution for in second person. This should ideally be a one-liner so that users can quickly scan it and see if it’s relevant to what they’re trying to do.

- Solution: This section should describe possible solutions for the problem with code snippets and text before and after that describe what is needed to run the code, as well as what it generates and how this solves the problem.

- Discussion: This section can go into detail on when this kind of problem can arise, caveats to running the code, and any related.

- See also: Links to other relevant documentation or resources.

Here is a template for how-tos:

```
Write a how-to
===============

Problem
-------

You want to write a how-to.

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



### Experiment guides

The main goal of `qiskit-experiment` experiment guides is to serve as user guides for
the various package components such as the characterization and calibration 
experiments. To this end, each documentation should cover the main (if not all) use-cases
of the documented functionality, including code examples and expected outputs.
Another objective of the documentation is to provide the user with basic background
on each experiment method. Hence a good practice would
be to have in the beginning of the documentation a short background explanation, 
preferably 1 or 2 paragraphs long which includes the main literature references 
as well as a link to the relevant chapter in the Qiskit textbook, if available. See for example the
[Randomized Benchmarking](randomized_benchmarking.ipynb) documentation.

Below are more concrete guidelines pertaining to various documentation aspects: 

## Formatting guidelines 
* For experiments, documentation title should be just the name of the experiment. Use regular capitalization. 
* For sub titles of how-to steps - use present progressive. E.e. "Saving exp data to the DB" (instead of "Save exp data to the DB")
* Use math notation as much as possible (e.g. use $\frac{\pi}{2}$ instead of pi-half or pi/2)
* Use headers, subheaders, subsubheaders etc. for hierarchical text organization. No need to number the headers
* Use device names as shown in the IBM Quantum Services dashboard, e.g. ibmq_lima instead of IBMQ Lima
* put identifier names (e.g. osc_freq) in code blocks using backticks, i.e. `osc_freq` 
 
## Content guidelines 

* First section should be a general explanation on the topic. Put 2-3 most relevant references (papers and Qiskit textbook)
* Cover the common use-cases of the documented functionality (e.g. experiment) 
* For each use-case, provide an example output, such as console printings and plot figures 
* Cover all the required and common params (e.g. experiment and analysis options)
* For an experiment documentation, cover using the experiment in a composite experiment setting


