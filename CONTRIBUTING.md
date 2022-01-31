# Contributing

First read the overall project contributing guidelines. These are all
included in the qiskit documentation:

https://qiskit.org/documentation/contributing_to_qiskit.html

## Contributing to Qiskit Experiments

In addition to the general guidelines there are specific details for
contributing to qiskit experiments, these are documented below.

### Pull request checklist

When submitting a pull request and you feel it is ready for review,
please ensure that:

1. The code follows the code style of the project and successfully
   passes the tests. For convenience, you can execute `tox` locally,
   which will run these checks and report any issues.
2. The documentation has been updated accordingly. In particular, if a
   function or class has been modified during the PR, please update the
   *docstring* accordingly.
3. You update the relevant tutorial or write a new one. Use this [tutorial guidelines](docs/tutorials/GUIDELINES.md)
   document for further details.

   In case the PR needs to be merged without delay (e.g. for a high priority fix), open an issue for updating or adding
   the tutorial later.
4. If it makes sense for your change that you have added to include new tests that cover the changes.
5. If your change has an end user facing impact (new feature,
   deprecation, removal etc) that you have added a reno release note for that
   change and that the PR is tagged for the changelog.

### API documentation

Qiskit Experiments provides a special syntax and macros as [Sphinx](https://www.sphinx-doc.org/en/master/) extensions
to format experiment and analysis class docstring. If you implement new experiment or analysis,
you should use following style so that documentations are formatted in the same manner throughout our experiment library.
You can use standard [reStructuredText](https://www.sphinx-doc.org/en/master/usage/restructuredtext/index.html)
directives along with our syntax.

#### Experiment class documenation

You should complete the class documentation and method documentation for
`_default_experiment_options`. You can use several predefined sections for the class docstring.

```buildoutcfg
   """One line simple summary of this experiment.
   
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
       Note that this is NOT reference nor import path of the class. 
       You should write the pass to the docstring, i.e.
       
       :py:class:`~qiskit_experiments.framework.BaseAnalysis`
   
   # section: warning
       If user must take special care of the experiment (e.g. API is not stabilized) 
       you should write this section. 
   
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
       To write reference in ArXiv you can use convenient macro.
       
       .. ref_arxiv:: Auth2020a 21xx.01xxx
       
       This collects the latest article information from web and automatically 
       generates nicely formatted citation. You just need to provide ArXiv ID.
       
       For referring to the website
       
       .. ref_website:: Qiskit Experiment Github, https://github.com/Qiskit/qiskit-experiments
       
       you can use above macro, where you can provide a string for the hyperlink and 
       the destination location separated by single comma.
   
   # section: tutorial
       Optional. Link to tutorial of this experiment if exist.
   
   # section: see_also
       Optional. You can list relevant experiment or module.
       Here you cannot write any comment. 
       You just need to list absolute paths to relevant API documents, i.e.
       
       qiskit_experiments.framework.BaseExperiment
       qiskit_experiments.framework.BaseAnalysis
   """
```

You also need to provide the experiment option description in the `_default_experiment_options` method 
if you add new options. This description will be automatically propagated through child classes, 
so you don't need to manualy copy documentation.
Of course you can override documentation in the child class if it behaves differently there.

```buildoutcfg
    """Default experiment options.
    
    Experiment Options:
        opt1 (int): Description of opt1.
        opt2 (float): Description of opt2.
        opt3 (List[SomeClass]): Description of opt3.
    """
```

Note that you should use [Google docstring style](https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html).
Numpy nor other docstring styles cannot be not parsed by our Sphinx extension, 
and the section header should be named `Experiment Options` (NOT `Args`).
Since this is a private method, any other documentation except for option descriptions
is not rendered in the web page. Documentation for options are 
automatically formatted and inserted into the class documentation.

#### Analysis class documentation

You can use the same syntax and section headers for the analysis class documentation.
In addition to them, we can use extra sections, `fit_model` and `fit_parameters`, if needed.

```buildoutcfg
   """One line simple summary of this analysis.
   
   # section: overview
       Overview of this analysis.
   
   # section: fit_model
       Optional. If this analysis fits something, probably it is worth describing 
       the fit model. You can use math mode where latex commands are available.
       
       .. math::
       
           F(x) = a\exp(x) + b
       
       It is recommended to remove `*` symbol for multiplication (looks ugly in math mode), 
       and you should carefully choose parameter name so that symbols matches with
       variable names shown in analysis results. You can write symbol :math:`a` here too.
   
   # section: fit_parameters
       Optional. Description for fit parametres in the model.
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

You also need to provide description for analysis class options in `_default_options` method.

```buildoutcfg
    """Default analysis options.
    
    Analysis Options:
        opt1 (int): Description of opt1.
        opt2 (float): Description of opt2.
        opt3 (List[SomeClass]): Description of opt3.
    """
```

This is the same syntax with experiment options in the experiment class.
Note that header should be named `Analysis Options` to be parsed correctly.

#### Rendering

After you complete documentation of your classes, you must add documentation to the toctree 
so that it can be rendered as the API documentation.
In Qiskit Experiments, we have different tables of contents per 
experiment module (e.g. [characterization](https://qiskit.org/documentation/experiments/apidocs/mod_characterization.html)) 
and for the [entire library](https://qiskit.org/documentation/experiments/apidocs/library.html).
Thus we should add document to the tree of the paticular module and then add reference
of it to the entire module.

In case of writing the characterization experiment and analysis, first you add your documentation
to the table of the module.

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

Note that there are different stylesheets `experiment.rst` and `analysis.rst` for
experiment class and analysis class, respectively.
You need to take care to place your documentation under the correct stylesheet
otherwise it may not be rendered properly. Then for the table of the entire library,

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

here the reference start with `~`. We only add experiment classes to the table of entire library.

### Changelog generation

The changelog is automatically generated as part of the release process
automation. This works through a combination of the git log and the pull
request. When a release is tagged and pushed to github the release automation
bot looks at all commit messages from the git log for the release. It takes the
PR numbers from the git log (assuming a squash merge) and checks if that PR had
a `Changelog:` label on it. If there is a label it will add the git commit
message summary line from the git log for the release to the changelog.

If there are multiple `Changelog:` tags on a PR the git commit message summary
line from the git log will be used for each changelog category tagged.

The current categories for each label are as follows:

| PR Label               | Changelog Category |
| -----------------------|--------------------|
| Changelog: Deprecation | Deprecated         |
| Changelog: New Feature | Added              |
| Changelog: API Change  | Changed            |
| Changelog: Removal     | Removed            |
| Changelog: Bugfix      | Fixed              |

### Release Notes

When making any end user facing changes in a contribution we have to make sure
we document that when we release a new version of qiskit-experiments. The expectation
is that if your code contribution has user facing changes that you will write
the release documentation for these changes. This documentation must explain
what was changed, why it was changed, and how users can either use or adapt
to the change. The idea behind release documentation is that when a naive
user with limited internal knowledge of the project is upgrading from the
previous release to the new one, they should be able to read the release notes,
understand if they need to update their program which uses qiskit, and how they
would go about doing that. It ideally should explain why they need to make
this change too, to provide the necessary context.

To make sure we don't forget a release note or if the details of user facing
changes over a release cycle we require that all user facing changes include
documentation at the same time as the code. To accomplish this we use the
[reno](https://docs.openstack.org/reno/latest/) tool which enables a git based
workflow for writing and compiling release notes.

#### Adding a new release note

Making a new release note is quite straightforward. Ensure that you have reno
installed with::

    pip install -U reno

Once you have reno installed you can make a new release note by running in
your local repository checkout's root::

    reno new short-description-string

where short-description-string is a brief string (with no spaces) that describes
what's in the release note. This will become the prefix for the release note
file. Once that is run it will create a new yaml file in releasenotes/notes.
Then open that yaml file in a text editor and write the release note. The basic
structure of a release note is restructured text in yaml lists under category
keys. You add individual items under each category and they will be grouped
automatically by release when the release notes are compiled. A single file
can have as many entries in it as needed, but to avoid potential conflicts
you'll want to create a new file for each pull request that has user facing
changes. When you open the newly created file it will be a full template of
the different categories with a description of a category as a single entry
in each category. You'll want to delete all the sections you aren't using and
update the contents for those you are. For example, the end result should
look something like::

```yaml
features:
  - |
    Introduced a new feature foo, that adds support for doing something to
    ``QuantumCircuit`` objects. It can be used by using the foo function,
    for example::

      from qiskit import foo
      from qiskit import QuantumCircuit
      foo(QuantumCircuit())

  - |
    The ``qiskit.QuantumCircuit`` module has a new method ``foo()``. This is
    the equivalent of calling the ``qiskit.foo()`` to do something to your
    QuantumCircuit. This is the equivalent of running ``qiskit.foo()`` on
    your circuit, but provides the convenience of running it natively on
    an object. For example::

      from qiskit import QuantumCircuit

      circ = QuantumCircuit()
      circ.foo()

deprecations:
  - |
    The ``qiskit.bar`` module has been deprecated and will be removed in a
    future release. Its sole function, ``foobar()`` has been superseded by the
    ``qiskit.foo()`` function which provides similar functionality but with
    more accurate results and better performance. You should update your calls
    ``qiskit.bar.foobar()`` calls to ``qiskit.foo()``.
```

You can also look at other release notes for other examples.

You can use any restructured text feature in them (code sections, tables,
enumerated lists, bulleted list, etc) to express what is being changed as
needed. In general you want the release notes to include as much detail as
needed so that users will understand what has changed, why it changed, and how
they'll have to update their code.

After you've finished writing your release notes you'll want to add the note
file to your commit with `git add` and commit them to your PR branch to make
sure they're included with the code in your PR.

##### Linking to issues

If you need to link to an issue or other github artifact as part of the release
note this should be done using an inline link with the text being the issue
number. For example you would write a release note with a link to issue 12345
as:

```yaml
fixes:
  - |
    Fixes a race condition in the function ``foo()``. Refer to
    `#12345 <https://github.com/Qiskit/qiskit-experiments/issues/12345>` for more
    details.
```

#### Generating the release notes

After release notes have been added if you want to see what the full output of
the release notes. In general the output from reno that we'll get is a rst
(ReStructuredText) file that can be compiled by
[sphinx](https://www.sphinx-doc.org/en/master/). To generate the rst file you
use the ``reno report`` command. If you want to generate the full qiskit-experiments release
notes for all releases (since we started using reno during 0.9) you just run::

    reno report

but you can also use the ``--version`` argument to view a single release (after
it has been tagged::

    reno report --version 0.9.0

At release time ``reno report`` is used to generate the release notes for the
release and the output will be submitted as a pull request to the documentation
repository's [release notes file](
https://github.com/Qiskit/qiskit/blob/master/docs/release_notes.rst)

#### Building release notes locally

Building The release notes are part of the standard qiskit-experiments documentation
builds. To check what the rendered html output of the release notes will look
like for the current state of the repo you can run: `tox -edocs` which will
build all the documentation into `docs/_build/html` and the release notes in
particular will be located at `docs/_build/html/release_notes.html`

### Test

Once you've made a code change, it is important to verify that your change
does not break any existing tests and that any new tests that you've added
also run successfully. Before you open a new pull request for your change,
you'll want to run the test suite locally.

The easiest way to run the test suite is to use
[**tox**](https://tox.readthedocs.io/en/latest/#). You can install tox
with pip: `pip install -U tox`. Tox provides several advantages, but the
biggest one is that it builds an isolated virtualenv for running tests. This
means it does not pollute your system python when running. Additionally, the
environment that tox sets up matches the CI environment more closely and it
runs the tests in parallel (resulting in much faster execution). To run tests
on all installed supported python versions and lint/style checks you can simply
run `tox`. Or if you just want to run the tests once run for a specific python
version: `tox -epy37` (or replace py37 with the python version you want to use,
py35 or py36).

If you just want to run a subset of tests you can pass a selection regex to
the test runner. For example, if you want to run all tests that have "dag" in
the test id you can run: `tox -epy37 -- dag`. You can pass arguments directly to
the test runner after the bare `--`. To see all the options on test selection
you can refer to the stestr manual:
https://stestr.readthedocs.io/en/stable/MANUAL.html#test-selection

If you want to run a single test module, test class, or individual test method
you can do this faster with the `-n`/`--no-discover` option. For example:

to run a module:
```
tox -epy37 -- -n test.python.test_examples
```
or to run the same module by path:

```
tox -epy37 -- -n test/python/test_examples.py
```
to run a class:

```
tox -epy37 -- -n test.python.test_examples.TestPythonExamples
```
to run a method:
```
tox -epy37 -- -n test.python.test_examples.TestPythonExamples.test_all_examples
```

##### STDOUT/STDERR and logging capture

When running tests in parallel using `stestr` either via tox, the Makefile
(`make test_ci`), or in CI we set the env variable
`QISKIT_TEST_CAPTURE_STREAMS` which will capture any text written to stdout,
stderr, and log messages and add them as attachments to the tests run so
output can be associated with the test case it originated from. However, if
you run tests with `stestr` outside of these mechanisms by default the streams
are not captured. To enable stream capture just set the
`QISKIT_TEST_CAPTURE_STREAMS` env variable to `1`. If this environment
variable is set outside of running with `stestr` the streams (STDOUT, STDERR,
and logging) will still be captured but **not** displayed in the test runners
output. If you are using the stdlib unittest runner a similar result can be
accomplished by using the
[`--buffer`](https://docs.python.org/3/library/unittest.html#command-line-options)
option (e.g. `python -m unittest discover --buffer ./test/python`).

### Code style

The qiskit-experiments repo uses black for code formatting and style and pylint
is run to do a deeper check on the code. You can run these checks locally with

```
tox -elint
```

If there is a code formatting issue identified by black you can just run
``black`` locally to fix this (or ``tox -eblack`` which will install it and
run it).


### Development Cycle

The development cycle for qiskit-experiments is all handled in the open using
the project boards in Github for project management. We use milestones
in Github to track work for specific releases. The features or other changes
that we want to include in a release will be tagged and discussed in Github.
As we're preparing a new release we'll document what has changed since the
previous version in the release notes.

### Branches

* `master`:

The master branch is used for development of the next version of qiskit-experiments.
It will be updated frequently and should not be considered stable. The API
can and will change on master as we introduce and refine new features.

* `stable/*` branches:
Branches under `stable/*` are used to maintain released versions of qiskit-experiments.
It contains the version of the code corresponding to the latest release for
that minor version on pypi. For example, stable/0.8 contains the code for the
0.8.2 release on pypi. The API on these branches are stable and the only changes
merged to it are bugfixes.

### Release cycle

When it is time to release a new minor version of qiskit-experiments we will:

1.  Create a new tag with the version number and push it to github
2.  Change the `master` version to the next release version.

The release automation processes will be triggered by the new tag and perform
the following steps:

1.  Create a stable branch for the new minor version from the release tag
    on the `master` branch
2.  Build and upload binary wheels to pypi
3.  Create a github release page with a generated changelog
4.  Generate a PR on the meta-repository to bump the qiskit-experiments version and
    meta-package version.

The `stable/*` branches should only receive changes in the form of bug
fixes.
