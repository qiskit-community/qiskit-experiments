# Contributing Guide

To contribute to Qiskit Experiments, first read the overall [Qiskit project contributing
guidelines](https://github.com/Qiskit/qiskit/blob/main/CONTRIBUTING.md). In addition
to the general guidelines, the specific guidelines for contributing to Qiskit
Experiments are documented below.

Contents:

- [Contributing Guide](#contributing-guide)
    - [Proposing a new experiment](#proposing-a-new-experiment)
    - [Choosing an issue to work on](#choosing-an-issue-to-work-on)
    - [Pull request checklist](#pull-request-checklist)
    - [Testing your code](#testing-your-code)
      - [STDOUT/STDERR and logging capture](#stdoutstderr-and-logging-capture)
      - [Other testing related settings](#other-testing-related-settings)
    - [Code style](#code-style)
    - [Changelog generation](#changelog-generation)
    - [Release notes](#release-notes)
      - [Adding a new release note](#adding-a-new-release-note)
        - [Linking to issues](#linking-to-issues)
      - [Generating release notes](#generating-release-notes)
    - [Documentation](#documentation)
      - [Updating the documentation](#updating-the-documentation)
      - [Building documentation locally](#building-documentation-locally)
    - [Deprecation policy](#deprecation-policy)
      - [Adding deprecation warnings](#adding-deprecation-warnings)
    - [Development cycle](#development-cycle)
    - [Branches](#branches)
    - [Release cycle](#release-cycle)

### Proposing a new experiment

We welcome suggestions for new experiments to be added to Qiskit Experiments. Good
candidates for experiments should be either be well-known or based upon a research paper
or equivalent source, with a use case that is of interest to the Qiskit and quantum
experimentalist community.

If there is an experiment you would like to see added, you can propose it by creating a
[new experiment proposal
issue](https://github.com/Qiskit-Community/qiskit-experiments/issues/new?assignees=&labels=enhancement&template=NEW_EXPERIMENT.md&title=)
in GitHub. The issue template will ask you to fill in details about the experiment type,
protocol, analysis, and implementation, which will give us the necessary information to
decide whether the experiment is feasible to implement and useful to include in our
package library.

### Choosing an issue to work on
We use the following labels to help non-maintainers find issues best suited to their
interests and experience level:

* [good first
  issue](https://github.com/Qiskit-Community/qiskit-experiments/issues?q=is%3Aopen+is%3Aissue+label%3A%22good+first+issue%22)
  - these issues are typically the simplest available to work on, perfect for newcomers.
  They should already be fully scoped, with a clear approach outlined in the
  descriptions.
* [help
  wanted](https://github.com/Qiskit-Community/qiskit-experiments/issues?q=is%3Aopen+is%3Aissue+label%3A%22help+wanted%22)
  - these issues are generally more complex than good first issues. They typically cover
  work that core maintainers don't currently have capacity to implement and may require
  more investigation/discussion. These are a great option for experienced contributors
  looking for something a bit more challenging.

### Pull request checklist

When submitting a pull request for review, please ensure that:

1. The code follows the code style of the project and successfully passes the tests.
2. The API documentation has been updated accordingly.
3. You have updated the relevant documentation or written new docs. In case the PR needs
   to be merged without delay (e.g. for a high priority fix), open an issue for updating
   or adding the documentation later.
4. You've added tests that cover the changes you've made, if relevant.
5. If your change has an end user facing impact (new feature, deprecation, removal,
   etc.), you've added or updated a reno release note for that change and tagged the PR
   for the changelog.
6. If your code requires a change to dependencies, you've updated the corresponding
   sections of `pyproject.toml`: `project.dependencies` for core dependencies,
   `project.optional-dependencies` for dependencies for optional features, and
    `dependency-groups.dev` for dependencies required for running tests and
    building documentation.

The sections below go into more detail on the guidelines for each point.

### Testing your code

It is important to verify that your code changes don't break any existing tests and that
any new tests you've added also run successfully. Before you open a new pull request for
your change, you'll want to run the test suite locally.

The easiest way to run the test suite is to use
[**tox**](https://tox.readthedocs.io/en/latest/#). You can install tox with pip: `pip
install -U tox`. Tox provides several advantages, but the biggest one is that it builds
an isolated virtualenv for running tests. This means it does not pollute your system
python when running. Additionally, the environment that tox sets up matches the CI
environment more closely and it runs the tests in parallel (resulting in much faster
execution). To run tests on all installed supported python versions and lint/style
checks you can simply run `tox`. Or if you just want to run the tests once for a
specific python version such as 3.10: `tox -epy310`.

If you just want to run a subset of tests you can pass a selection regex to the test
runner. For example, if you want to run all tests that have "dag" in the test id you can
run: `tox -- dag`. You can pass arguments directly to the test runner after the bare
`--`. To see all the options on test selection you can refer to the stestr manual:
https://stestr.readthedocs.io/en/stable/MANUAL.html#test-selection

If you want to run a single test module, test class, or individual test method you can
do this faster with the `-n`/`--no-discover` option. For example, to run a module:
```
tox -epy310 -- -n test.framework.test_composite
```

To run a class:
```
tox -epy310 -- -n test.framework.test_composite.TestCompositeExperimentData
```

To run a method:
```
tox -epy310 -- -n test.framework.test_composite.TestCompositeExperimentData.test_composite_save_load
```

Note that tests will fail automatically if they do not finish execution within 60 seconds.

#### Alternatives to `tox`

If you prefer not to use `tox`, the required test environment dependencies can be installed by using the `extras` optional dependency specifier and the `dev` dependency group.
With `pip` version 25.1, installing all of the dependencies could be done with:

    pip install -e .[extras] --group dev

Prior to `pip` 25.1, the `dev` group can be installed with

    python -m dependency_groups dev

after installing `dependency-groups` (`pip install dependency-groups`).
The `tox` configuration should still be used as a reference for the preferred testing commands and environment variables.

#### STDOUT/STDERR and logging capture

When running tests in parallel using `stestr` either via tox
or in CI, we set the env variable `QISKIT_TEST_CAPTURE_STREAMS`, which will
capture any text written to stdout, stderr, and log messages and add them as attachments
to the tests run so output can be associated with the test case it originated from.
However, if you run tests with `stestr` outside of these mechanisms, by default the
streams are not captured. To enable stream capture, just set the
`QISKIT_TEST_CAPTURE_STREAMS` env variable to `1`. If this environment variable is set
outside of running with `stestr`, the streams (STDOUT, STDERR, and logging) will still
be captured but **not** displayed in the test runners output. If you are using the
stdlib unittest runner, a similar result can be accomplished by using the
[`--buffer`](https://docs.python.org/3/library/unittest.html#command-line-options)
option (e.g. `python -m unittest discover --buffer ./test/python`).

#### Other testing related settings

The test code defines some environment variables that may occasionally be useful to set:

+ `TEST_TIMEOUT`: An integer representing the maximum time a test can take
  before it is considered a failure.
+ `QE_USE_TESTTOOLS`: Set this variable to `FALSE`, `0`, or `NO` to have the
  tests use `unittest.TestCase` as the base class. Otherwise, the default is
`testtools.TestCase` which is an extension of `unittest.TestCase`. In some
situations, a developer may wish to use a workflow that is not compatible with
the `testtools` extensions.

### Code style

The qiskit-experiments repository uses `black` for code formatting and style and
`pylint` for linting. You can run these checks locally with

```
tox -elint
```

If there is a code formatting issue identified by black you can just run ``black``
locally to fix this (or ``tox -eblack`` which will install it and run it).

Because `pylint` analysis can be slow, there is also a `tox -elint-incr` target, which
only applies `pylint` to files which have changed from the source github. On rare
occasions this will miss some issues that would have been caught by checking the
complete source tree, but makes up for this by being much faster (and those rare
oversights will still be caught by the CI after you open a pull request).

### Changelog generation

The changelog is automatically generated as part of the release process automation. This
works through a combination of the git log and the pull request. When a release is
tagged and pushed to github the release automation bot looks at all commit messages from
the git log for the release. It takes the PR numbers from the git log (assuming a squash
merge) and checks if that PR had a `Changelog:` label on it. If there is a label it will
add the git commit message summary line from the git log for the release to the
changelog.

If there are multiple `Changelog:` tags on a PR, the git commit message summary line
from the git log will be used for each changelog category tagged.

The current categories for each label are as follows:

| PR Label               | Changelog Category |
| ---------------------- | ------------------ |
| Changelog: Deprecation | Deprecated         |
| Changelog: New Feature | Added              |
| Changelog: API Change  | Changed            |
| Changelog: Removal     | Removed            |
| Changelog: Bugfix      | Fixed              |

### Release notes

All end user facing changes have to be documented with each release of Qiskit
Experiments. The expectation is that if your code contribution has user facing changes
that you will write the release documentation for these changes in the form of a release
note. This note must explain what was changed, why it was changed, and how users can
either use or adapt to the change. When a naive user with limited internal knowledge of
the project is upgrading from the previous release to the new one, they should be able
to read the release notes, understand if they need to update their existing code which
uses Qiskit Experiments, and how they would go about doing that. It ideally should
explain why they need to make this change too, to provide the necessary context.

To make sure we don't forget a release note or the details of user facing changes over a
release cycle, we require that all pull requests with user facing changes include a note
describing the changes along with the code. To accomplish this, we use the
[reno](https://docs.openstack.org/reno/latest/) tool which enables a git based workflow
for writing and compiling release notes.

Note that these notes are meant to document a release, not individual pull requests. So
if your pull request updates or reverts a change made in a previous pull request in the
same release, you should update the corresponding release note that already exists
instead of writing a new one, which would be confusing to users. You can use `git blame`
to see which previous pull requests(s) are relevant to the part of the code you're
editing, and see whether they are tagged with the milestone of the current release.

#### Adding a new release note

Making a new release note is quite straightforward. Ensure that you have reno installed
with:

    pip install -U reno

Once you have reno installed, you can make a new release note by running in your local
repository checkout's root:

    reno new short-description-string

where short-description-string is a brief string (with no spaces) that describes what's
in the release note. This will become the prefix for the release note file. Once that is
run, it will create a new yaml file in `releasenotes/notes`. Then open that yaml file in
a text editor and write the release note.

The basic structure of a release note is restructured text in yaml lists under category
keys. You add individual items under each category, and they will be grouped
automatically by release when the release notes are compiled. A single file can have as
many entries in it as needed, but to avoid potential conflicts, you'll want to create a
new file for each pull request that has user facing changes. When you open the newly
created file it will be a full template of the different categories with a description
of a category as a single entry in each category. You'll want to delete all the sections
you aren't using and update the contents for those you are. For example, the end result
should look something like:

```yaml
features_expclass:
  - |
    Introduced a new feature foo that adds support for doing something to
    :class:`~qiskit.circuit.QuantumCircuit` objects. It can be used by using the foo function,
    for example::

      from qiskit import foo
      from qiskit import QuantumCircuit
      foo(QuantumCircuit())

  - |
    The :class:`~qiskit.circuit.QuantumCircuit` class has a new method :meth:`.foo`. This is
    the equivalent of calling :func:`qiskit.foo` to do something to your
    QuantumCircuit. This is the equivalent of running :func:`qiskit.foo` on
    your circuit, but provides the convenience of running it natively on
    an object. For example::

      from qiskit import QuantumCircuit

      circ = QuantumCircuit()
      circ.foo()

deprecations:
  - |
    The :mod:`qiskit.bar` module has been deprecated and will be removed in a
    future release. Its sole function, :func:`foobar` has been superseded by the
    :func:`qiskit.foo` function which provides similar functionality but with
    more accurate results and better performance. You should update your calls
    :func:`qiskit.bar.foobar` calls to :func:`qiskit.foo`.
```

Note that we are using subsections within the `features`, `upgrade`, and `fixes` sections to
organize the notes by functional area. We strongly encourage you to file your note under the most
appropriate category. You can see the current list of categories in
[release_notes/config.yaml](https://github.com/Qiskit-Community/qiskit-experiments/blob/main/releasenotes/config.yaml).

You can use any restructured text feature in them (code sections, tables, enumerated
lists, bulleted list, etc.) to express what is being changed as needed. In general, you
want the release notes to include as much detail as needed so that users will understand
what has changed, why it changed, and how they'll have to update their code.

After you've finished writing your release notes you'll want to add the note file to
your commit with `git add` and commit them to your PR branch to make sure they're
included with the code in your PR.

##### Linking to issues

If you need to link to an issue or another GitHub artifact as part of the release note,
this should be done using an inline link with the text being the issue number. For
example you would write a release note with a link to issue 12345 as:

```yaml
fixes:
  - |
    Fixed a race condition in the function ``foo()``. Refer to
    `#12345 <https://github.com/Qiskit-Community/qiskit-experiments/issues/12345>` for more
    details.
```

#### Generating release notes

After adding your release note, you should generate it to check that the output looks as
expected. In general, the output from reno that we'll get is a `.rst` (ReStructuredText)
file that can be compiled by [sphinx](https://www.sphinx-doc.org/en/master/). If you
want to generate the full Qiskit Experiments release notes for all releases, simply run:

    reno report

You can also use the ``--version`` argument to view a single release (after it has been
tagged):

    reno report --version 0.9.0

At release time, ``reno report`` is used to generate the release notes for the release,
and the output will be submitted as a pull request to the documentation repository's
[release notes file](
https://github.com/Qiskit-Community/qiskit-experiments/blob/main/docs/release_notes.rst).

### Documentation

The [Qiskit Experiments documentation](https://qiskit-community.github.io/qiskit-experiments) is
rendered from `.rst` files as well as experiment and analysis class docstrings into HTML
files.

#### Updating the documentation

Any change that would affect existing documentation, or a new feature that requires a
documentation, should be updated correspondingly. Before updating, review the [existing
documentation](https://qiskit-community.github.io/qiskit-experiments) for their style and
content, and read the [documentation guidelines](docs/GUIDELINES.md) for further
details.

#### Building documentation locally

To check what the rendered html output of the API documentation, tutorials, and release
notes will look like for the current state of the repo, run:

    tox -e docs

This will build all the documentation into `docs/_build/html`. The main page
`index.html` will link to the relevant pages in the subdirectories, or you can navigate
manually:

* `tutorials/`: Contains the built tutorials.
* `howtos/`: Contains the built how-to guides.
* `manuals/`: Contains the built experiment manuals.
* `apidocs/`:  Contains the API docs automatically compiled from module docstrings.
* `release_notes.html`: Contains the release notes.

Sometimes Sphinx's caching can get in a bad state. First, try running `tox -e docs-clean`, which 
will remove Sphinx's cache. If you are still having issues, try adding `-r` your command, 
e.g. `tox -e docs -r`. `-r` tells Tox to reinstall the dependencies. If you encounter a build 
error involving `config-inited`, you need to be in the root of
the qiskit-experiments git repository then run `git remote add upstream
https://github.com/Qiskit-Community/qiskit-experiments` and `git fetch upstream` before building.

There are a few other build options available:

* `tox -e docs-minimal`: build documentation without executing Jupyter code cells
* `tox -e docs-parallel`: do a full build with multiprocessing (may crash on Macs)

### Deprecation policy

Any change to the existing package code that affects how the user interacts with the package
should give the user clear instructions and advanced warning if the change is nontrivial.
Qiskit Experiments's deprecation policy is based on [Qiskit's
policy](https://github.com/Qiskit/qiskit/blob/1.0.0rc1/DEPRECATION.md) prior to its 1.0 release, but
we impose less stringent requirements such that developers can iterate more quickly.
Deprecations and feature removals can only happen on minor releases and not on patch releases.

The deprecation policy depends on the significance of the user-facing change, which we have divided into
three categories:

A **core feature change** is one that affects how the framework functions, for example a
change to `BaseExperiment`. The timeline for deprecating an existing core feature is as follows:

* Minor release 1: An alternative path is provided. A `PendingDeprecationWarning` 
  should be issued when the old path is used, indicating to users how to switch to
  the new path and the release in which the old path will no longer be available. The
  developer may choose to directly deprecate the feature and issue a `DeprecationWarning` instead,
  in which case the release note should indicate the feature has been deprecated and how to switch
  to the new path.
* Minor release 2: The `PendingDeprecationWarning` becomes a `DeprecationWarning`, or the
  `DeprecationWarning` remains in place. The release note should indicate the feature has
  been deprecated and how to switch to the new path.
* Minor release 3: The old feature is removed. The release note should indicate that the feature has
  been removed and how to switch to the new path.

If the three-release cycle takes fewer than three months, the feature removal must wait for more
releases until three months has elapsed since the first issuing of the `PendingDeprecationWarning`
or `DeprecationWarning`.

A **non-core feature change** may be a change to a specific experiment class or modules such as the
plotter. The timeline is shortened for such a change:

* Minor release 1: An alternative path is provided. A `DeprecationWarning` should be issued
  when the old path is used, indicating to users how to switch to the new path and the release
  in which the old path will no longer be available.
* Minor release 2: The old feature is removed. The release note should indicate that the feature has
  been removed and how to switch to the new path.

Lastly, a **minor, non-core change** could be a cosmetic change such as output file names or a
change to helper functions that isn't directly used in the package codebase. These can be made in
one release without a deprecation process as long as the change is clearly described in the
release notes.

#### Adding deprecation warnings

We use the deprecation wrappers in [Qiskit
Utilities](https://docs.quantum.ibm.com/api/qiskit/utils) to add warnings:

```python

  from qiskit.utils.deprecation import deprecate_func

  @deprecate_func(
      since="0.5",
      additional_msg="Use ``new_function`` instead.",
      pending=True,
      removal_timeline="after 0.7",
      package_name="qiskit-experiments",
  )
  def old_function(*args, **kwargs):
      pass
  
  def new_function(*args, **kwargs):
      pass
```

Note that all warnings emitted by Qiskit Experiments, including pre-deprecation and deprecation
warnings, will cause the CI to fail, but features up for deprecation should continue to be tested
until their removal. For more information on how to use wrappers and test deprecated functionality,
consult [Qiskit's
policy](https://github.com/Qiskit/qiskit/blob/1.0.0rc1/DEPRECATION.md#issuing-deprecation-warnings).

### Development cycle

The development cycle for Qiskit Experiments is all handled in the open using project
boards in GitHub for project management. We use
[milestones](https://github.com/Qiskit-Community/qiskit-experiments/milestones) in GitHub to track
work for specific releases. Features or other changes that we want to include in a
release will be tagged and discussed in GitHub.

### Branches

* `main`: The main branch is used for development of the next version of
qiskit-experiments. It will be updated frequently and should not be considered stable.
The API can and will change on main as we introduce and refine new features.

* `stable/*` branches: Branches under `stable/*` are used to maintain released versions
of qiskit-experiments. It contains the version of the code corresponding to the latest
release for that minor version on pypi. For example, `stable/0.1` contains the code for
the 0.1.0 release on pypi. The API on these branches are stable and the only changes
merged to it are bug fixes.

### Release cycle

When it is time to release a new minor version of qiskit-experiments, we will:

1.  Create a new tag with the version number and push it to github
2.  Change the `main` version to the next release version.

The release automation processes will be triggered by the new tag and perform the
following steps:

1.  Create a stable branch for the new minor version from the release tag on the `main`
    branch
2.  Build and upload binary wheels to PyPI
3.  Create a github release page with a generated changelog
4.  Generate a PR on the meta-repository to bump the qiskit-experiments version and
    meta-package version.

The `stable/*` branches should only receive changes in the form of bug fixes. If you're making a bug fix PR that you believe should be backported to the current stable release, tag it with `backport stable potential`.

