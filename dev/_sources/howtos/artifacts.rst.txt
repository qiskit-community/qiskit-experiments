Work with experiment artifacts
==============================

Problem
-------

You want to view, add, remove, and save artifacts associated with your :class:`.ExperimentData` instance.

Solution
--------

Artifacts are used to store auxiliary data for an experiment that don't fit neatly in the
:class:`.AnalysisResult` model. Any data that can be serialized, such as fit data, can be added as
:class:`.ArtifactData` artifacts to :class:`.ExperimentData`.

For example, after an experiment that uses :class:`.CurveAnalysis` is run, its :class:`.ExperimentData`
object is automatically populated with ``fit_summary`` and ``curve_data`` artifacts. The ``fit_summary``
artifact has one or more :class:`.CurveFitResult` objects that contain parameters from the fit. The
``curve_data`` artifact has a :class:`.ScatterTable` object that contains raw and fitted data in a pandas
:class:`~pandas:pandas.DataFrame`.

Viewing artifacts
~~~~~~~~~~~~~~~~~

Here we run a parallel experiment consisting of two :class:`.T1` experiments in parallel and then view the output
artifacts as a list of :class:`.ArtifactData` objects accessed by :meth:`.ExperimentData.artifacts`:

.. jupyter-execute::
    :hide-code:

    # Temporary workaround for missing support in Qiskit and qiskit-ibm-runtime
    from qiskit_experiments.test.patching import patch_sampler_test_support
    patch_sampler_test_support()

.. jupyter-execute::

    from qiskit_ibm_runtime.fake_provider import FakePerth
    from qiskit_aer import AerSimulator
    from qiskit_experiments.library import T1
    from qiskit_experiments.framework import ParallelExperiment
    import numpy as np

    backend = AerSimulator.from_backend(FakePerth())
    exp1 = T1(physical_qubits=[0], delays=np.arange(1e-6, 6e-4, 5e-5))
    exp2 = T1(physical_qubits=[1], delays=np.arange(1e-6, 6e-4, 5e-5))
    data = ParallelExperiment([exp1, exp2]).run(backend).block_for_results()
    data.artifacts()

Artifacts can be accessed using either the artifact ID, which has to be unique in each
:class:`.ExperimentData` object, or the artifact name, which does not have to be unique and will return
all artifacts with the same name:

.. jupyter-execute::

    print("Number of curve_data artifacts:", len(data.artifacts("curve_data")))
    # retrieve by name and index
    curve_data_id = data.artifacts("curve_data")[0].artifact_id
    # retrieve by ID
    scatter_table = data.artifacts(curve_data_id).data
    print("The first curve_data artifact:\n")
    scatter_table.dataframe
    
In composite experiments, artifacts behave like analysis results and figures in that if
``flatten_results`` isn't ``True``, they are accessible in the :meth:`.artifacts` method of each
:meth:`.child_data`. The artifacts in a large composite experiment with ``flatten_results=True`` can be
distinguished from each other using the :attr:`~.ArtifactData.experiment` and
:attr:`~.ArtifactData.device_components`
attributes.

One useful pattern is to load raw or fitted data from ``curve_data`` for further data manipulation. You
can work with the dataframe using standard pandas dataframe methods or the built-in
:class:`.ScatterTable` methods:

.. jupyter-execute::

    import matplotlib.pyplot as plt

    exp_type = data.artifacts(curve_data_id).experiment
    component = data.artifacts(curve_data_id).device_components[0]

    raw_data = scatter_table.filter(category="raw")
    fitted_data = scatter_table.filter(category="fitted")

    # visualize the data
    plt.figure()
    plt.errorbar(raw_data.x, raw_data.y, yerr=raw_data.y_err, capsize=5, label="raw data")
    plt.errorbar(fitted_data.x, fitted_data.y, yerr=fitted_data.y_err, capsize=5, label="fitted data")
    plt.title(f"{exp_type} experiment on {component}")
    plt.xlabel('x')
    plt.ylabel('y')    
    plt.legend()
    plt.show()

Adding artifacts
~~~~~~~~~~~~~~~~

You can add arbitrary data as an artifact as long as it's serializable with :class:`.ExperimentEncoder`,
which extends Python's default JSON serialization with support for other data types commonly used with
Qiskit Experiments.

.. jupyter-execute::

    from qiskit_experiments.framework import ArtifactData

    new_artifact = ArtifactData(name="experiment_notes", data={"content": "Testing some new ideas."})
    data.add_artifacts(new_artifact)
    data.artifacts("experiment_notes")

.. jupyter-execute::

    print(data.artifacts("experiment_notes").data)

Saving and loading artifacts
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. note::
    This feature is only for those who have access to the cloud service. You can 
    check whether you do by logging into the IBM Quantum interface 
    and seeing if you can see the `database <https://quantum.ibm.com/experiments>`__.

Artifacts are saved and loaded to and from the cloud service along with the rest of the
:class:`ExperimentData` object. Artifacts are stored as ``.zip`` files in the cloud service grouped by
the artifact name. For example, the composite experiment above will generate two artifact files, ``fit_summary.zip`` and
``curve_data.zip``. Each of these zipfiles will contain serialized artifact data in JSON format named
by their unique artifact ID:

.. jupyter-execute::
    :hide-code:

    print("fit_summary.zip")
    print(f"|- {data.artifacts('fit_summary')[0].artifact_id}.json")
    print(f"|- {data.artifacts('fit_summary')[1].artifact_id}.json")
    print("curve_data.zip")
    print(f"|- {data.artifacts('curve_data')[0].artifact_id}.json")
    print(f"|- {data.artifacts('curve_data')[1].artifact_id}.json")
    print("experiment_notes.zip")
    print(f"|- {data.artifacts('experiment_notes').artifact_id}.json")

Note that for performance reasons, the auto save feature does not apply to artifacts. You must still
call :meth:`.ExperimentData.save` once the experiment analysis has completed to upload artifacts to the
cloud service.

Note also though individual artifacts can be deleted, currently artifact files cannot be removed from the
cloud service. Instead, you can delete all artifacts of that name
using :meth:`~.delete_artifact` and then call :meth:`.ExperimentData.save`.
This will save an empty file to the service, and the loaded experiment data will not contain
these artifacts.

See Also
--------

* :ref:`Curve Analysis: Data management with scatter table <data_management_with_scatter_table>` tutorial
* :class:`.ArtifactData` API documentation
* :class:`.ScatterTable` API documentation
* :class:`.CurveFitResult` API documentation
