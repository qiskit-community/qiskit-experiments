Work with experiment artifacts
==============================

Problem
-------

You want to view, add, remove, and save artifacts associated with your :class:`ExperimentData` instance.

Solution
--------

Viewing artifacts
~~~~~~~~~~~~~~~~~

After an experiment that uses :class:`.CurveFit` is run, the :class:`ExperimentData` is
automatically populated with ``fit_summary`` and ``curve_data`` artifacts.

The ``fit_summary`` artifact contains a :class:`CurveFitResult``

.. jupyter-execute::

    from qiskit_ibm_runtime.fake_provider import FakePerth
    from qiskit_aer import AerSimulator
    from qiskit_experiments.library import T1
    from qiskit_experiments.framework import ParallelExperiment
    import numpy as np

    backend = AerSimulator.from_backend(FakePerth())
    exp1 = T1(physical_qubits=[0], delays=np.arange(1e-6, 6e-4, 5e-5))
    exp2 = T1(physical_qubits=[1], delays=np.arange(1e-6, 6e-4, 5e-5))
    data = ParallelExperiment([exp1, exp2], flatten_results=True).run(backend).block_for_results()
    data.artifacts()

Artifacts can be accessed using either the artifact ID or the name, which does not have to be unique and
will return all artifacts with the same name:

.. jupyter-execute::

    print("Number of curve_data artifacts:", len(data.artifacts("curve_data")))
    curve_data_id = data.artifacts("curve_data")[0].artifact_id
    scatter_table = data.artifacts(curve_data_id).data
    print("The first curve_data artifact:\n", scatter_table.dataframe)
    
The artifacts in a large composite experiment with ``flatten_results=True`` can be distinguished from
each other using the :attr:`~.ArtifactData.experiment` and :attr:`~.ArtifactData.device_components`
attributes.

One useful pattern is to load raw or fitted data from ``curve_data`` for further data manipulation:

.. jupyter-execute::

    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2)

    exp_type = data.artifacts(curve_data_id).experiment
    component = data.artifacts(curve_data_id).device_components[0]

    raw_data = scatter_table.filter(category="raw").dataframe
    fitted_data = scatter_table.filter(category="fitted").dataframe
    ax1.errorbar(raw_data['xval'], raw_data['yval'], yerr=raw_data['yerr'], capsize=5)
    ax1.set_title(f"Raw data, {exp_type} experiment on {component}")
    ax2.errorbar(fitted_data['xval'], fitted_data['yval'], yerr=fitted_data['yerr'], capsize=5)
    ax2.set_title(f"Fitted data, {exp_type} experiment on {component}")
    plt.tight_layout()
    plt.show()

Adding artifacts
~~~~~~~~~~~~~~~~

You can add arbitrary serializable data as an artifact.

.. jupyter-execute::

    from qiskit_experiments.framework import ArtifactData

    new_artifact = ArtifactData(name="experiment_notes", data="Testing some new ideas.")
    data.add_artifacts(new_artifact)
    data.artifacts("experiment_notes")

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

* The :doc:`Curve Analysis </tutorials/curve_analysis>` tutorial shows how to parse and
  use the curve fit results as stored in :class:`.ScatterTable`.