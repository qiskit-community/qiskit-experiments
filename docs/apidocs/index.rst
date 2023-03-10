API Reference
=============

.. warning::

    This package is still under active development and it is very likely
    that there will be breaking API changes in future releases.
    If you encounter any bugs, please open an issue on
    `GitHub <https://github.com/Qiskit/qiskit-experiments/issues>`_.

The API documentation is organized into two sections below. The package modules include the framework, the
experiment library, experiment modules, and test utilities. Experiment modules are 
the main categories of the experiment library itself, such as qubit characterization
and experimental suites like tomography.

Package Modules
---------------

.. toctree::
    :maxdepth: 1

    main
    framework
    library
    data_processing
    curve_analysis
    calibration_management
    database_service
    visualization
    test

Experiment Modules
------------------

.. toctree::
    :maxdepth: 1
    
    mod_calibration
    mod_characterization
    mod_randomized_benchmarking
    mod_tomography
    mod_quantum_volume
