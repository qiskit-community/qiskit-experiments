Readout Mitigation
==================

Readout mitigation is part of ``qiskit-terra``. The readout mitigators
can be initalized based on existing backend data, or via a readout
mitigation experiment.

In a readout mitigation experiment, simple circuits are generated for
various combinations of “0” and “1” readout values. The results give us
a matrix describing the probability to obtain a wrong measurement. This
matrix is used to initialize the readout mitigation object, which is
given as the result of the expriment.

.. jupyter-execute::

    import numpy as np
    import matplotlib.pyplot as plt
    from qiskit import QuantumCircuit
    from qiskit.visualization import plot_histogram
    from qiskit_experiments.library import ReadoutMitigationExperiment
    # For simulation
    from qiskit.providers.aer import AerSimulator
    from qiskit.test.mock import FakeParis
    
    from qiskit.result.mitigation.utils import (
        expval_with_stddev,
        str2diag,
        counts_probability_vector
    )
    
    backend = AerSimulator.from_backend(FakeParis())

.. jupyter-execute::

    SHOTS = 1024
    qubits = [0,1,2,3]
    num_qubits = len(qubits)

Standard mitigation experiment
==============================

The default mitigation experiment is *local*, meaning error probability
is measured individually for each qubit. The experiment generates two
circuits, one for all “0” and one for all “1” results.

.. jupyter-execute::

    exp = ReadoutMitigationExperiment(qubits)
    for c in exp.circuits():
        print(c)


.. jupyter-execute::

    result = exp.run(backend).block_for_results()
    mitigator = result.analysis_results(0).value

The resulting measurement matrix can be illustrated by comparing it to
the identity.

.. jupyter-execute::

    result.figure(0)

Mitigation matrices
-------------------

The individual mitigation matrices can be read off the mitigator.

.. jupyter-execute::

    for m in mitigator._mitigation_mats:
        print(m)
        print()


Mitigation Example
------------------

.. jupyter-execute::

    qc = QuantumCircuit(num_qubits)
    qc.h(0)
    for i in range(1, num_qubits):
        qc.cx(i - 1, i)
    qc.measure_all()

.. jupyter-execute::

    counts = backend.run(qc, shots=SHOTS, seed_simulator=42, method="density_matrix").result().get_counts()
    unmitigated_probs = {label: count / SHOTS for label, count in counts.items()}

.. jupyter-execute::

    mitigated_quasi_probs = mitigator.quasi_probabilities(counts)
    mitigated_stddev = mitigated_quasi_probs._stddev_upper_bound
    mitigated_probs = (mitigated_quasi_probs.nearest_probability_distribution().binary_probabilities())

Probabilities
~~~~~~~~~~~~~

.. jupyter-execute::

    legend = ['Mitigated Probabilities', 'Unmitigated Probabilities']
    plot_histogram([mitigated_probs, unmitigated_probs], legend=legend, sort="value_desc", bar_labels=False)


Expectation value
-----------------

.. jupyter-execute::

    diagonal_labels = ["ZZZZ", "ZIZI", "IZII", "1ZZ0"]
    ideal_expectation = []
    diagonals = [str2diag(d) for d in diagonal_labels]
    qubit_index = {i: i for i in range(num_qubits)}
    unmitigated_probs_vector, _ = counts_probability_vector(unmitigated_probs, qubit_index=qubit_index)
    unmitigated_expectation = [expval_with_stddev(d, unmitigated_probs_vector, SHOTS) for d in diagonals]
    mitigated_expectation = [mitigator.expectation_value(counts, d) for d in diagonals]

.. jupyter-execute::

    mitigated_expectation_values, mitigated_stddev = zip(*mitigated_expectation)
    unmitigated_expectation_values, unmitigated_stddev = zip(*unmitigated_expectation)
    legend = ['Mitigated Expectation', 'Unmitigated Expectation']
    fig, ax = plt.subplots()
    X = np.arange(4)
    ax.bar(X + 0.00, mitigated_expectation_values, yerr=mitigated_stddev, color='b', width = 0.25, label="Mitigated Expectation")
    ax.bar(X + 0.25, unmitigated_expectation_values, yerr=unmitigated_stddev, color='g', width = 0.25, label="Unmitigated Expectation")
    ax.set_xticks([0.125 + i for i in range(len(diagonals))])
    ax.set_xticklabels(diagonal_labels)
    ax.legend()

Correlated readout mitigation
-----------------------------

In correlated readout mitigation on :math:`n` qubits, a circuit is
generated for each of the possible :math:`2^n` combinations of “0” and
“1”. This results in more accurate mitigation in the case where the
readout errors are correlated and not independent, but requires a large
amount of circuits and storage space, and so is infeasible for more than
a few qubits.

.. jupyter-execute::

    qubits = [0,3]
    num_qubits = len(qubits)
    exp = ReadoutMitigationExperiment(qubits, method="correlated")
    for c in exp.circuits():
        print(c)

.. jupyter-execute::

    import qiskit.tools.jupyter
    %qiskit_copyright
