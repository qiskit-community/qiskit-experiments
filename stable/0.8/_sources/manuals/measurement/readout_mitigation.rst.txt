Readout Mitigation
==================

Readout errors affect quantum computation during the measurement of the
qubits in a quantum device. By characterizing the readout errors, it is
possible to construct a *readout error mitigator* that is used both to
obtain a more accurate distribution of the outputs, and more accurate
measurements of expectation value for measurables.

The readout mitigator is generated from an *assignment matrix*: a
:math:`2^n \times 2^n` matrix :math:`A` such that :math:`A_{y,x}` is the
probability to observe :math:`y` given the true outcome should be
:math:`x`. The assignment matrix is used to compute the *mitigation
matrix* used in the readout error mitigation process itself.

A *Local readout mitigator* works under the assumption that readout
errors are mostly *local*, meaning readout errors for different qubits
are independent of each other. In this case, the assignment matrix is
the tensor product of :math:`n` :math:`2 \times 2` matrices, one for
each qubit, making it practical to store the assignment matrix in
implicit form, by storing the individual :math:`2 \times 2` assignment
matrices. The corresponding class in Qiskit is the 
:class:`~qiskit.result.LocalReadoutMitigator`.

A *Correlated readout mitigator* uses the full :math:`2^n \times 2^n`
assignment matrix, meaning it can only be used for small values of
:math:`n`. The corresponding class in Qiskit is the 
:class:`~qiskit.result.CorrelatedReadoutMitigator`.

This notebook demonstrates the usage of both the local and correlated
experiments to generate the corresponding mitigators.

.. note::
    This tutorial requires the :external+qiskit_aer:doc:`qiskit-aer <index>` and :external+qiskit_ibm_runtime:doc:`qiskit-ibm-runtime <index>`
    packages to run simulations.  You can install them with ``python -m pip
    install qiskit-aer qiskit-ibm-runtime``.

.. jupyter-execute::

    import numpy as np
    import matplotlib.pyplot as plt
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Operator
    from qiskit.visualization import plot_distribution
    from qiskit_experiments.data_processing import LocalReadoutMitigator
    from qiskit_experiments.library import LocalReadoutError, CorrelatedReadoutError

    from qiskit_aer import AerSimulator
    from qiskit_ibm_runtime.fake_provider import FakePerth

    backend = AerSimulator.from_backend(FakePerth())


Standard mitigation experiment
------------------------------

The default mitigation experiment is *local*, meaning error probability
is measured individually for each qubit. The experiment generates two
circuits, one for all “0” and one for all “1” results.

.. jupyter-execute::

    shots = 1024
    qubits = [0,1,2,3]
    num_qubits = len(qubits)

    exp = LocalReadoutError(qubits)
    for c in exp.circuits():
        print(c)

    exp.analysis.set_options(plot=True)
    result = exp.run(backend)
    mitigator = result.analysis_results("Local Readout Mitigator").value

The resulting measurement matrix can be illustrated by comparing it to
the identity.

.. jupyter-execute::

    result.figure(0)


Mitigation matrices
-------------------

The individual mitigation matrices can be read off the mitigator.

.. jupyter-execute::

    for qubit in mitigator.qubits:
        print(f"Qubit: {qubit}")
        print(mitigator.mitigation_matrix(qubits=qubit))


Mitigation example
------------------

.. jupyter-execute::

    qc = QuantumCircuit(num_qubits)
    qc.sx(0)
    for i in range(1, num_qubits):
        qc.cx(i - 1, i)
    qc.measure_all()

    counts = backend.run(qc, shots=shots, seed_simulator=42, method="density_matrix").result().get_counts()
    unmitigated_probs = {label: count / shots for label, count in counts.items()}

    mitigated_quasi_probs = mitigator.quasi_probabilities(counts)
    mitigated_stddev = mitigated_quasi_probs._stddev_upper_bound
    mitigated_probs = (mitigated_quasi_probs.nearest_probability_distribution().binary_probabilities())

Probabilities
~~~~~~~~~~~~~

.. jupyter-execute::

    legend = ['Mitigated Probabilities', 'Unmitigated Probabilities']
    plot_distribution([mitigated_probs, unmitigated_probs], legend=legend, sort="value_desc", bar_labels=False)


Expectation value
-----------------

.. jupyter-execute::

    diagonal_labels = ["ZZZZ", "ZIZI", "IZII", "1ZZ0"]
    diagonals = [
        np.diag(np.real(Operator.from_label(d).to_matrix()))
        for d in diagonal_labels
    ]

    # Create a mitigator with no mitigation so that we can use its
    # expectation_values method to generate an unmitigated expectation value to
    # compare to the mitigated one.
    identity_mitigator = LocalReadoutMitigator([np.eye(2) for _ in range(4)])

    qubit_index = {i: i for i in range(num_qubits)}
    unmitigated_expectation = [identity_mitigator.expectation_value(counts, d) for d in diagonals]
    mitigated_expectation = [mitigator.expectation_value(counts, d) for d in diagonals]

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
    exp = CorrelatedReadoutError(qubits)
    for c in exp.circuits():
        print(c)


See also
--------

* API documentation: :mod:`~qiskit_experiments.library.characterization.LocalReadoutError`, 
  :mod:`~qiskit_experiments.library.characterization.CorrelatedReadoutError`
* Qiskit Textbook: `Measurement Error Mitigation <https://github.com/Qiskit/textbook/blob/main/notebooks/quantum-hardware/measurement-error-mitigation.ipynb>`__
