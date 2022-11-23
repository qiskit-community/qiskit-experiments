Writing a custom experiment
===========================

In this tutorial, we'll use what we've learned so far to make a full experiment from
the :class:`.BaseExperiment` template.

A randomized measurement experiment
===================================


This experiment creates a list of copies of an input circuit
and randomly samples an N-qubit Paulis to apply to each one before
a final N-qubit Z-basis measurement to randomized the expected
ideal output bitstring in the measured.

The analysis uses the applied Pauli frame of a randomized
measurement experiment to de-randomize the measured counts
and combine across samples to return a single counts dictionary
the original circuit.

This has the effect of Pauli-twirling and symmetrizing the
measurement readout error. 