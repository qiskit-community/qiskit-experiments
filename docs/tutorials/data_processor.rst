Data Processor: Wrangling data
==============================

Data processing is the act of taking the data returned by the backend and
converting it into a format that can be analyzed.
It is implemented as a chain of data processing steps that transform various input data,
e.g. IQ data, into a desired format, e.g. population, which can be analyzed.

These data transformations may consist of multiple steps, such as kerneling and discrimination.
Each step is implemented by a :class:`~qiskit_experiments.data_processing.data_action.DataAction`
also called a `node`.

The data processor implements the :meth:`__call__` method. Once initialized, it
can thus be used as a standard python function:

.. code-block:: python

    processor = DataProcessor(input_key="memory", [Node1(), Node2(), ...])
    out_data = processor(in_data)

The data input to the processor is a sequence of dictionaries each representing the result
of a single circuit. The output of the processor is a numpy array whose shape and data type
depend on the combination of the nodes in the data processor.

Uncertainties that arise from quantum measurements or finite sampling can be taken into account
in the nodes: a standard error can be generated in a node and can be propagated
through the subsequent nodes in the data processor.
Correlation between computed values is also considered.
