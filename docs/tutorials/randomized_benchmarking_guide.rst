Randomized Benchmarking
=======================

Introduction
------------
A randomized benchmarking (RB) experiment is a protocol to estimate noise on particular gates.
It consists of the following steps:

1. Generate of series of circuits increasing in length on the given qubits. Every such circuit
consists of randomly selected Clifford gates. The final gate in each circuit is the inverse of all
previous gates, so that the unitary computed by the circuits is the identity and the final state
should ideally be identical to the initial state (:math:`\left|0\right\rangle`), up to a global phase.
At the end of each circuit, the qubits are measured.

2. Run the circuits on the given backend - either on the Qiskit Aer Simulator (with some noise model)
or on the IBMQ provider. Obtain a list of results.

3. Analyze the results: the analysis counts number of shots resulting in an error
(i.e., :math:`\left|1\right\rangle`) for every circuit. From this data, we infer the
Error Per Clifford (EPC) as well as error estimates for the quantum gates on the given backend.

See the `Qiskit
Textbook <https://qiskit.org/textbook/ch-quantum-hardware/randomized-benchmarking.html>`__
for an explanation on the RB method, which is based on Ref. [1, 2].

Standard RB experiment
----------------------

To run the RB experiment we must provide the following RB parameters:

-  ``qubits``: The number of qubits or list of physical qubits for the
   experiment.

-  ``lengths``: A list of RB sequences' lengths.

-  ``num_samples``: Number of samples to generate for each sequence
   length.

-  ``seed``: Seed or generator object for random number generation. If
   ``None`` then ``default_rng`` will be used.

-  ``full_sampling``: If `True`, all Cliffords are sampled independently
   for every length. If `False`, the circuits are constructed
   incrementally so that the prefix of each circuit is identical to the
   previous circuit, assuming the circuits are constructed in order of
   increasing length. The default is `False`.

The analysis results of the RB Experiment may include:

-  ``EPC``: The estimated Error Per Clifford.

-  ``alpha``: The depolarizing parameter. The fitting function is
   :math:`a \cdot \alpha^m + b`, where :math:`m` is the Clifford length

-  ``EPG``: The Error Per Gate calculated from the EPC, only for 1-qubit
   or 2-qubit quantum gates (see Ref. [3]).

Running a 1-qubit RB experiment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The Standard RB experiment provides estimates on gate errors for all the basis gates
that constitute the Clifford gates. Note that you can only obtain a single EPC value :math:`\cal E`
from a single RB experiment. Computing the error values for multiple gates :math:`\{g_i\}`
requires some assumption on the contribution of each gate to the total depolarizing error.
This is called the ``gate_error_ratio`` and is included in the analysis options.

Provided we have :math:`n_i` gates with independent error :math:`e_i` per Clifford,
the total EPC is estimated by the composition of error from every basis gate,

.. math::

    {\cal E} = 1 - \prod_{i} (1 - e_i)^{n_i} \sim \sum_{i} n_i e_i + O(e^2),

where :math:`e_i \ll 1` and the higher order terms can be ignored.

We cannot distinguish :math:`e_i` with a single EPC value :math:`\cal E` as explained,
however by defining an error ratio :math:`r_i` with respect to
some standard value :math:`e_0`, we can compute EPG :math:`e_i` for each basis gate.

.. math::

    {\cal E} \sim e_0 \sum_{i} n_i r_i

The EPG of the :math:`i` th basis gate will be

.. math::

    e_i \sim r_i e_0 = \dfrac{r_i{\cal E}}{\sum_{i} n_i r_i}.

Because EPGs are computed based on this simple assumption,
this does not necessarily represent the true gate error on the hardware.
For multiple kinds of basis gates with an unknown error ratio :math:`r_i`,
interleaved RB experiment will provide a more accurate error value :math:`e_i`.


Running a 2-qubit RB experiment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In the same way, we can compute the EPC for a two-qubit RB experiment.
However, the EPC value obtained by the experiment indicates a depolarization
which is a composition of underlying error channels for 2Q gates and 1Q gates in each qubit.
Usually 1Q gate contribution is small enough to ignore, but in case this
contribution is significant compared to the 2Q gate error,
we can decompose the contribution of 1Q gates [3].

.. math::

    \alpha_{2Q,C} = \frac{1}{5} \left( \alpha_0^{N_1/2} + \alpha_1^{N_1/2} +
     3 \alpha_0^{N_1/2} \alpha_1^{N_1/2} \right) \alpha_{01}^{N_2},

where :math:`\alpha_i` is the single qubit depolarizing parameter of channel :math:`i`,
and :math:`\alpha_{01}` is the two qubit depolarizing parameter of interest.
:math:`N_1` and :math:`N_2` are total count of single and two qubit gates, respectively.

Note that the single qubit gate sequence in the channel :math:`i` may consist of
multiple kinds of basis gates :math:`\{g_{ij}\}` with different EPG :math:`e_{ij}`.
Therefore the :math:`\alpha_i^{N_1/2}` should be computed from EPGs,
rather than directly using the :math:`\alpha_i`, which is usually a composition of
depolarizing maps of every single qubit gate.
As such, it is recommended to measure EPGs in advance, using separate single-qubit RBs.

.. math::

    \alpha_i^{N_1/2} = \alpha_{i0}^{n_{i0}} \cdot \alpha_{i1}^{n_{i1}} \cdot ...,

where :math:`\alpha_{ij}^{n_{ij}}` indicates a depolarization due to
a particular basis gate :math:`j` in the channel :math:`i`.
Here we assume EPG :math:`e_{ij}` corresponds to the depolarizing probability
of the map of :math:`g_{ij}`, and thus we can express :math:`\alpha_{ij}` with EPG.

.. math::

    e_{ij} = \frac{2^n - 1}{2^n} (1 - \alpha_{ij}) =  \frac{1 - \alpha_{ij}}{2},

for the single qubit channel :math:`n=1`. Accordingly,

.. math::

    \alpha_i^{N_1/2} = \prod_{j} (1 - 2 e_{ij})^{n_{ij}},

as a composition of depolarization from every primitive gates per qubit.
This correction will give you two EPC values as a result of the two-qubit RB experiment.
The corrected EPC should be closer to the outcome of the interleaved RB.
The EPGs of two-qubit RB are analyzed with the corrected EPC if available.


Interleaved RB experiment
-------------------------

The Interleaved RB experiment is used to estimate the gate error of a specific gate by
inserting it between every two Cliffords in the RB circuits (see Ref. [4]).

In addition to the usual RB parameters, we also need to provide:

-  ``interleaved_element``: the element to interleave, given either as a
   group element or as an instruction/circuit

The analysis results of the RB Experiment includes the following:

-  ``EPC``: The estimated error of the interleaved gate

-  ``alpha`` and ``alpha_c``: The depolarizing parameters of the
   original and interleaved RB sequences respectively

Extra analysis results include

-  ``EPC_systematic_err``: The systematic error of the interleaved gate
   error (see Ref. [4])

-  ``EPC_systematic_bounds``: The systematic error bounds of the
   interleaved gate error (see Ref. [4])

References
----------

[1] Easwar Magesan, J. M. Gambetta, and Joseph Emerson, *Robust
randomized benchmarking of quantum processes*,
https://arxiv.org/pdf/1009.3639

[2] Easwar Magesan, Jay M. Gambetta, and Joseph Emerson, *Characterizing
Quantum Gates via Randomized Benchmarking*,
https://arxiv.org/pdf/1109.6887

[3] David C. McKay, Sarah Sheldon, John A. Smolin, Jerry M. Chow, and
Jay M. Gambetta, *Three Qubit Randomized Benchmarking*,
https://arxiv.org/pdf/1712.06550

[4] Easwar Magesan, Jay M. Gambetta, B. R. Johnson, Colm A. Ryan, Jerry
M. Chow, Seth T. Merkel, Marcus P. da Silva, George A. Keefe, Mary B.
Rothwell, Thomas A. Ohki, Mark B. Ketchen, M. Steffen, *Efficient
measurement of quantum gate error by interleaved randomized
benchmarking*, https://arxiv.org/pdf/1203.4550

[5] Jay M. Gambetta, A. D. C´orcoles, S. T. Merkel, B. R. Johnson, John
A. Smolin, Jerry M. Chow, Colm A. Ryan, Chad Rigetti, S. Poletto, Thomas
A. Ohki, Mark B. Ketchen, and M. Steffen, *Characterization of
addressability by simultaneous randomized benchmarking*,
https://arxiv.org/pdf/1204.6308

.. jupyter-execute::

    import qiskit.tools.jupyter
    %qiskit_copyright
