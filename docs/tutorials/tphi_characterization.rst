Experiment: :math:`T_\varphi` characterization
==============================================

:math:`T_\varphi`, or :math:`1/\Gamma_\varphi`, is the pure dephasing time of
depolarization in the :math:`x - y` plane of the Bloch sphere. We compute
:math:`\Gamma_\varphi` by computing :math:`\Gamma_2`, the *transverse relaxation rate*,
and subtracting :math:`\Gamma_1`, the *longitudinal relaxation rate*. It follows that
:math:`\frac{1}{T_\varphi} = \frac{1}{T_2} - \frac{1}{2T_1}`.

We therefore create a composite experiment consisting of a :math:`T_1` experiment and a
:math:`T_2` experiment. Both Ramsey and Hahn echo experiments can be used here, with
different effects. The :math:`T_2^*` estimate of the Ramsey experiment is sensitive to
inhomogeneous broadening, low-frequency fluctuations that vary between experiments due
to :math:`1/f`-type noise. The :math:`T_{2}` estimate from the Hahn echo (defined as
:math:`T_{2E}` in [#]_) is less sensitive to inhomogeneous broadening due to its
refocusing pulse, and so it is at least as large as :math:`T_2^*`.

From the :math:`T_1` and :math:`T_2` estimates, we compute the results for :math:`T_\varphi.`

.. jupyter-execute::

    import numpy as np
    import qiskit
    from qiskit_experiments.library.characterization import Tphi

    # An Aer simulator
    from qiskit.providers.fake_provider import FakeVigo
    from qiskit_aer import AerSimulator
    from qiskit_aer.noise import NoiseModel
    
    # Create a pure relaxation noise model for AerSimulator
    noise_model = NoiseModel.from_backend(
        FakeVigo(), thermal_relaxation=True, gate_error=False, readout_error=False
    )
    
    # Create a fake backend simulator
    backend = AerSimulator.from_backend(FakeVigo(), noise_model=noise_model)
    
    # Time intervals to wait before measurement for t1 and t2
    delays_t1 = np.arange(1e-6, 300e-6, 10e-6)
    delays_t2 = np.arange(1e-6, 50e-6, 2e-6)
    
By default, the :class:`.Tphi` experiment will use the Ramsey experiment for its transverse
relaxation time estimate. We can see that the component experiments of the batch 
:class:`.Tphi` experiment are what we expect for :class:`.T1` and :class:`.T2Ramsey`:

.. jupyter-execute::

    # Create an experiment for qubit 0 with the specified time intervals
    exp = Tphi(physical_qubits=[0], delays_t1=delays_t1, delays_t2=delays_t2, osc_freq=1e5)
    exp.component_experiment(0).circuits()[-1].draw("mpl")

.. jupyter-execute::

    exp.component_experiment(1).circuits()[-1].draw("mpl")

Run the experiment and print results:

.. jupyter-execute::

    expdata = exp.run(backend=backend, seed_simulator=101).block_for_results()
    result = expdata.analysis_results("T_phi")
    print(result)

You can also retrieve the results and figures of the constituent experiments. :class:`.T1`:

.. jupyter-execute::

    print(expdata.analysis_results("T1"))
    display(expdata.figure(0))

And :class:`.T2Ramsey`:

.. jupyter-execute::

    print(expdata.analysis_results("T2star"))
    display(expdata.figure(1))

Let's now run the experiment with :class:`.T2Hahn` by setting the ``t2star`` option to ``False``:

.. jupyter-execute::

    exp = Tphi(physical_qubits=[0], delays_t1=delays_t1, delays_t2=delays_t2, num_echoes=1, t2star=False)
    
    expdata = exp.run(backend=backend, seed_simulator=101).block_for_results()
    result = expdata.analysis_results("T_phi")

.. jupyter-execute::

    print(expdata.analysis_results("T_phi"))
    display(expdata.figure(1))

As expected, because :math:`T_2 > T_2^*`, the obtained :math:`T_{\varphi}` is larger
when the Hahn echo experiment is used.

|

.. [#] Krantz, Philip, et al. "A Quantum Engineer's Guide to Superconducting Qubits." 
       `arXiv:1904.06560 (2019) <https://arxiv.org/abs/1904.06560>`_.

.. jupyter-execute::

    import qiskit.tools.jupyter
    %qiskit_copyright
