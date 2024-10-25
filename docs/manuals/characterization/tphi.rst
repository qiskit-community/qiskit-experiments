Tφ Characterization
===================

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
refocusing pulse, and so it is strictly larger than :math:`T_2^*` on a real device. In
superconducting qubits, :math:`T_2^*` tends to be significantly smaller than
:math:`T_1`, so :math:`T_2` is usually used.

From the :math:`T_1` and :math:`T_2` estimates, we compute the results for
:math:`T_\varphi.`

.. note::
    This tutorial requires the :external+qiskit_aer:doc:`qiskit-aer <index>` and :external+qiskit_ibm_runtime:doc:`qiskit-ibm-runtime <index>`
    packages to run simulations.  You can install them with ``python -m pip
    install qiskit-aer qiskit-ibm-runtime``.

.. jupyter-execute::
    :hide-code:

    # Temporary workaround for missing support in Qiskit and qiskit-ibm-runtime
    from qiskit_experiments.test.patching import patch_sampler_test_support
    patch_sampler_test_support()

.. jupyter-execute::

    import numpy as np
    import qiskit
    from qiskit_experiments.library.characterization import Tphi

    # An Aer simulator
    from qiskit_aer import AerSimulator
    from qiskit_aer.noise import NoiseModel
    from qiskit_ibm_runtime.fake_provider import FakePerth
    
    # Create a pure relaxation noise model for AerSimulator
    noise_model = NoiseModel.from_backend(
        FakePerth(), thermal_relaxation=True, gate_error=False, readout_error=False
    )
    
    # Create a fake backend simulator
    backend = AerSimulator.from_backend(FakePerth(), noise_model=noise_model)
    
    # Time intervals to wait before measurement for t1 and t2
    delays_t1 = np.arange(1e-6, 300e-6, 10e-6)
    delays_t2 = np.arange(1e-6, 50e-6, 2e-6)
    
By default, the :class:`.Tphi` experiment will use the Hahn echo experiment for its transverse
relaxation time estimate. We can see that the component experiments of the batch 
:class:`.Tphi` experiment are what we expect for :class:`.T1` and :class:`.T2Hahn`:

.. jupyter-execute::

    exp = Tphi(physical_qubits=(0,), delays_t1=delays_t1, delays_t2=delays_t2, num_echoes=1)
    exp.component_experiment(0).circuits()[-1].draw(output="mpl", style="iqp")

.. jupyter-execute::

    exp.component_experiment(1).circuits()[-1].draw(output="mpl", style="iqp")

Run the experiment and print results:

.. jupyter-execute::

    expdata = exp.run(backend=backend, seed_simulator=100).block_for_results()
    result = expdata.analysis_results("T_phi")
    print(result)

You can also retrieve the results and figures of the constituent experiments. :class:`.T1`:

.. jupyter-execute::

    print(expdata.analysis_results("T1"))
    display(expdata.figure(0))

And :class:`.T2Hahn`:

.. jupyter-execute::

    print(expdata.analysis_results("T2"))
    display(expdata.figure(1))

Let's now run the experiment with :class:`.T2Ramsey` by setting the ``t2type`` option to
``ramsey`` and specifying ``osc_freq``. Now the second component experiment is a Ramsey
experiment:

.. jupyter-execute::

    exp = Tphi(physical_qubits=(0,), 
               delays_t1=delays_t1, 
               delays_t2=delays_t2, 
               t2type="ramsey", 
               osc_freq=1e5)

    exp.component_experiment(1).circuits()[-1].draw(output="mpl", style="iqp")

Run and display results:

.. jupyter-execute::

    expdata = exp.run(backend=backend, seed_simulator=100).block_for_results()
    print(expdata.analysis_results("T_phi"))
    display(expdata.figure(1))

Because we are using a simulator that doesn't model inhomogeneous broadening, the
:math:`T_2` and :math:`T_2^*` values are not significantly different. On a real
superconducting device, :math:`T_{\varphi}` should be significantly larger when the Hahn
echo experiment is used.

References
----------

.. [#] Krantz, Philip, et al. *A Quantum Engineer's Guide to Superconducting Qubits*.
       `arXiv:1904.06560 (2019) <https://arxiv.org/abs/1904.06560>`_.

See also
--------

* API documentation: :mod:`~qiskit_experiments.library.characterization.Tphi`
