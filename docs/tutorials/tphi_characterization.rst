Experiment: :math:`T_\varphi` characterization
==============================================

:math:`\Gamma_\varphi` is defined as the rate of pure dephasing or
depolarization in the :math:`x - y` plane. We compute
:math:`\Gamma_\varphi` by computing :math:`\Gamma_2*`, the transverse
relaxation rate, and subtracting :math:`\Gamma_1`, the longitudinal
relaxation rate. The pure dephasing time is defined by
:math:`T_\varphi = 1/\Gamma_\varphi`. Or more precisely,
:math:`1/T_\varphi = 1/T_{2*} - 1/2T_1`

We therefore create a composite experiment consisting of a :math:`T_1`
experiment and a :math:`T_2*` experiment. From the results of these two,
we compute the results for :math:`T_\varphi.`

.. jupyter-execute::

    import numpy as np
    import qiskit
    from qiskit_experiments.library.characterization import Tphi, TphiAnalysis, T1Analysis, T2RamseyAnalysis

.. jupyter-execute::

    # An Aer simulator
    from qiskit.providers.fake_provider import FakeVigo
    from qiskit.providers.aer import AerSimulator
    from qiskit.providers.aer.noise import NoiseModel
    
    # Create a pure relaxation noise model for AerSimulator
    noise_model = NoiseModel.from_backend(
        FakeVigo(), thermal_relaxation=True, gate_error=False, readout_error=False
    )
    
    # Create a fake backend simulator
    backend = AerSimulator.from_backend(FakeVigo(), noise_model=noise_model)
    
    # Time intervals to wait before measurement for t1 and t2
    delays_t1 = np.arange(1e-6, 300e-6, 10e-6)
    delays_t2 = np.arange(1e-6, 50e-6, 2e-6)
    
    

.. jupyter-execute::

    # Create an experiment for qubit 0 with the specified time intervals
    exp = Tphi(qubit=0, delays_t1=delays_t1, delays_t2=delays_t2, osc_freq=1e5)
    
    tphi_analysis = TphiAnalysis([T1Analysis(), T2RamseyAnalysis()])
    expdata = exp.run(backend=backend, analysis=tphi_analysis, seed_simulator=101).block_for_results()
    result = expdata.analysis_results("T_phi")

.. jupyter-execute::

    # Print the result for T_phi
    print(result)


.. jupyter-execute::

    # It is possible to see the results of the sub-experiments:
    print(expdata)


.. jupyter-execute::

    print(expdata.analysis_results("T1"))

.. jupyter-execute::

    display(expdata.figure(0))


.. jupyter-execute::

    print(expdata.analysis_results("T2star"))


.. jupyter-execute::

    display(expdata.figure(1))

.. jupyter-execute::

    import qiskit.tools.jupyter
    %qiskit_copyright

