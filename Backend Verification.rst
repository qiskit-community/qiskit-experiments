
Verification of backends with a Composite Experiment
=========================================================

There are information about qubit frequencies,T1, T2, 1&2 qubit gate errors 
and other properties corresponding to all qubits for each backends
at `IBM Quantum Service. <https://quantum-computing.ibm.com/services?services=systems>`_  
We are going to choose the backend for experiment, 
characterize frequency, T1, T2 and gate errors 
corresponding to every qubits to verify the quantum system 
and compare whether the results are similar to the reported ones.
As this guide is for characterizing multiple qubits, 
selecting a backend which has more than one qubit would be meaningful.
In this guide, we will try ``ibmq_lima`` backend which has 5 qubits and especially use composite experiment framework.


Brief explanation about Composite Experiment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Composite experiment framework consists of 2 experiment classes : 
``ParallelExperiment`` and ``BatchExperiment``.
These classes literally provide a way to perform a single composite experiment which made up of many component experiments. 

* ParallelExperiment 

    ParallelExperiment is suitable when running experiments each defined on independent subset of device qubits.
    For example, when we perform a RB experiment on qubit0, qubit1, qubit2, 
    it would be time saving to make each experiment as a single parallelexperiment
    which generate only 1 job by combining all the sub experiments circuits into circuits 
    which run the component gates in parallel on the respective qubits whereas 
    if they were made as independent experiments, 3 jobs would have been generated and 3 queue times. 
    Furthermore, parallelexperiment can provide similar noise environment across all qubits.

* BatchExperiment
  
    Batchexperiment is a single large job constists of all the circuits from sub experiments in series.
    If sub experiments are run on overlapping qubtis, 
    parallelexperiment is impossible and we need to use a batchexperiment. 
    For example, if we want to run 2qubit RB experiments on qubit pair 
    (1,2) and (1,4) we have to make these sub experiments performed serially
    by composing a batchexperiment. Again, it would save queue time. 

We will use ``configuration`` to check the reported backend's property. 
Since we will do 1 and 2 qubit gate RB, 
information about qubit coupling and gate length are helpful 
in finding out the preferred cx-gate direction. 
In addition, knowing maximum number of experiments per job is helpful 
to construct possible job.

.. jupyter-execute::

 # Get the provider and the backend
 IBMQ.providers()
 provider=IBMQ.providers()[#select the provider you want with the index]
 backend=provider.get_backend('ibmq_lima')
 
..jupyter-execute::

 #get the basic feautures with configuration()
 config = backend.configuration()
 backend_job_limit=backend.job_limit()

 print("{0} is on version {1}. It has {2} qubit{3}. It "
      "{4} OpenPulse programs. The basis gates supported on this device are {5}.{6} are the list of connected qubits."
      "{7} can handle maximum {8} circuits(experiments) per job."
      "Maximum number of active job i can have on it (cocurrent jobs that a user is able to submit to a backend) is {9}."
      "".format(config.backend_name,
                config.backend_version,
                config.n_qubits,
                '' if config.n_qubits == 1 else 's',
                'supports' if config.open_pulse else 'does not support',
                config.basis_gates,
                config.coupling_map,
               config.backend_name,
               config.max_experiments,
               backend_job_limit.maximum_jobs))

..jupyter-execute::
    
    # Before sending a job to the backend, check how many pending jobs are there
    # to estimate how long the experiment will take.
    status=backend.status()
    jobs_in_queue=status.pending_jobs
    print(jobs_in_queue)

 

1. T1 Characterization
-----------------------------


Import required packages and compose a ParallelExperiment for qubits' T1 values.

..jupyter-execute::

 from qiskit_experiments.framework import ParallelExperiment
 from qiskit_experiments.library import T1

 backend = backend
 delays = list(range(1, 150, 5))

    exps=[]
    for i in range(config.n_qubits):
        exp = T1(qubit=i,
                delays=delays,
                unit="us")
    exps.append(exp) 

 parallel_exp = ParallelExperiment(exps)
 parallel_data = parallel_exp.run(backend, shots=8192).block_for_results()

Now we will view the result data.

..jupyter-execute::

    for result in parallel_data.analysis_results():
    print(result)
    print("\nextra:")
    print(result.extra)

Finally, let's get every sub-experiment data and figures.

..jupyter-execute::

 for i in range(parallel_exp.num_experiments):
    print(f"Component experiment {i}")
    sub_data = parallel_data.component_experiment_data(i)
    display(sub_data.figure(0))
    for result in sub_data.analysis_results():
     print(result)


2. T2* and Ramsey Characterization
----------------------------------------

We will continue to use the ``imbq_lima`` backend for our T2 characterization.
In this Experiment, we will get T2* and Ramsey frequency as a result data.
Start by importing required module, and defining sub experiments.

..jupyter-execute::

    T2_exps=[]
    delays = list(range(1, 150, 5))

    for i in range(config.n_qubits):
        exp = T2Ramsey(qubit=i,
                delays=delays,
                unit="us",
                  osc_freq=1e4)
    exp.set_analysis_options(plot=True)
    T2_exps.append(exp)
   
    print(T2_exps)

    # print corresponding circuits to see how it consists of.
    print(exp.circuits()[3])

..jupyter-execute::

    # choose the shot number according to your required accuracy.
    parallel_exp = ParallelExperiment(T2_exps)
    parallel_data = parallel_exp.run(backend, shots=8192).block_for_results()

Now let's see the result data and each sub-experiment data

..jupyter-execute::

    for result in parallel_data.analysis_results():
    print(result)
    print("\nextra:")
    print(result.extra)

    # print sub-experiment data
    for i in range(parallel_exp.num_experiments):
    print(f"Component experiment {i}")
    sub_data = parallel_data.component_experiment_data(i)
    display(sub_data.figure(0))
    for result in sub_data.analysis_results():
        print(result)

3. Finding qubits with Qubit Spectroscopy
---------------------------------------------------

We will sweep the frequency around the known qubit frequency to see the resonance 
at the qubit frequency reported by the backend. 

..jupyter-execute::

    backend = backend

    exps=[]
    for i in range(config.n_qubits):
    
        freq_estimate = backend.defaults().qubit_freq_est[i]
        frequencies = np.linspace(freq_estimate -15e6, freq_estimate + 15e6, 51)
        exp = QubitSpectroscopy(i, frequencies)
            
        exps.append(exp)

    print(exps)

Check how the spectroscopy experiment is constructed by drawing circuits.

..jupyter-execute::

    circuit_Q0 = exp.circuits(backend)[0]
    circuit_Q0.draw(output="mpl")

Now, lets construct a parallelexperiment to get the frequencies of multiple qubits.

..jupyter-execute::

    parallel_exp = ParallelExperiment(exps)
    parallel_data = parallel_exp.run(backend, shots=8192).block_for_results()


..jupyter-execute::

    # View result data
    for result in parallel_data.analysis_results():
    print(result)
    print("\nextra:")
    print(result.extra)

    # Print sub-experiment data
    for i in range(parallel_exp.num_experiments):
    print(f"Component experiment {i}")
    sub_data = parallel_data.component_experiment_data(i)
    display(sub_data.figure(0))
    for result in sub_data.analysis_results():
        print(result)


Now that we have finished characterizing qubit properties, 
we will characterize gate properties
by utilizing Randomized Benchmarking method.
Import some necessay modules first.

..jupyter-execute::
    
 import numpy as np
 from qiskit import QuantumCircuit, transpile, Aer, IBMQ
 from qiskit.tools.jupyter import *
 from qiskit.visualization import *
 import time
 from qiskit.providers.aer import QasmSimulator
 from qiskit_experiments.library import StandardRB
 from qiskit_experiments.framework import ParallelExperiment, BatchExperiment
 from qiskit_experiments.library.randomized_benchmarking import RBUtils
 import qiskit.circuit.library as circuits


4-1 Running 1 qubit RB experiment on multiple qubit
------------------------------------------------------

..jupyter-execute::

    lengths = np.arange(1, 1000, 100)  
    num_samples = 10  
    seed = 1010  
    #qubits = [0:config.n_qubits]
    qubits=range(config.n_qubits)

    # Run an RB experiment on every qubit in the backend.
    exps = [StandardRB([i], lengths, num_samples=num_samples, seed=seed + i) for i in qubits]
    par_exp=ParallelExperiment(exps)

    start=time.time()
    par_expdata=par_exp.run(backend).block_for_results()
    duration=time.time()-start
    #par_expdata.save() : if you want to save
    par_results=par_expdata.analysis_results()

    # View result data
    for result in par_results:
        print(result)
        print("\nextra:")
        print(result.extra)

..jupyter-execute::

    # Check how long the experiment took 
    #since RB takes quite a long time 
    #It is good to know the timescale of the experiment.

    print(duration)

Viewing sub experiment data 
--------------------------------
..jupyter-execute::

    # Print sub-experiment data
    # sub_result is a list consists of 6 analysis results(parameter, alpha, EPC, EPG rz, EPG sx, EPG x) components
    # sub_results is a list consists of 5*6 analysis results components
    # sub_results_list is a 2 dimensional list consists of list components where each list components have 6 analysis data
    sub_results=[]
    sub_results_list=[]
    for i in range(par_exp.num_experiments):
        print(f"Component experiment {i}")
        sub_data = par_expdata.component_experiment_data(i)
        display(sub_data.figure(0))
        sub_result=sub_data.analysis_results()
        sub_results += sub_result
        sub_results_list.append(sub_result)
        for result in sub_result:
            print(result)
    print('=========================================================')
    print(sub_results)
    print('==========================================================')
    print(sub_results_list)

4-2 Running 2qubit RB Experiment on native qubit pairs.
------------------------------------------------------------

The IBM Q experience uses the cross-resonance interaction as the basis for the cx-gate. 
Generally, a qubit with a higher frequency becomes controlling one 
and lower frequency target one since cross resonance is stronger in this way.
Therefore CNOT has a preferred direction depending of the qubit frequency. 
However there are some exceptions to this rule. 
Therefore, we will choose CNOT direction referencing the gate length. 
CNOT with native direction takes slightly shorter time since there is 
one extra single qubit gate in the opposite direction 
to make opposite direction of CNOT also possible for the backend. 
The function `native_cnot` will render you the native direction of CNOT 
which has shorter gate length to perform 2qubit RB experiemnt.
You can check the gate length of CNOTs for both direction in
`IBM Quantum Service <https://quantum-computing.ibm.com/services?services=systems>`_. 
Though gate length of both directions are diffrent the error rates are same. 
This is because single qubit gate errors are typically 1-2 orders of magnitude lower 
than the CNOT gate errors and this fact is not reflected.
In this guide, let's also consider the native direction!

..jupyter-execute::

    coupled_qubit=config.coupling_map
    def native_cnot(coupled_qubit):
        native_cnot=[]
        coupling_map=list(map(tuple, coupled_qubit))
        print(f'coupling_map={coupling_map}')
        print('\n')
        
        for i in range(0, len(coupling_map)-1):
            for j in range(i+1, len(coupling_map)):        
                if coupling_map[i][0]==coupling_map[j][1] and coupling_map[i][1]==coupling_map[j][0]:                
                    i_direction=backend.properties().gate_length('cx',(coupling_map[i][0],coupling_map[i][1]))
                    j_direction=backend.properties().gate_length('cx',(coupling_map[j][0],coupling_map[j][1]))
                    print(f'cx{coupling_map[i]} takes {i_direction}sec')
                    print(f'cx{coupling_map[j]} takes {j_direction}sec')
                    print('----------------------------------------------')
                    if i_direction > j_direction:
                        native_cnot.append(coupling_map[j])
                    else:
                        native_cnot.append(coupling_map[i])       
        return native_cnot

    native_cnot=native_cnot(coupled_qubit)        
                
    print(native_cnot)  

We will construct 2 qubit gate (cx gate) RB experiment in native direction 
with the ordered pairs obatained above.

..jupyter-execute::

    # Make a list of 2qubit gate RB experiments on native CNOT direction
    lengths_2q=np.arange(1,200,30)
    exps_2q =[]
    for i in range(0,len(native_cnot)):
        exps_2q.append(StandardRB(native_cnot[i],lengths_2q, num_samples=num_samples, seed=seed+i))

    print(exps_2q)

Before running the 2qubit RB,
use EPG data of 1 qubit RB experiment to ensure correct 2 qubit EPG computation.

..jupyter-execute::

    # Make a 2dimensional list 'epg_1q' which constists of lists
    # each list consists of 2*6 analysis data of paired qubit tuple
    N=native_cnot
    epg_1q=[]

    for i in range(len(N)):
        epg_1q_pair=par_expdata.component_experiment_data(N[i][0]).analysis_results()+par_expdata.component_experiment_data(N[i][1]).analysis_results()
        epg_1q.append(epg_1q_pair)
    
    print(epg_1q)

..jupyter-execute::

    # give 1qubit EPG data to 2qubit RB experiment as analysis option
    i=0
    for RBi in exps_2q:
        RBi.set_analysis_options(epg_1_qubit=epg_1q[i])
        i += 1
    # Run 2qubit RB experiments on coupled qubit in native directions
    # RB2qResults is a 2dimesional list consists of
    # RB2qResult which is a list consists of 4 analysis data(Parameter analysis, alpha, EPC, EPG_cx)
    RB2qResults=[]
    for RBexp in exps_2q:
        RBexpdata=RBexp.run(backend).block_for_results() 
        RB2qResult=RBexpdata.analysis_results()
        RB2qResults.append(RB2qResult)
    print(RB2qResults)

..jupyter-execute::

    # Compare the computed EPG of the cx gate with the backend's recorded cx gate error:
    for i in range(len(native_cnot)):
        expected_epg = RBUtils.get_error_dict_from_backend(backend, native_cnot[i])[(native_cnot[i], 'cx')]
        exp_2q_epg = RB2qResults[i][3]
    
        print("Backend's reported EPG of the cx gate:", expected_epg)
        print("Experiment computed EPG of the cx gate:", exp_2q_epg)
        print('------------------------------------------------------')