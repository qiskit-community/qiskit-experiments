---
name: "\U0001F41B New experiment proposal"
about: "Proposal and workflow guideline for adding a new experiment \U0001F914."
title: ''
labels: enhancement
assignees: ''

---

<!-- ⚠️ If you do not respect this template, your issue will be closed -->
<!-- ⚠️ Make sure to browse the opened and closed issues -->

# New experiment proposal
Provide the experiment and implementation details below and ask the qiskit-experiments core team to review 
your proposal. 

## General details

### Experiment name
What is the experiment class name ? This name will also be used in the API documentation and tutorial.

### Experiment type
What type the experiment is ? Characterization, calibration, verification, validation, else? 

### Experiment protocol
Provide a concise description of the experiment. Make sure you cover the following aspects:  
* What is the main goal of the experiment ?
* Describe the circuits/pulses and their main parameters needed for the experiment
* What are the main outputs of the experiment ?

### Experiment analysis
Provide a concise description of the experiment's analysis. Make sure you cover the following aspects:
* What is the main fit model for the experiment ? 
* What are the main fit parameters ?
* How would you evaluate whether the fit is good or bad ?

### References 
If applicable, provide a few of the most relevant references here.

## Implementation details
Provide additional details pertaining to the implementation of the experiment and its analysis. Use the following as 
guiding points. Provide code snippets and usage examples where appropriate. 
### Experiment implementation
* The classes the experiment should subclasses
* Input parameters, required and optional, and types of each
* Experiment options and default values
* Are there any limitations or special requirements for running the experiment run as part of a composite experiments? 
Provide details.

### Experiment analysis
* Analysis options and default values
* How does it generate initial guesses
* What plots will be generated ? 

---

# Workflow
These are the steps required for adding a new experiment to qiskit-experiments. It is advisable to follow this workflow
in order and get the proper review approvals from the qiskit-experiments core team before moving on to subsequent steps 
in the workflow. 

NOTE: An experiment PR failing to complete all the workflow steps below will not be merged !

- [ ] Complete the proposal section above and have your proposal reviewed and approved 
- [ ] Open a PR and link it to the new experiment issue
- [ ] Create the main module files (at least one for the experiment and one for its analysis). Place them in the right 
folders 
- [ ] Define and implement the experiments APIs 
- [ ] Have the experiment API reviewed and approved
- [ ] Implement all the experiment and analysis functionality
- [ ] Add unit testing for the experiment and analysis classes. If needed implement a mock-backend for your experiment
- [ ] Ask for a detailed implementation review
- [ ] Verify that your experiment runs properly on a real device and that the results make sense
- [ ] Verify that your experiment runs properly in the context of a parallel experiment, where sub-experiments run on 
different qubits
- [ ] Verify aggregated cases, where `run` is executed with the `experiment_data` parameter set to an existing experiment 
data
- [ ] Verify that figures look OK: for regular experiments, parallel experiments, aggregated experiments
- [ ] Write API docs for all your API methods. Follow the guideline [here](https://github.com/Qiskit/qiskit-experiments/blob/main/CONTRIBUTING.md)
- [ ] Write a tutorial for your experiment. Follow the guideline [here](https://github.com/Qiskit/qiskit-experiments/blob/main/docs/tutorials/GUIDELINES.md) 
- [ ] Ask for a final review of the documentation and implementation
- [ ] Celebrate !

  