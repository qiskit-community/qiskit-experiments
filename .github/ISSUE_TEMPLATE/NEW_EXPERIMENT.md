---
name: "\U0001F41B Bug report"
about: "Create a report to help us improve \U0001F914."
title: ''
labels: bug
assignees: ''

---

<!-- ⚠️ If you do not respect this template, your issue will be closed -->
<!-- ⚠️ Make sure to browse the opened and closed issues -->

# New experiment proposal
Provide the experiment details and implementation details below and ask the qiskit-experiments core team to review 
your proposal. 

## General details
Provide general detail about the proposed experiment and its analysis by providing the required information in the 
sections below. 

### Experiment name
What is the experiment class name ? This name will also be used in the API documentation and tutorial.

### Experiment type
What type the experiment is ? Characterization, calibration, verification, validation, else? 

### Experiment protocol
Provide a concise description of the experiment. Make sure you cover the following aspects:  
* What is the main goal of the experiment ?
* Describe the circuits/pulses and their main parameters needed for the experiment
* What are the main outputs of the experiments ?

### Experiment analysis
Provide a short description of the required analysis for this experiment. Make sure you cover the following aspects:
* What is the main fit model for the experiment ? 
* What are the main fit parameters ?
* How would you evaluate whether the fit is good or bad ?


### References 
If applicable, provide a few most relevant references here.

## Implementation details
Provide additional details pertaining to the implementation of the experiment and its analysis. Use the following as 
guiding points. Provide code snipptes and usage examples where appropriate. 
### Experiment implementation
* The classes the experiment should subclasses
* Input parameters, required and optional, and types of each
* Experiment options and default values
* Should or could the experiment run as part of a composite experiments. Provide details.

### Experiment analysis
* Analysis options and default values
* How does it generate initial guesses
* What plots will be generated ? 

---

# Workflow
These are the steps required for adding a new experiment to qiskit-experiments. It is advisable to follow this workflow
in order and get the proper review approvals from the qiskit-experiments core team before moving on to subsequent steps 
in the workflos [TBD: repharase, there is partial order, handle it]. 

NOTE: An experiment PR failing to complete all the workflow steps will not be merged !

- [ ] Complete the proposal section above and have your proposal reviewed 
- [ ] Open a PR and link it to the issue
- [ ] Create the main module files (at least one for the experiment and one for its anlaysis), place them in the right 
folder. 
- [ ] Define and implement the experiments APIs. 
- [ ] Ask for an API review
- [ ] Implement all the experiment and analysis functionality
- [ ] Implement unit testing for the experiment and analysis classes. If needed add a mock-backend for your experiment
- [ ] Ask for a detailed implementation review
- [ ] Check that your experiment runs properly on a real device and that the results make sense
- [ ] 
- [ ] Write API docs for all your API methods. Follow the guideline here [TODO]
- [ ] Write a tutorial for your experiment. Follow the guidelien here [TODO]
- [ ] Ask for a documentation ane final reviews 

  