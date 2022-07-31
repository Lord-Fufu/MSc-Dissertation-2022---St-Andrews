# MSc-Dissertation-2022---St-Andrews

This repository includes the simulation code I wrote for my Master Thesis at the University of St Andrews (2022).
I worked on stochastic modelling of delayed chemical systems with fast environmental switching.
More specifically I simulated the production system of a protein: Hes1.

Python Files:
- method from the master equation using the Gillespie algorithm (reference)
- approximation methods using SDE Langevin equations (method to be compared to reference)
- utility and post processing

Notebook:
- perform computation, data saves and plots.

The 'SANDBOX' section in the notebook allows the users to plot trajectories, means, spectra etc. for different sets of parameters.
Hence, they can investigate the influence of the parameters themselves.

The 'MAKING PLOTS' section in the notebook produces data files and associated plots. They are located in a new directory, named 'output' followed by the date and hour on which it was created. The users can change the data to be read by changing the value of the variable 'read_directory' in the notebook (just after the subtitle 'Plots').
