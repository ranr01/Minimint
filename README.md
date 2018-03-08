# Minimint

Minimint is an even more bare-bone version of [Spearmint-lite](https://github.com/JasperSnoek/spearmint/).
Besides being compatible with Python 3, the idea is to provide a
simple Class to be used in Bayesian Hyperparameters Optimization.

I provide the class MinimintOptimizer that does the book keeping of the sampled hyperparameters sets and proposes new points to sample using the Spearmint-lite algorithms.

It does not handle any managing of tasks or results and therefor can be combined with any job management framework.
Minimint_test.ipynb is an Jupyter notebook that present a simple example of using Minimint together with [Ipyparallel](https://github.com/ipython/ipyparallel).
