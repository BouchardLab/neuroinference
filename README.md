# Accurate inference in parametric models reshapes neuroscientific interpretation and improves data-driven discovery

Code for the paper "Accurate inference in parametric models reshapes
neuroscientific interpretation and improves data-driven discovery" by P.S.
Sachdeva, J.A. Livezey, M.E. Dougherty, B. Gu, J.D. Berke, and K.E. Bouchard.

This code was used to perform all model fits and can be used to reproduce the
figures given those fits.

The package is divided into several directories:

* `neuroinference` contains utility code to help perform some of the analysis.
* `figures`: contains Jupyter notebooks that reproduce the figures.
* `notebooks`: contains Jupyter notebooks for additional analyses on the model
  fits.
* `scripts`: contains the scripts that were run to generate the model fits.

The run the model fits, the package `pyuoi` is required. It can be installed via
pip or conda-forge. The Poisson baseline fits require the `glmnet_python`
package. Some of the visualizations require `graph-tool`, which unfortunately
cannot be installed via Anaconda.

