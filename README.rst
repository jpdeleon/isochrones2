isochrones
==========

|CI| |docs|

.. |CI| image:: https://github.com/jpdeleon/isochrones2/actions/workflows/ci.yml/badge.svg
    :target: https://github.com/jpdeleon/isochrones2/actions/workflows/ci.yml
    :alt: CI status

.. |docs| image:: https://github.com/jpdeleon/isochrones2/actions/workflows/docs.yml/badge.svg
    :target: https://jpdeleon.github.io/isochrones2/
    :alt: Documentation

A simple interface to grids of stellar evolution models (currently MIST), for

* interpolating stellar model values at any location in the grid,
* generating properties of synthetic stellar populations, and
* inferring the properties of single stars, multiple-star systems and
  clusters from arbitrary observables, with a Bayesian forward model.

**Documentation:** https://jpdeleon.github.io/isochrones2/

Installation
------------

Python 3.12 or newer is required::

    git clone https://github.com/jpdeleon/isochrones2.git
    cd isochrones2
    pip install -e .            # add ".[dev]" for pytest, coverage and hypothesis

The MIST model grids are large and are downloaded automatically the first
time you ask for them (for example with ``get_ichrone('mist')``).  They are
stored in ``$ISOCHRONES``, which defaults to ``~/.isochrones``.

Quick start
-----------

From the command line
~~~~~~~~~~~~~~~~~~~~~

Make a file called ``star.ini`` that looks something like this::

    Teff = 5770, 100
    feh = 0.0, 0.15
    logg = 4.5, 0.1
    V = 10.0, 0.05

Any combination of spectroscopic or supported photometric properties, as well
as parallax, can go into this file.  Then run::

    starfit mystar              # a folder containing star.ini; "." by default
    starfit --binary mystar     # also --triple, or --all for single+binary+triple

``starfit`` writes an HDF5 file containing the saved ``StarModel`` (reload it
with ``StarModel.load_hdf``) and corner plots of the fit.  It uses MultiNest
when ``pymultinest`` is installed and falls back to ``emcee`` otherwise
(force it with ``--emcee``).

From Python
~~~~~~~~~~~

.. code-block:: python

    from isochrones import get_ichrone, SingleStarModel

    mist = get_ichrone("mist", bands=["BP", "RP"])
    params = {
        "Teff": (5700, 100),
        "logg": (4.5, 0.1),
        "feh": (0.0, 0.15),
        "BP": (10.42, 0.01),
        "RP": (9.54, 0.01),
        "parallax": (10, 0.5),  # mas
    }

    mod = SingleStarModel(mist, **params)
    mod.fit()

    mod.derived_samples.describe()   # posterior samples of mass, radius, age, ...
    mod.corner_physical()

Posterior summaries
~~~~~~~~~~~~~~~~~~~

Besides quantiles, summaries report the posterior **mode** and the central
(CCI) and minimum (MCI) credible intervals, following Valls-Gabaud (2016):

.. code-block:: python

    import numpy as np
    from isochrones.summary import summary_statistics

    summary_statistics(np.random.normal(9.2, 0.15, 5000), alpha=0.68)
    # {'05': ..., '16': ..., '50': ..., '84': ..., '95': ...,
    #  'mode': ..., 'cci_lo_68': ..., 'cci_hi_68': ..., 'mci_lo_68': ..., 'mci_hi_68': ...}

More runnable examples (priors, goodness of fit with Saha's *W*, catalogs,
multiple-star systems, clusters, batch summaries) are in the
`examples page <https://jpdeleon.github.io/isochrones2/examples.html>`_.

Command-line tools
------------------

* ``starfit``: fit one or more folders containing ``star.ini``.
* ``batch-starfit``: launch ``starfit`` over a list of folders (slurm job array).
* ``starfit-summarize``: collect quantiles, modes and credible intervals of
  many fits into one table (``summary.h5`` by default).
* ``starmodel-select``: model selection between different MultiNest runs.

Testing
-------

::

    pip install -e ".[dev]"
    pytest                                  # needs the MIST grids (large download)

    # the grid-free subset that runs on every pull request:
    pytest --ignore=isochrones/tests/test_basic.py \
           --ignore=isochrones/tests/test_ini.py \
           --ignore=isochrones/tests/test_likelihood.py \
           --ignore=isochrones/tests/test_populations.py

The CI workflow runs the grid-free subset on every push and pull request; the
full suite can be started by hand with *Run workflow* in the Actions tab.

Known limitations
-----------------

Parts of the Valls-Gabaud (2016) *BayesGM* formalism are not implemented, and a
few are pinned as expected failures in the test suite.  See the
`limitations page <https://jpdeleon.github.io/isochrones2/limitations.html>`_
and issue `#1 <https://github.com/jpdeleon/isochrones2/issues/1>`_.

Attribution
-----------

If you use ``isochrones`` in your research, please cite `this ASCL reference
<http://adsabs.harvard.edu/cgi-bin/nph-bib_query?bibcode=2015ascl.soft03010M&data_type=BIBTEX&db_key=AST&nocookieset=1>`_.
The package was created by Timothy D. Morton.
