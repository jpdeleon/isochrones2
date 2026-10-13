Examples
========

Short, copy-and-paste recipes.  Each section is marked with what it needs:

* **Runs anywhere**: needs only the installed package.  The printed output
  shown was produced by running the code.
* **Needs the MIST grids**: downloads the model grids on first use (see
  :doc:`install`).  Code is shown without output; the notebooks in this
  documentation execute the same calls.

.. contents::
   :local:
   :depth: 1


Summarise a posterior: mode, CCI and MCI
----------------------------------------

*Runs anywhere.*  Point estimates and intervals follow Valls-Gabaud (2016):
the **mode** is the robust point estimate, the **central credible interval**
(CCI) has equal tails and always contains the median, and the **minimum
credible interval** (MCI) is the shortest interval holding the requested
posterior fraction and always contains the mode.

.. code-block:: python

    import numpy as np
    from isochrones.summary import summary_statistics

    rng = np.random.default_rng(42)
    # a skewed age posterior: a narrow peak plus a broad old tail
    samples = np.concatenate([rng.normal(9.2, 0.15, 3000),
                              rng.normal(9.8, 0.30, 1000)])

    stats = summary_statistics(samples, alpha=0.68)
    for key, value in stats.items():
        print(f"{key:10s} {value:7.3f}")

.. code-block:: text

    05           8.963
    16           9.076
    50           9.257
    84           9.696
    95          10.034
    mode         9.211
    cci_lo_68    9.076
    cci_hi_68    9.696
    mci_lo_68    8.987
    mci_hi_68    9.445

The MCI (8.99-9.45) is much tighter than the CCI (9.08-9.70) and sits on the
peak, because the PDF is skewed.  Non-finite samples are ignored.

.. note::

   The MCI returned here is a *single* interval.  For a multimodal posterior
   the paper's MCI can be disconnected; see :doc:`limitations`.

The same statistics for every derived parameter of a saved fit:

.. code-block:: python

    from isochrones.summary import get_quantiles, get_summary_df

    # one row for star "mystar", loaded from mystar/mist_starmodel_single.h5
    df = get_quantiles("mystar", rootdir=".", columns=["mass", "age"], alpha=0.95)
    df.filter(like="_mode")            # <parameter>_mode, _cci_lo_95, _mci_hi_95, ...

    # many stars at once (optionally pass pool=multiprocessing.Pool(8))
    table = get_summary_df(["star1", "star2", "star3"], rootdir="results")

Missing or unreadable model files give an empty frame (or raise with
``raise_exceptions=True``).


Is the fit a good fit?  Saha's W statistic
------------------------------------------

*Runs anywhere.*  Compare binned model counts :math:`m_i` with observed
counts :math:`s_i` (for example Hess-diagram bins).  Larger :math:`W` means
the model counts are more consistent with the observations.  ``log_w`` is the
fixed-total form; ``log_w_poisson`` treats the totals as Poisson expectations
and accepts real-valued model counts.

.. code-block:: python

    import numpy as np
    from isochrones.wstat import log_w, log_w_poisson

    observed = np.array([12, 30, 25, 8, 2])
    good = np.array([11.0, 28.5, 27.0, 9.0, 2.5])   # close to the data
    bad = np.array([2.0, 10.0, 40.0, 20.0, 5.0])    # wrong shape

    for name, model in [("good", good), ("bad", bad)]:
        print(name, round(log_w(model, observed), 3),
              round(log_w_poisson(model, observed), 3))

.. code-block:: text

    good 98.345 -29.591
    bad 84.081 -43.135

Both statistics rank the good model above the bad one.  Inputs must be
non-empty 1-D arrays of the same length (a ``ValueError`` is raised
otherwise).

To bin a model and an observed CMD on the same Hess grid use ``bin_cmd``.
``w_goodness_of_fit`` then draws data sets from the model (each bin Poisson
with mean :math:`m_i`), computes :math:`W` for each, and reports where the
observed :math:`W` falls:

.. code-block:: python

    from isochrones.wstat import w_goodness_of_fit

    for name, model in [("good", good), ("bad", bad)]:
        r = w_goodness_of_fit(model, observed, n_sim=1000, seed=0)
        print(name, round(r.log_w_observed, 3), round(r.p_value, 3))

.. code-block:: text

    good -29.591 0.914
    bad -43.135 0.0

``p_value`` is the fraction of simulated data sets with a :math:`W` no larger
than the observed one, so a small value means the observed CMD matches the
model worse than model-generated data do.


Look at and change the priors
-----------------------------

*Runs anywhere* for the prior objects themselves.

.. code-block:: python

    import numpy as np
    from isochrones.priors import AgePrior, ChabrierPrior, FehPrior, GaussianPrior

    AgePrior().bounds                         # (5, 10.15): log10(age/yr)
    float(np.median(ChabrierPrior().sample(5000)))     # ~0.26 Msun
    float(np.median(FehPrior(halo_fraction=0.5).sample(5000)))  # ~-0.44 dex

    # a Gaussian age prior with bounds, evaluated at its centre
    GaussianPrior(9.6, 1, bounds=(8, 10))(9.6)         # 0.664

*Needs the MIST grids* to attach them to a model:

.. code-block:: python

    from isochrones import get_ichrone, SingleStarModel
    from isochrones.priors import FlatPrior, DistancePrior

    mist = get_ichrone("mist")
    mod = SingleStarModel(mist, Teff=(5700, 80), logg=(4.5, 0.1), feh=(0.0, 0.1),
                          parallax=(10, 0.1))

    mod._priors                                # the defaults, per parameter
    mod.set_prior(feh=FlatPrior((-2, 0.5)), distance=DistancePrior(1000))
    mod.set_bounds(eep=(1, 600), age=(8, 10))  # restrict the sampled ranges
    samples = mod.sample_from_prior(1000)      # inspect before fitting

.. warning::

   The default nuisance priors are astrophysical, not flat; see
   :doc:`deviations`.


Fit a star
----------

*Needs the MIST grids.*

.. code-block:: python

    from isochrones import get_ichrone, SingleStarModel

    mist = get_ichrone("mist", bands=["BP", "RP"])
    mod = SingleStarModel(mist,
                          Teff=(5700, 100), logg=(4.5, 0.1), feh=(0.0, 0.15),
                          BP=(10.42, 0.01), RP=(9.54, 0.01),
                          parallax=(10, 0.5),            # mas
                          name="demo")
    mod.fit()                                  # MultiNest; mod.fit_mcmc() uses emcee

    mod.samples.describe()                     # sampled parameters
    mod.derived_samples.describe()             # mass, radius, density, ...
    mod.corner_physical()
    mod.corner_observed()                      # posterior predictive vs. data

    mod.save_hdf("demo.h5")                    # later: SingleStarModel.load_hdf("demo.h5")

Evaluate the posterior at a given point to see how the likelihood responds:

.. code-block:: python

    eep = mist.get_eep(1.0, 9.74, -0.05, accurate=True)
    pars = [eep, 9.74, -0.05, 100, 0.02]       # eep, age, feh, distance, AV
    mod.lnprior(pars), mod.lnlike(pars), mod.lnpost(pars)


Binary and triple systems
-------------------------

*Needs the MIST grids.*  The components share age, metallicity, distance and
extinction (they are coeval by construction, see :doc:`limitations`).

.. code-block:: python

    from isochrones import BinaryStarModel, TripleStarModel

    mod2 = BinaryStarModel(mist, **params)     # eep_0, eep_1, age, feh, distance, AV
    mod2.set_bounds(eep=(1, 600), age=(8, 10))
    mod2.fit(n_live_points=2000)               # binaries usually need more live points
    mod2.corner_derived(["mass_0", "mass_1", "age", "feh", "distance", "AV"])

Spatially resolved components are described with an observation tree
(see the *Multiple star systems* notebook).


Generate synthetic stars
------------------------

*Needs the MIST grids.*

.. code-block:: python

    import numpy as np
    from isochrones import get_ichrone
    from isochrones.priors import ChabrierPrior

    tracks = get_ichrone("mist", tracks=True)

    # one star (mass, log10 age, [Fe/H]) -> dict of properties
    tracks.generate(1.03, 9.72, -0.11, return_dict=True)

    # a 1000-star cluster at 8 kpc; dropna() removes stars off the grid
    masses = ChabrierPrior().sample(1000)
    df = tracks.generate(masses, np.log10(6e9), -1.8, distance=8000., AV=0.15).dropna()
    df["BP-RP"] = df.BP - df.RP


Interpolate the model grids
---------------------------

*Needs the MIST grids.*

.. code-block:: python

    from isochrones import get_ichrone

    mist = get_ichrone("mist")
    eep = mist.get_eep(1.01, 9.76, 0.03, accurate=True)   # mass, log age, feh
    mist.interp_value([eep, 9.76, 0.03], ["Teff", "logg", "radius", "density"])
    mist.interp_mag([eep, 9.76, 0.03, 200, 0.1], bands=["G", "BP", "RP"])  # + distance, AV


Magnitude arithmetic
--------------------

*Runs anywhere.*

.. code-block:: python

    from isochrones.utils import addmags

    addmags(10.0, 10.0)                  # 9.247  (two equal stars: -2.5 log10 2)
    addmags((10.0, 0.02), (11.0, 0.03))  # (9.636, 0.0166): magnitude and error

Use ``addmags`` to build the blended magnitude of an unresolved pair.


Hold a catalog of stars
-----------------------

*Runs anywhere* to build and inspect it.

.. code-block:: python

    import pandas as pd
    from isochrones.catalog import StarCatalog

    df = pd.DataFrame(
        {"G_mag": [10.1, 11.2], "G_mag_unc": [0.01, 0.01],
         "BP_mag": [10.6, 11.9], "BP_mag_unc": [0.01, 0.02],
         "Teff": [5700.0, 5100.0], "Teff_unc": [80.0, 90.0]},
        index=["star_a", "star_b"],
    )
    cat = StarCatalog(df, props=["Teff"])
    cat.bands, cat.props, len(cat)       # (('G', 'BP'), ('Teff',), 2)

Every ``<band>_mag`` column becomes a band (override with ``bands=``), and
each measurement needs a matching ``_unc`` column (a ``ValueError`` names the
missing one).

*Needs the MIST grids* to turn rows into fit-ready models or ``star.ini``
folders:

.. code-block:: python

    cat.set_prior(feh=FlatPrior((-2, 0.5)))
    for mod in cat.iter_models(N=1):     # SingleStarModel per row; N=2 binary
        mod.fit()

    cat.write_ini(root="runs", N=1)      # one folder per star; then: starfit runs/<name>


Fit a cluster
-------------

*Needs the MIST grids.*  ``StarClusterModel`` fits age, metallicity, distance,
extinction, the IMF slope, the mass-ratio slope and the binary fraction to a
catalog of members, integrating each star's likelihood along the isochrone.

.. code-block:: python

    from isochrones import get_ichrone
    from isochrones.cluster import StarClusterModel

    mist = get_ichrone("mist", bands=["G", "BP", "RP"])
    cluster = StarClusterModel(mist, cat, name="my_cluster",
                               max_AV=1.0, max_distance=5000, minq=0.1)
    cluster.param_names     # age, feh, distance, AV, alpha, gamma, fB

.. warning::

   Read :doc:`limitations` before trusting cluster ages: the IMF
   normalisation and the binary mass-ratio integral are known open issues.


The ``star.ini`` file format
----------------------------

Used by ``starfit`` (see :doc:`cli`).

Properties and tagged photometry in a single file:

.. code-block:: ini

    RA = 299.268036
    dec = 45.227428
    Teff = 4135, 98.0
    feh = -0.46, 0.16
    logg = 4.711, 0.1

    [twomass]
    J = 13.513, 0.02
    H = 12.845, 0.02
    K = 12.693, 0.02

    [WISE]
    W1 = 12, 0.05
    W2 = 11.5, 0.05

A resolved binary, with the companion given as an offset from the primary
(``relative = True`` means the magnitudes are differential measurements):

.. code-block:: ini

    [twomass]
    resolution = 4.
    relative = True
    J = 13.513, 0.02
    separation_1 = 10
    PA_1 = 100.
    J_1 = 12.513, 0.02
