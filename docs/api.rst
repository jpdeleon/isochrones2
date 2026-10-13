.. _api:

API reference
=============

Most work goes through :func:`~isochrones.get_ichrone` (model-grid
interpolation) and a star model (inference).  The modules below are the
public surface; the heavy numerical kernels are compiled with numba.

Star models
-----------

.. autoclass:: isochrones.starmodel.StarModel
   :members: fit, fit_mcmc, fit_multinest, lnprior, lnlike, lnpost, set_prior,
             set_bounds, sample_from_prior, save_hdf, load_hdf, corner_physical,
             corner_observed

.. autoclass:: isochrones.starmodel.SingleStarModel
.. autoclass:: isochrones.starmodel.BinaryStarModel
.. autoclass:: isochrones.starmodel.TripleStarModel

Clusters and catalogs
---------------------

.. autoclass:: isochrones.catalog.StarCatalog
   :members:

.. autoclass:: isochrones.cluster.StarClusterModel
   :members: lnlike, bounds

Posterior summaries
-------------------

.. automodule:: isochrones.summary
   :members: summary_statistics, mode_of_samples, central_credible_interval,
             hpd, get_quantiles, get_summary_df

Goodness of fit
---------------

.. automodule:: isochrones.wstat
   :members: log_w, log_w_poisson, bin_cmd, w_goodness_of_fit, WGoodnessOfFit

Priors
------

.. automodule:: isochrones.priors
   :members: AgePrior, ChabrierPrior, FehPrior, DistancePrior, AVPrior,
             FlatPrior, FlatLogPrior, GaussianPrior, PowerLawPrior, EEP_prior,
             powerlaw_lnpdf

Utilities
---------

.. automodule:: isochrones.utils
   :members: addmags, band_pairs, distance, download_file
