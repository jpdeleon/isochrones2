isochrones
==========

**Isochrones** is a python package that provides a simple interface to grids
of stellar evolution models, enabling the following common use cases:

* Interpolating stellar model values at desired locations.
* Generating properties of synthetic stellar populations.
* Determining stellar properties of single- or multiple-star systems and of
  clusters, based on arbitrary observables.

The central goal of **isochrones** is to standardize model-grid-based stellar
parameter inference, and to enable such inference under different sets of
stellar models.  For now, only MIST models are included, but we hope to
incorporate YAPSI and PARSEC models as well.

Posteriors are summarised with the mode and central/minimum credible
intervals, and Saha's *W* statistic is available for goodness of fit,
following Valls-Gabaud (2016).  See :doc:`limitations` for what is not
implemented yet.

.. note::

   The source lives at https://github.com/jpdeleon/isochrones2; an
   `overview and quick start <https://github.com/jpdeleon/isochrones2#readme>`_
   is in the README.

.. toctree::
   :maxdepth: 2
   :caption: Getting started

   install.ipynb
   quickstart.ipynb
   examples
   cli

.. toctree::
   :maxdepth: 2
   :caption: Guides

   interpolate.ipynb
   modelgrids.ipynb
   bc.ipynb
   grid_interpolator.ipynb
   starmodel.ipynb
   multiple.ipynb
   populations.ipynb

.. toctree::
   :maxdepth: 2
   :caption: Reference

   api
   deviations
   limitations
