Deviations from the paper
=========================

This package implements the Bayesian grid-based stellar parameter
inference of Valls-Gabaud (2016), *Bayesian Isochrone Fitting and
Stellar Ages* (arXiv:1607.03000, hereafter V16).  The following are
deliberate deviations from the paper's prescribed methodology, kept for
pragmatic or design reasons.  They are documented here because they
change the meaning of the outputs relative to the paper's test cases.

Coevality is imposed by construction
------------------------------------

``StarClusterModel`` shares a single ``age`` parameter across all members
of a cluster.  V16 instead *avoids* assuming coevality a priori: each
star is fitted with its own age PDF, and membership is assessed by
checking whether the individual age PDFs agree (V16, Sec. 4; tex:1208-1231).
The shared-age construction guarantees coeval fits by construction, but
removes the membership diagnostic the paper obtains for free: an
interloper with a conflicting age PDF is absorbed into the common age
rather than being flagged and rejected.

Nuisance priors are informative, not flat
-----------------------------------------

V16 adopts flat priors on the nuisance parameters (V16, Sec. 3; tex:718-720).
The defaults in this package are instead astrophysical priors:

- ``AgePrior`` -- flat in :math:`\log_{10}(\mathrm{age})`, bounds ``(5, 10.15)``
  (flat in the logarithm, not flat in age);
- ``DistancePrior`` -- volume-density prior :math:`\rho \propto d^2`;
- ``FehPrior`` -- a Bovy/APOGEE disk-mixture prior plus halo component.

These are better informed than flat priors for Galactic work, and they
reflect reasonable defaults, but they are not V16's test case.  In
particular the :math:`d^2` distance prior is a strong assumption in
extragalactic contexts, where a flat (or explicitly chosen) distance
prior may be preferable.

Evolution-track grids are isochrone-first
-----------------------------------------

The EEP prior's change-of-variables correctly carries the IMF weight for
both grid kinds (``isochrones.priors.EEP_prior``; see also the P4 fix in
``link:https://github.com/jpdeleon/isochrones2/pull/6``).  However,
end-to-end fitting on evolution-track grids (``eep_replaces == 'age'``)
is not supported: ``StarModel.lnprior`` raises
``NotImplementedError`` for that branch (``starmodel.py:624``), and
track grids are surfaced through ``IsoTrackModel`` for partial use only.
The package is thus isochrone-grid-first, consistent with V16's focus.