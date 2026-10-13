Known limitations
=================

This package implements part of the *BayesGM* formalism of Valls-Gabaud
(2016), *Bayesian Isochrone Fitting and Stellar Ages*
(`arXiv:1607.03000 <https://arxiv.org/abs/1607.03000>`_).  The items below
were found by comparing the code with the paper.  The tracking issue is
`#1 <https://github.com/jpdeleon/isochrones2/issues/1>`_.

Each item that can be tested without a design decision is pinned in the test
suite as a *strict* expected failure, so it flips to a hard failure when the
behaviour is fixed and the marker has to be removed.

Open
----

IMF not normalised over the isochrone's mass range
    The paper normalises the IMF by :math:`K=\int_{m_{lim}}^{m_{top}}\rho(m)\,dm`
    for the isochrone being evaluated (Eq. 18-19).  The code normalises over
    grid-wide mass bounds, so an isochrone's weight does not track how much
    of the mass range exists at its age.  Affects ages and evidences.
    *Pinned:* ``test_cluster_imf_normalisation.py``.

Secondary-EEP Jacobian in the binary grid
    ``calc_lnlike_grid`` integrates over the secondary's EEP with a density
    in the mass ratio :math:`q=m_k/m_j` but without
    :math:`(dm_k/d\mathrm{EEP}_k)/m_j`.  The single-star branch ``(1-fB)``
    is therefore not weighted by :math:`\int p(q)\,dq=1`, which can distort
    the binary fraction and the mass posterior.
    *Pinned:* ``test_cluster_mass_ratio_jacobian.py``.

Minimum credible interval columns are a single interval
    The paper's MCI can be disconnected.  ``summary.mci_intervals`` returns
    every connected component, but ``summary.hpd`` and the ``mci_lo_*`` /
    ``mci_hi_*`` summary columns remain the shortest single interval, which
    for a bimodal posterior spans the gap.
    *Tested:* ``test_summary_mci_disconnected.py``.

Goodness of fit is only partly provided
    ``isochrones.wstat`` implements Saha's :math:`W` (both forms).  Binning
    model and observed CMDs, and comparing :math:`W` for simulated and real
    data, are left to the caller.

No correlated uncertainties
    All Gaussian likelihood terms are independent per band or property; the
    paper's correlation coefficient between observables (Eq. 14) is not
    available.

Binary branch ignores the binary for spectroscopic properties
    In the cluster likelihood, properties such as ``Teff`` are compared with
    the primary's model value in both branches; the paper combines the
    luminosity and effective temperature of the unresolved system.

Coevality is imposed
    Components of a multiple system, and members of a cluster, share age,
    metallicity, distance and extinction.  Independent per-component ages
    (paper Sec. 4 and Figs. 6-8), and coevality as a testable hypothesis, are
    not available.  See :doc:`deviations`.

Evolution-track grids
    ``EEP_prior`` carries the IMF weight for track grids, but
    ``StarModel.lnprior`` raises ``NotImplementedError`` when the EEP axis
    replaces age, so end-to-end track-grid fits are not supported.

Not implemented
---------------

* Composite resolved populations as linear combinations of partial CMDs
  (paper Sec. 6, Eq. 25).
* Pixel CMDs / unresolved populations (paper Sec. 7).

Caveats worth knowing
---------------------

* Likelihood ratios are only meaningful for nested models with strictly
  positive parameters (paper Sec. 8); keep this in mind when comparing
  ``lnZ`` between runs.
* ``EEP_prior`` is intentionally unnormalised.  Its pdf is conditional on
  (age, feh) or (mass, feh), so one constant cannot normalise it; compare
  absolute ``lnprior`` values across models with care.
* ``calc_lnlike_grid`` is compiled with ``fastmath=True`` and relies on
  ``-inf`` and ``log(0)`` at the edges (``fB`` of 0 or 1, ``q`` below
  ``minq``).  The edge values are covered by
  ``test_cluster_grid_edge_values.py``.
