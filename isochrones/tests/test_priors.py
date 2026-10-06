import numpy as np


def test_age():
    from isochrones.priors import AgePrior

    age_prior = AgePrior()
    age_prior.test_integral()
    age_prior.test_sampling()


def test_distance():
    from isochrones.priors import DistancePrior

    distance_prior = DistancePrior()
    distance_prior.test_integral()
    distance_prior.test_sampling()


def test_AV():
    from isochrones.priors import AVPrior

    AV_prior = AVPrior()
    AV_prior.test_integral()
    AV_prior.test_sampling()


def test_q():
    from isochrones.priors import QPrior

    q_prior = QPrior()
    q_prior.test_integral()
    q_prior.test_sampling()


def test_salpeter():
    from isochrones.priors import SalpeterPrior

    salpeter_prior = SalpeterPrior()
    salpeter_prior.test_integral()
    salpeter_prior.test_sampling()


def test_feh():
    from isochrones.priors import FehPrior

    feh_prior = FehPrior()
    feh_prior.test_integral()
    feh_prior.test_sampling()
    feh_prior.bounds = (-3, 0.25)
    feh_prior.test_integral()
    feh_prior.test_sampling()
    assert feh_prior(-3.5) == 0
    assert feh_prior(0.4) == 0


def test_chabrier():
    from isochrones.priors import ChabrierPrior

    chabrier_prior = ChabrierPrior()
    chabrier_prior.test_integral()
    chabrier_prior.test_sampling()


class FakeTrackIC:
    """Minimal evolution-track grid: eep replaces age, mass constant per track."""
    eep_replaces = "age"
    eep_bounds = (0, 100)

    def interp_value(self, pars, cols):
        mass, eep, feh = pars
        return np.array([[eep / 10.0, 5.0]])


class FakeIsoIC:
    """Minimal isochrone grid: eep replaces mass, age constant per isochrone."""
    eep_replaces = "mass"
    eep_bounds = (0, 100)

    def interp_value(self, pars, cols):
        eep, age, feh = pars
        return np.array([[0.8 + 0.01 * eep, 3.0]])


def test_eep_prior_track_grid_includes_imf():
    """For eep -> age grids the IMF must enter as a mass_prior factor.

    P4 from #1: previously the track-grid branch was only
    age_prior(age) * dt_deep, which is flat in log10(age) with no mass
    weighting at all.  With mass_prior supplied the EEP pdf is
    age_prior(age) * dt_deep * IMF(mass).
    """
    from isochrones.priors import AgePrior, ChabrierPrior, EEP_prior

    ic = FakeTrackIC()
    age_prior = AgePrior()
    imf = ChabrierPrior()
    eep, mass, feh = 70.0, 1.0, 0.0

    p = EEP_prior(ic, age_prior, mass_prior=imf)
    expected = age_prior(eep / 10.0) * 5.0 * imf(mass)
    assert np.isclose(p.pdf(eep, mass=mass, feh=feh), expected, rtol=1e-12)

    # Without mass_prior the old behavior is preserved (no IMF factor).
    plain = EEP_prior(ic, age_prior)
    assert np.isclose(plain.pdf(eep, mass=mass, feh=feh), age_prior(eep / 10.0) * 5.0, rtol=1e-12)
    assert not np.isclose(p.pdf(eep, mass=mass, feh=feh), plain.pdf(eep, mass=mass, feh=feh))


def test_eep_prior_iso_grid_unchanged_by_mass_prior():
    """For eep -> mass grids the IMF already comes from orig_prior.

    Passing a mass_prior too must not double-count it.
    """
    from isochrones.priors import AgePrior, ChabrierPrior, EEP_prior

    ic = FakeIsoIC()
    imf = ChabrierPrior()
    eep, age, feh = 40.0, 9.0, 0.0

    orig = EEP_prior(ic, imf)
    with_mass = EEP_prior(ic, imf, mass_prior=imf)

    expected = imf(0.8 + 0.01 * eep) * 3.0
    assert np.isclose(orig.pdf(eep, age=age, feh=feh), expected, rtol=1e-12)
    assert np.isclose(with_mass.pdf(eep, age=age, feh=feh), orig.pdf(eep, age=age, feh=feh), rtol=1e-12)
