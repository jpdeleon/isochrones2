"""Prior probability distributions for stellar parameter estimation.

This module provides prior probability distribution classes for Bayesian
stellar parameter inference. It includes both simple priors (uniform, normal)
and astrophysically-motivated priors (Salpeter IMF, metallicity distributions).

Classes:
    Prior: Base class for prior distributions.
    FlatPrior: Uniform prior within specified bounds.
    NormalPrior: Gaussian prior.
    LogNormalPrior: Log-normal prior.
    SalpeterIMF: Salpeter initial mass function prior.
    and others...
"""

from __future__ import print_function, division
from __future__ import annotations

from typing import Optional, Tuple, Any, Union

import numpy as np
import pandas as pd
import scipy.stats
from scipy.stats import uniform, lognorm
from scipy.special import ndtr as _norm_cdf
from scipy.integrate import quad

import matplotlib.pyplot as plt
import numba as nb
from math import log, log10


from .logger import getLogger

_norm_pdf_C = np.sqrt(2 * np.pi)
ONE_OVER_ROOT_2PI = 1.0 / _norm_pdf_C
_norm_pdf_logC = np.log(_norm_pdf_C)
LOG_ONE_OVER_ROOT_2PI = np.log(ONE_OVER_ROOT_2PI)


def _norm_pdf(x: float) -> float:
    """Compute unnormalized normal PDF."""
    return np.exp(-(x ** 2) / 2.0) / _norm_pdf_C


def _norm_logpdf(x: float) -> float:
    """Compute log of unnormalized normal PDF."""
    return -(x ** 2) / 2.0 - _norm_pdf_logC


class Prior:
    """Base class for prior probability distributions.

    This class provides a common interface for prior distributions used in
    Bayesian stellar parameter estimation. Subclasses should implement the
    _pdf method to define the unnormalized probability density function.

    Attributes:
        bounds: Tuple of (lower, upper) bounds for the prior support.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize the prior."""
        self._norm: float = 1.0
        if not hasattr(self, "_bounds"):
            self._bounds: Optional[Tuple[float, float]] = None

    def __call__(self, x: float, **kwargs: Any) -> float:
        """Evaluate the prior probability density.

        Args:
            x: Value at which to evaluate the PDF.
            **kwargs: Additional keyword arguments.

        Returns:
            Probability density at x.
        """
        return self.pdf(x, **kwargs)

    @property
    def bounds(self) -> Tuple[float, float]:
        """The bounds of the prior distribution."""
        if getattr(self, "_bounds", None) is None:
            return (-np.inf, np.inf)
        return self._bounds  # type: ignore[return-value]

    @bounds.setter
    def bounds(self, new: Tuple[float, float]) -> None:
        """Set the bounds and normalize the prior.

        Args:
            new: Tuple of (lower, upper) bounds.

        Raises:
            ValueError: If the integral test fails.
        """
        self._norm = quad(self._pdf, *new)[0]
        self._bounds = new
        try:
            self.test_integral()
        except AssertionError:
            raise ValueError(f"Problem setting bounds to {new}; integral test failed.")

    def _pdf(self, x: float, **kwargs: Any) -> float:
        """Compute the unnormalized probability density.

        This method must be implemented by subclasses.

        Args:
            x: Value at which to evaluate the PDF.
            **kwargs: Additional keyword arguments.

        Returns:
            Unnormalized probability density at x.
        """
        raise NotImplementedError

    def pdf(self, x: float, **kwargs: Any) -> float:
        """Compute the normalized probability density.

        Args:
            x: Value at which to evaluate the PDF.
            **kwargs: Additional keyword arguments.

        Returns:
            Normalized probability density at x, or 0 if outside bounds.
        """
        lo, hi = self.bounds
        if x < lo or x > hi:
            return 0
        else:
            return self._pdf(x, **kwargs) / self._norm

    def lnpdf(self, x: float, **kwargs: Any) -> float:
        """Compute the log probability density.

        Args:
            x: Value at which to evaluate the log PDF.
            **kwargs: Additional keyword arguments.

        Returns:
            Log probability density at x, or -inf if probability is zero.
        """
        if hasattr(self, "_lnpdf"):
            return self._lnpdf(x, **kwargs)
        else:
            pdf = self(x, **kwargs)
            return np.log(pdf) if pdf else -np.inf

    def sample(self, n: int, **kwargs: Any) -> Union[float, np.ndarray]:
        """Draw samples from the prior distribution.

        Args:
            n: Number of samples to draw.
            **kwargs: Additional keyword arguments for sampling.

        Returns:
            Array of samples, or single sample if n=1.

        Raises:
            NotImplementedError: If the distribution doesn't have a sampling method.
        """
        if hasattr(self, "distribution"):
            return self.distribution.rvs(n)
        else:
            raise NotImplementedError

    def test_integral(self) -> None:
        """Test that the prior integrates to 1 within its bounds."""
        assert np.isclose(1, quad(self.pdf, *self.bounds)[0])

    def test_sampling(self, n: int = 100000, plot: bool = False) -> None:
        """Test that sampling matches the PDF.

        Args:
            n: Number of samples to draw. Defaults to 100000.
            plot: If True, plot the histogram and PDF. Defaults to False.
        """
        x = self.sample(n)
        if hasattr(self, "bounds"):
            if self.bounds == (-np.inf, np.inf):
                rng = None
            else:
                rng = self.bounds
        else:
            rng = None
        hn, _ = np.histogram(x, range=rng)
        h, b = np.histogram(x, density=True, range=rng)
        logger = getLogger()
        logger.debug("bins: {}".format(b))
        logger.debug("raw: {}".format(hn))
        pdf = np.array([quad(self.pdf, lo, hi)[0] / (hi - lo) for lo, hi in zip(b[:-1], b[1:])])
        sigma = 1.0 / np.sqrt(hn)
        resid = np.absolute(pdf - h) / pdf
        logger.debug("pdf: {}".format(pdf))
        logger.debug("hist: {}".format(h))
        logger.debug("sigma: {}".format(sigma))
        logger.debug("{}".format(resid / sigma))

        c = (b[:-1] + b[1:]) / 2
        if plot:
            plt.plot(c, h, "_")
            plt.plot(c, pdf, "x")
        else:
            assert max((resid / sigma)[hn > 50]) < 6


class BoundedPrior(Prior):
    def __init__(self, bounds=None):
        self._bounds = bounds
        super(BoundedPrior, self).__init__()

    def __call__(self, x, **kwargs):
        if self.bounds is not None:
            lo, hi = self.bounds
            if x < lo or x > hi:
                return 0
        return self.pdf(x, **kwargs)

    @property
    def bounds(self):
        return self._bounds

    @bounds.setter
    def bounds(self, new):
        self._bounds = new
        try:
            self.test_integral()
        except AssertionError:
            raise ValueError(f"Problem setting bounds to {new}; integral test failed.")

    def lnpdf(self, x, **kwargs):
        if self.bounds is not None:
            lo, hi = self.bounds
            if x < lo or x > hi:
                return -np.inf
        if hasattr(self, "_lnpdf"):
            return self._lnpdf(x, **kwargs)
        else:
            pdf = self.pdf(x, **kwargs)
            return np.log(pdf) if pdf else -np.inf


class BrokenPrior(Prior):
    """Composition of multiple stitched-together priors, with breakpoint(s)

    Parameters
    ----------
    components : list of `Prior` objects
        Components going into prior
    breakpoints : array
        List of breakpoints separating components. Length = n(components) - 1
    """

    def __init__(self, components, breakpoints, bounds=None):
        self.components = components
        self.n_components = len(components)
        self.breakpoints = breakpoints

        if bounds is None:
            bounds = (-np.inf, np.inf)
        self._bounds = bounds
        self._norm = 1.0

        self.quad_args = dict(limit=200)
        self._initialize()

    @property
    def bounds(self):
        return super().bounds

    @bounds.setter
    def bounds(self, new):
        self._bounds = new
        self._initialize()

    def _initialize(self):
        # lo = min([c.bounds[0] for c in self.components] + [self.bounds[0]])
        # hi = max([c.bounds[1] for c in self.components] + [self.bounds[1]])
        lo, hi = self.bounds
        full_domain = [lo] + list(self.breakpoints) + [hi]
        domains = [(a, b) for a, b in zip(full_domain[:-1], full_domain[1:])]
        self.domains = domains

        norms = np.ones(self.n_components)
        for i in range(1, self.n_components):
            x = self.breakpoints[i - 1]
            norms[i] = self.components[i](x) / self.components[i - 1](x)

        # Compute total norm
        tot = 0
        for comp, (a, b), norm in zip(self.components, domains, norms):
            fn = lambda x: comp(x) / norm
            tot += quad(fn, a, b, **self.quad_args)[0]

        self.norms = norms * tot
        self.lognorms = np.log(self.norms)

        cumnorm = np.zeros(self.n_components)
        for i, (comp, (a, b), norm) in enumerate(zip(self.components, domains, self.norms)):
            fn = lambda x: comp(x) / norm
            cumnorm[i] = quad(fn, a, b, **self.quad_args)[0]

        self.cumnorm = cumnorm

    def _pdf(self, x):
        i = np.digitize(x, self.breakpoints)
        return self.components[i](x) / self.norms[i]

    def _lnpdf(self, x):
        i = np.digitize(x, self.breakpoints)
        return self.components[i].lnpdf(x) - self.lognorms[i]

    def sample(self, n, **kwargs):
        u = np.random.random(n)

        x = np.zeros(n)
        u_cumthresh = 0
        for comp, u_thresh, (a, b) in zip(self.components, self.cumnorm, self.domains):
            u_cumthresh += u_thresh
            mask = (u < u_cumthresh) & (x == 0.0)
            n_comp = mask.sum()
            samples = comp.sample(n_comp)
            oob = (samples < a) | (samples > b)
            n_oob = oob.sum()
            while n_oob:
                samples[oob] = comp.sample(n_oob)
                oob = (samples < a) | (samples > b)
                n_oob = oob.sum()

            x[mask] = samples

        return x


class GaussianPrior(BoundedPrior):
    def __init__(self, mean, sigma, bounds=None):
        self.mean = mean
        self.sigma = sigma
        self._bounds = bounds
        self._norm = 1.0

        if bounds:
            lo, hi = bounds
            a, b = (lo - mean) / sigma, (hi - mean) / sigma
            self.distribution = scipy.stats.truncnorm(a, b, loc=mean, scale=sigma)
            self.norm = _norm_cdf(b) - _norm_cdf(a)
            self.lognorm = np.log(self.norm)
        else:
            self.distribution = scipy.stats.norm(mean, sigma)
            self.norm = 1.0
            self.lognorm = 0.0

    def _pdf(self, x):
        return _norm_pdf((x - self.mean) / self.sigma) / self.sigma / self.norm

    def _lnpdf(self, x):
        return _norm_logpdf((x - self.mean) / self.sigma) - np.log(self.sigma) - self.lognorm


class LogNormalPrior(Prior):
    def __init__(self, mu, sigma, bounds=None):
        self.mu = mu
        self.sigma = sigma
        self.scale = np.exp(mu)
        self.log_s = np.log(sigma)

        self.distribution = lognorm(sigma, scale=np.exp(mu))
        self._bounds = (0, np.inf)

        super().__init__(self)

    def _pdf(self, x):
        s = self.sigma
        y = x / self.scale
        return ONE_OVER_ROOT_2PI / (s * y) * np.exp(-0.5 * (np.log(y) / s) ** 2) / self.scale

    def _lnpdf(self, x):
        s = self.sigma
        y = x / self.scale
        return LOG_ONE_OVER_ROOT_2PI - (self.log_s + np.log(y)) - 0.5 * (np.log(y) / s) ** 2 - self.mu


class FlatPrior(BoundedPrior):
    def __init__(self, bounds):
        super().__init__(bounds=bounds)

    def _pdf(self, x):
        lo, hi = self.bounds
        return 1.0 / (hi - lo)

    def sample(self, n, **kwargs):
        lo, hi = self.bounds
        return np.random.random(n) * (hi - lo) + lo


class FlatLogPrior(BoundedPrior):
    def __init__(self, bounds):
        super(FlatLogPrior, self).__init__(bounds=bounds)

    def _pdf(self, x):
        lo, hi = self.bounds
        return np.log(10) * 10 ** x / (10 ** hi - 10 ** lo)

    def sample(self, n, **kwargs):
        lo, hi = self.bounds
        return np.log10(np.random.random(n) * (10 ** hi - 10 ** lo) + 10 ** lo)


class PowerLawPrior(BoundedPrior):
    def __init__(self, alpha, bounds=None):
        self.alpha = alpha
        super(PowerLawPrior, self).__init__(bounds=bounds)

    def _pdf(self, x):
        lo, hi = self.bounds
        C = (1 + self.alpha) / (hi ** (1 + self.alpha) - lo ** (1 + self.alpha))
        # C = 1/(1/(self.alpha+1)*(1 - lo**(self.alpha+1)))
        return C * x ** self.alpha

    def _lnpdf(self, x):
        lo, hi = self.bounds
        C = (1 + self.alpha) / (hi ** (1 + self.alpha) - lo ** (1 + self.alpha))
        return np.log(C) + self.alpha * np.log(x)

    def sample(self, n, **kwargs):
        """
        cdf = C * int(x**a)|x=lo..x
            = C * [x**(a+1) / (a+1)] | x=lo..x
            = C * ([x**(a+1) / (a+1)] - [lo**(a+1) / (a+1)])
         u  = 

         u/C + [lo**(a+1) / (a+1)] = x**(a+1) / (a+1)
         (a+1) * (u/C + [lo**(a+1) / (a+1)]) = x**(a+1)
         [(a+1) * (u/C + [lo**(a+1) / (a+1)])] ** (1/(a+1)) = x
        """
        lo, hi = self.bounds
        C = (1 + self.alpha) / (hi ** (1 + self.alpha) - lo ** (1 + self.alpha))
        u = np.random.random(n)
        a = self.alpha
        return ((a + 1) * (u / C + (lo ** (a + 1) / (a + 1)))) ** (1 / (a + 1))


class FehPrior(Prior):
    """feh PDF based on local SDSS distribution

    From Jo Bovy:
    https://github.com/jobovy/apogee/blob/master/apogee/util/__init__.py#L3
    2D gaussian fit based on Casagrande (2011)
    """

    def __init__(self, halo_fraction=0.001, local=True, **kwargs):
        self.halo_fraction = halo_fraction
        self.local = local

        super().__init__(**kwargs)

    def _pdf(self, x):
        feh = x

        if self.local:
            disk_norm = 2.5066282746310007  # integral of the below from -np.inf to np.inf
            disk_fehdist = (
                1.0
                / disk_norm
                * (
                    0.8 / 0.15 * np.exp(-0.5 * (feh - 0.016) ** 2.0 / 0.15 ** 2.0)
                    + 0.2 / 0.22 * np.exp(-0.5 * (feh + 0.15) ** 2.0 / 0.22 ** 2.0)
                )
            )
        else:
            mu, sig = -0.3, 0.3
            disk_fehdist = 1.0 / np.sqrt(2 * np.pi) / sig * np.exp(-0.5 * (feh - mu) ** 2 / sig ** 2)

        halo_mu, halo_sig = -1.5, 0.4
        halo_fehdist = (
            1.0 / np.sqrt(2 * np.pi * halo_sig ** 2) * np.exp(-0.5 * (feh - halo_mu) ** 2 / halo_sig ** 2)
        )

        return self.halo_fraction * halo_fehdist + (1 - self.halo_fraction) * disk_fehdist

    def sample(self, n, **kwargs):
        if self.local:
            w1, mu1, sig1 = 0.8, 0.016, 0.15
            w2, mu2, sig2 = 0.2, -0.15, 0.22
        else:
            w1, mu1, sig1 = 1.0, -0.3, 0.3
            w2, mu2, sig2 = 0.0, 0, 1

        halo_mu, halo_sig = -1.5, 0.4

        x1 = np.random.randn(n) * sig1 + mu1
        x2 = np.random.randn(n) * sig2 + mu2

        xhalo = np.random.randn(n) * halo_sig + halo_mu

        u1 = np.random.random(n)
        x = x1
        m1 = u1 < w2
        x[m1] = x2[m1]

        u2 = np.random.random(n)
        m2 = u2 < self.halo_fraction
        x[m2] = xhalo[m2]
        return x


class EEP_prior(BoundedPrior):
    """Prior on the EEP sampling index that changes variables with d(EEP) weight.

    The EEP axis replaces either mass (isochrone grids) or age (evolution
    track grids), so the prior on EEP must carry the corresponding change of
    variables (dm_deep / dt_deep) times the replacement-variable prior.  For
    isochrone grids the replacement prior is the IMF itself.  For track grids
    ``orig_prior`` is the age prior and the IMF enters as an extra
    multiplicative ``mass_prior`` weight on the (constant along a track)
    initial mass.
    """

    def __init__(self, ic, orig_prior, bounds=None, mass_prior=None):
        self.ic = ic
        self.orig_prior = orig_prior
        self.mass_prior = mass_prior
        self._bounds = bounds if bounds is not None else ic.eep_bounds
        # Deliberately 1.0: this prior is NOT normalized.  BoundedPrior.pdf
        # divides by ``_norm``, so it must exist, but a single constant
        # cannot normalize a pdf that depends on the conditioning
        # (age, feh) / (mass, feh).  The missing factor is not constant in
        # those parameters, so treat lnprior as unnormalized (fine for MCMC at
        # fixed data; do not compare absolute values across models).
        self._norm = 1.0
        self.orig_par = ic.eep_replaces
        if self.orig_par == "age":
            self.deriv_prop = "dt_deep"
        elif self.orig_par == "mass":
            self.deriv_prop = "dm_deep"
        else:
            raise ValueError("wtf.")

    def _pdf(self, eep, **kwargs):
        if self.orig_par == "age":
            pars = [kwargs["mass"], eep, kwargs["feh"]]
        elif self.orig_par == "mass":
            pars = [eep, kwargs["age"], kwargs["feh"]]
        orig_val, dx_deep = self.ic.interp_value(pars, [self.orig_par, self.deriv_prop]).squeeze()
        pdf = self.orig_prior(orig_val) * dx_deep
        if self.orig_par == "age" and self.mass_prior is not None:
            pdf *= self.mass_prior(kwargs["mass"])
        return pdf

    def sample(self, n, **kwargs):
        eeps = pd.Series(np.arange(self.bounds[0], self.bounds[1])).sample(n, replace=True)

        if self.orig_par == "age":
            mass = kwargs["mass"]
            if isinstance(mass, pd.Series):
                mass = mass.values
            feh = kwargs["feh"]
            if isinstance(feh, pd.Series):
                feh = feh.values
            values = self.ic.interp_value([mass, eeps, feh], ["dt_deep", "age"])
            deriv_val, orig_val = values[:, 0], values[:, 1]
            orig_pr = np.array([self.orig_prior(v) for v in orig_val])
            # weights = orig_pr * np.log10(orig_val)/np.log10(np.e) * deriv_val  # why like this?
            weights = orig_pr * deriv_val
        elif self.orig_par == "mass":
            age = kwargs["age"]
            if isinstance(age, pd.Series):
                age = age.values
            feh = kwargs["feh"]
            if isinstance(feh, pd.Series):
                feh = feh.values
            values = self.ic.interp_value([eeps, age, feh], ["dm_deep", "mass"])
            deriv_val, orig_val = values[:, 0], values[:, 1]
            orig_pr = np.array([self.orig_prior(v) for v in orig_val])
            weights = orig_pr * deriv_val

        try:
            return eeps.sample(n, weights=weights, replace=True).values
        except ValueError:
            # If there are no valid samples, just run it again until you get valid results
            return self.sample(n, **kwargs)

    def test_integral(self):
        pass


# Utility numba PDFs for speed!
@nb.jit(nopython=True)
def powerlaw_pdf(x, alpha, lo, hi):
    alpha_plus_one = alpha + 1
    C = alpha_plus_one / (hi ** alpha_plus_one - lo ** alpha_plus_one)
    return C * x ** alpha


@nb.jit(nopython=True)
def powerlaw_lnpdf(x, alpha, lo, hi):
    alpha_plus_one = alpha + 1
    C = alpha_plus_one / (hi ** alpha_plus_one - lo ** alpha_plus_one)
    return log(C) + alpha * log(x)


class AgePrior(FlatLogPrior):
    """Uniform true age prior, where 'age' is actually log10(age)
    """

    def __init__(self, **kwargs):
        super().__init__(bounds=(5, 10.15), **kwargs)


class DistancePrior(PowerLawPrior):
    def __init__(self, max_distance=10000, **kwargs):
        super().__init__(alpha=2.0, bounds=(0, max_distance), **kwargs)


class AVPrior(FlatPrior):
    def __init__(self, **kwargs):
        bounds = kwargs.pop("bounds", (0, 1.0))
        super().__init__(bounds=bounds)


class QPrior(PowerLawPrior):
    def __init__(self, **kwargs):
        bounds = kwargs.pop("bounds", (0.1, 1))
        super().__init__(alpha=0.3, bounds=bounds, **kwargs)


class SalpeterPrior(PowerLawPrior):
    def __init__(self, **kwargs):
        bounds = kwargs.pop("bounds", (0.1, 10))
        super().__init__(alpha=-2.35, bounds=bounds, **kwargs)


class ChabrierPrior(BrokenPrior):
    def __init__(self, **kwargs):
        bounds = kwargs.pop("bounds", (0.1, 100.0))
        super().__init__(
            [LogNormalPrior(np.log(0.079), 0.69 * np.log(10)), PowerLawPrior(-2.35, (1.0, 100.0))], [1.0], bounds=bounds, **kwargs
        )  # from Chabrier 2003, Eqn 17
