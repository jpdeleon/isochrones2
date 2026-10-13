"""Saha's W statistic for model vs. observed CMD comparisons.

The W statistic (Saha 1998; Hernandez, Valls-Gabaud & Gilmore 1999;
see vallsgabaud2016, Sec. 5) compares binned model counts ``m_i`` with
binned observed counts ``s_i`` on an equal footing, and serves both
parameter fitting (vary ``m`` for fixed observed ``s``) and goodness of
fit (compare the W distribution for simulated data against that of the
actual data).

The logged forms here avoid the factorial overflow that the literal
``(m_i + s_i)! / (m_i! s_i!)`` product would incur for realistic counts,
and generalize to real-valued (Poisson expectation) model counts via
``gammaln``.
"""

from __future__ import annotations

from typing import NamedTuple, Optional, Sequence, Tuple

import numpy as np
from scipy.special import gammaln, xlogy


def log_w(m: Sequence[float], s: Sequence[float]) -> float:
    """Log of Saha's W for fixed totals (vallsgabaud2016, Eq. ``wstat1``).

    For a fixed number of bins ``B``, fixed total model count ``M`` and
    fixed total observed count ``S``, the normalization is constant and
    the posterior weight of the counts is

    .. math::

        W = \\prod_i \\frac{(m_i + s_i)!}{m_i!\\,s_i!}

    Args:
        m: Model counts per bin (length ``B``).
        s: Observed counts per bin (length ``B``).

    Returns:
        ``log(W)``.

    Raises:
        ValueError: If ``m`` and ``s`` have different lengths or are
            empty.
    """
    m_arr = _validate(m, s, "m")
    s_arr = _validate(s, m_arr, "s")
    return float(np.sum(gammaln(m_arr + s_arr + 1.0) - gammaln(m_arr + 1.0) - gammaln(s_arr + 1.0)))


def log_w_poisson(m: Sequence[float], s: Sequence[float]) -> float:
    """Log of the Poisson-modified W (vallsgabaud2016, Eq. ``wstat2``).

    When the totals ``M`` and ``S`` are themselves Poisson expectations
    rather than fixed, W picks up the normalization

    .. math::

        \\frac{e^{-M} M^M \\, e^{-S} S^S \\, (B-1)!}{(M+S+B-1)!}

    so that

    .. math::

        P(\\{s_i\\}|\\{m_i\\}) = e^{-M} M^M e^{-S} S^S \\frac{(B-1)!}{(M+S+B-1)!}
        \\prod_i \\frac{(m_i+s_i)!}{m_i!\\,s_i!}

    Args:
        m: Model counts per bin (length ``B``, may be real).
        s: Observed counts per bin (length ``B``).

    Returns:
        ``log(P)``.

    Raises:
        ValueError: If ``m`` and ``s`` have different lengths or are
            empty.
    """
    m_arr = _validate(m, s, "m")
    s_arr = _validate(s, m_arr, "s")
    B = float(m_arr.size)
    M, S = float(m_arr.sum()), float(s_arr.sum())
    fixed_totals = np.sum(gammaln(m_arr + s_arr + 1.0) - gammaln(m_arr + 1.0) - gammaln(s_arr + 1.0))
    # xlogy(x, x) is x*log(x) with the x -> 0 limit of 0 (no NaN for empty totals).
    poisson = -M + xlogy(M, M) - S + xlogy(S, S) + gammaln(B) - gammaln(M + S + B)
    return float(fixed_totals + poisson)


def bin_cmd(
    color: Sequence[float],
    mag: Sequence[float],
    color_edges: Sequence[float],
    mag_edges: Sequence[float],
    weights: Optional[Sequence[float]] = None,
) -> np.ndarray:
    """Bin stars on a colour-magnitude diagram into a flat vector of counts.

    The result can be passed as ``m`` or ``s`` to :func:`log_w` /
    :func:`log_w_poisson`.  Use the *same* edges for the model and the
    observed CMD; stars outside the edges are dropped.

    Args:
        color: Colour of each star.
        mag: Magnitude of each star.
        color_edges: Increasing bin edges along the colour axis.
        mag_edges: Increasing bin edges along the magnitude axis.
        weights: Optional weight per star (e.g. to scale model counts to an
            expected number of stars).

    Returns:
        1D array of length ``(len(color_edges) - 1) * (len(mag_edges) - 1)``.

    Raises:
        ValueError: If ``color`` and ``mag`` differ in length.
    """
    color = np.asarray(color, dtype=float)
    mag = np.asarray(mag, dtype=float)
    if color.shape != mag.shape or color.ndim != 1:
        raise ValueError("color and mag must be 1D arrays of the same length")
    counts, _, _ = np.histogram2d(color, mag, bins=[color_edges, mag_edges], weights=weights)
    return counts.ravel()


class WGoodnessOfFit(NamedTuple):
    """Result of :func:`w_goodness_of_fit`.

    Attributes:
        log_w_observed: ``log W`` of the model against the observed counts.
        log_w_simulated: ``log W`` of the model against each simulated data
            set drawn from the model.
        p_value: Fraction of simulated data sets whose ``log W`` is not
            larger than the observed one.  Small values mean the observed
            CMD is a worse match to the model than model-generated data.
    """

    log_w_observed: float
    log_w_simulated: np.ndarray
    p_value: float


def w_goodness_of_fit(
    m: Sequence[float],
    s: Sequence[float],
    n_sim: int = 1000,
    seed: Optional[int] = None,
) -> WGoodnessOfFit:
    """Compare W for simulated data sets with W for the observed data.

    Following vallsgabaud2016, Sec. 5: draw data sets from the model (each
    bin Poisson with mean ``m_i``), compute :func:`log_w_poisson` between
    the model and each, and ask where the same statistic for the real data
    falls in that distribution.  The same statistic therefore serves fitting
    and goodness of fit.

    Args:
        m: Model expected counts per bin (non-negative, may be real).
        s: Observed counts per bin.
        n_sim: Number of simulated data sets.
        seed: Seed for the random generator (for reproducibility).

    Returns:
        A :class:`WGoodnessOfFit`.

    Raises:
        ValueError: If ``m`` and ``s`` differ in length, are empty, if any
            ``m`` is negative, or if ``n_sim`` is not positive.
    """
    m_arr = _validate(m, s, "m")
    s_arr = _validate(s, m_arr, "s")
    if (m_arr < 0).any():
        raise ValueError("m must be non-negative")
    if n_sim < 1:
        raise ValueError("n_sim must be positive")

    rng = np.random.default_rng(seed)
    simulated = rng.poisson(m_arr, size=(n_sim, m_arr.size)).astype(float)
    log_w_sim = np.array([log_w_poisson(m_arr, row) for row in simulated])
    observed = log_w_poisson(m_arr, s_arr)
    return WGoodnessOfFit(observed, log_w_sim, float(np.mean(log_w_sim <= observed)))


def _validate(x: Sequence[float], other: Sequence[float], name: str) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    if arr.ndim != 1 or arr.size == 0:
        raise ValueError("{} must be a non-empty 1D array of bin counts".format(name))
    if arr.size != np.asarray(other).size:
        raise ValueError("m and s must have the same number of bins")
    return arr