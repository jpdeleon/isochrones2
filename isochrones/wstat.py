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

from typing import Sequence

import numpy as np
from scipy.special import gammaln


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
    poisson = -M + M * np.log(M) - S + S * np.log(S) + gammaln(B) - gammaln(M + S + B)
    return float(fixed_totals + poisson)


def _validate(x: Sequence[float], other: Sequence[float], name: str) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    if arr.ndim != 1 or arr.size == 0:
        raise ValueError("{} must be a non-empty 1D array of bin counts".format(name))
    if arr.size != np.asarray(other).size:
        raise ValueError("m and s must have the same number of bins")
    return arr