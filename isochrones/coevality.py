"""Per-component ages and a coevality test for detached binaries.

Valls-Gabaud (2016), Sec. 4 (Figs. 6-8): fit each component on its own, with
no coevality assumption, to get one age PDF per star; then either

* compare them: the PDF of ``t_A - t_B`` shows how far the components are
  from coeval (the overlap argues for or against coevality), or
* impose coevality with the prior ``pi(t) = delta(t_A - t_B)``, which
  multiplies the component PDFs into one much narrower PDF.

``StarModel(N=2)`` shares one age between components (see
``docs/deviations.rst``), so the independent ages come from two single-star
fits, one per component, and the functions here combine their age samples.
Both inputs must be samples of the *same* age variable (the package default
is ``age = log10(yr)``, which has a flat prior) from fits that used the same
age prior.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
from scipy.stats import gaussian_kde

_GRID_SIZE = 2000
_GRID_PAD_SIGMA = 4.0


def _clean(x: np.ndarray, name: str) -> np.ndarray:
    arr = np.asarray(x, dtype=float).ravel()
    arr = arr[np.isfinite(arr)]
    if arr.size < 2:
        raise ValueError("{} needs at least two finite samples".format(name))
    if np.ptp(arr) == 0.0:
        raise ValueError("{} samples are all identical; cannot estimate a PDF".format(name))
    return arr


def _common_grid(a: np.ndarray, b: np.ndarray, size: int = _GRID_SIZE) -> np.ndarray:
    pad = _GRID_PAD_SIGMA * max(a.std(ddof=1), b.std(ddof=1))
    return np.linspace(min(a.min(), b.min()) - pad, max(a.max(), b.max()) + pad, size)


def coeval_age_posterior(
    samples_a: np.ndarray,
    samples_b: np.ndarray,
    grid: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Age PDF with coevality imposed, ``pi(t) = delta(t_A - t_B)``.

    The component PDFs are estimated by Gaussian KDE, multiplied on a common
    grid and renormalised.

    Args:
        samples_a: Age samples of component A.
        samples_b: Age samples of component B.
        grid: Evaluation grid; by default a regular grid covering both.

    Returns:
        ``(grid, pdf)`` with ``pdf`` integrating to one.

    Raises:
        ValueError: If a sample set is too small or constant, or if the two
            PDFs do not overlap anywhere on the grid.
    """
    a, b = _clean(samples_a, "samples_a"), _clean(samples_b, "samples_b")
    grid = _common_grid(a, b) if grid is None else np.asarray(grid, dtype=float)
    product = gaussian_kde(a)(grid) * gaussian_kde(b)(grid)
    norm = np.trapezoid(product, grid)
    if not norm > 0.0:
        raise ValueError("component age PDFs do not overlap on the grid")
    return grid, product / norm


def age_difference_samples(
    samples_a: np.ndarray,
    samples_b: np.ndarray,
    n: int = 20000,
    seed: Optional[int] = None,
) -> np.ndarray:
    """Draw samples of ``t_A - t_B`` assuming the fits are independent.

    Args:
        samples_a: Age samples of component A.
        samples_b: Age samples of component B.
        n: Number of differences to draw.
        seed: Seed for the random generator.

    Returns:
        1D array of length ``n``.
    """
    a, b = _clean(samples_a, "samples_a"), _clean(samples_b, "samples_b")
    rng = np.random.default_rng(seed)
    return rng.choice(a, size=n) - rng.choice(b, size=n)


def coevality_summary(
    samples_a: np.ndarray,
    samples_b: np.ndarray,
    alpha: float = 0.95,
    n: int = 20000,
    seed: Optional[int] = None,
) -> Dict[str, float]:
    """Summarise how consistent the component ages are with coevality.

    Args:
        samples_a: Age samples of component A.
        samples_b: Age samples of component B.
        alpha: Coverage of the central interval of ``t_A - t_B``.
        n: Number of differences to draw.
        seed: Seed for the random generator.

    Returns:
        Dict with ``delta_median``, ``delta_lo`` / ``delta_hi`` (central
        interval of ``t_A - t_B``), ``zero_in_interval`` (1.0 if ``t_A = t_B``
        is inside it, else 0.0), ``overlap`` (``int min(p_A, p_B) dt``, 1 for
        identical PDFs and 0 for disjoint ones) and ``mode_coeval`` (mode of
        the coevality-imposed PDF).
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")
    a, b = _clean(samples_a, "samples_a"), _clean(samples_b, "samples_b")

    delta = age_difference_samples(a, b, n=n, seed=seed)
    tail = (1.0 - alpha) / 2.0
    lo, hi = np.quantile(delta, [tail, 1.0 - tail])

    grid = _common_grid(a, b)
    pa, pb = gaussian_kde(a)(grid), gaussian_kde(b)(grid)
    overlap = float(np.trapezoid(np.minimum(pa, pb), grid))
    _, pdf = coeval_age_posterior(a, b, grid)

    return {
        "delta_median": float(np.median(delta)),
        "delta_lo": float(lo),
        "delta_hi": float(hi),
        "zero_in_interval": float(lo <= 0.0 <= hi),
        "overlap": overlap,
        "mode_coeval": float(grid[np.argmax(pdf)]),
    }
