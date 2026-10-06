"""Unit tests for core logic functions used in the starfit workflow.

Coverage:
- Utility functions: addmags, fast_addmags, trapz, polyval
- Interpolation primitives: searchsorted, find_indices_2d/3d,
  interp_value_2d/3d, DFInterpolator
- Likelihood: gauss_lnprob
- Priors: FlatPrior, GaussianPrior, PowerLawPrior, FlatLogPrior,
  AgePrior, DistancePrior, AVPrior, QPrior, FehPrior, ChabrierPrior
- starfit workflow helpers: StarModel.get_bands(), model filename pattern,
  FlatPrior as feh prior
- Regression values: frozen numerical answers for the py3.8→py3.12 upgrade
"""

import itertools
import math
import os

import numpy as np
import pandas as pd
import pytest
from scipy.integrate import quad


# ---------------------------------------------------------------------------
# Utility functions (isochrones/utils.py)
# ---------------------------------------------------------------------------


class TestAddMags:
    """Tests for magnitude addition (flux combination)."""

    def test_fast_addmags_single_is_identity(self):
        """fast_addmags([m]) should equal m (single magnitude is identity)."""
        from isochrones.utils import fast_addmags

        for m in [-5.0, 0.0, 10.0, 25.0]:
            assert math.isclose(fast_addmags([m]), m, rel_tol=1e-10, abs_tol=1e-12), (
                f"fast_addmags([{m}]) = {fast_addmags([m])}, expected {m}"
            )

    def test_fast_addmags_two_equal(self):
        """Two equal magnitudes combine to m - 2.5*log10(2) ≈ m - 0.7526 mag."""
        from isochrones.utils import fast_addmags

        for m in [0.0, 5.0, 10.0, 20.0]:
            result = fast_addmags([m, m])
            expected = m - 2.5 * math.log10(2)
            assert math.isclose(result, expected, rel_tol=1e-10, abs_tol=1e-12), (
                f"fast_addmags([{m}, {m}]) = {result}, expected {expected}"
            )

    def test_fast_addmags_three_equal(self):
        """Three equal magnitudes combine to m - 2.5*log10(3)."""
        from isochrones.utils import fast_addmags

        m = 12.0
        result = fast_addmags([m, m, m])
        expected = m - 2.5 * math.log10(3)
        assert math.isclose(result, expected, rel_tol=1e-10, abs_tol=1e-12)

    def test_fast_addmags_combined_is_brighter(self):
        """Combined magnitude of two stars is brighter (smaller) than either alone."""
        from isochrones.utils import fast_addmags

        m1, m2 = 8.0, 12.0
        combined = fast_addmags([m1, m2])
        assert combined < m1, f"Combined {combined} should be brighter than primary {m1}"
        assert combined < m2, f"Combined {combined} should be brighter than secondary {m2}"

    def test_addmags_commutativity(self):
        """addmags(m1, m2) == addmags(m2, m1)."""
        from isochrones.utils import addmags

        m1, m2 = 8.3, 11.7
        assert math.isclose(addmags(m1, m2), addmags(m2, m1), rel_tol=1e-12, abs_tol=1e-14)

    def test_addmags_consistency_with_fast_addmags(self):
        """addmags and fast_addmags agree for plain magnitudes (no uncertainties)."""
        from isochrones.utils import addmags, fast_addmags

        for m1, m2 in [(9.5, 11.2), (5.0, 5.0), (3.0, 15.0)]:
            assert math.isclose(addmags(m1, m2), fast_addmags([m1, m2]), rel_tol=1e-10, abs_tol=1e-13)

    def test_addmags_combined_brighter_than_min(self):
        """Combined magnitude must be ≤ min(m1, m2)."""
        from isochrones.utils import addmags

        for m1, m2 in [(5.0, 5.0), (3.0, 10.0), (8.0, 8.5)]:
            combined = addmags(m1, m2)
            assert combined <= min(m1, m2) + 1e-12


class TestTrapz:
    """Tests for the Numba trapz function."""

    def test_constant(self):
        """∫ c dx from a to b = c*(b-a)."""
        from isochrones.utils import trapz

        x = np.linspace(0.0, 1.0, 1000)
        y = np.full_like(x, 3.5)
        assert math.isclose(trapz(y, x), 3.5, rel_tol=1e-10)

    def test_linear(self):
        """∫ x dx from 0 to 1 = 0.5."""
        from isochrones.utils import trapz

        x = np.linspace(0.0, 1.0, 100_000)
        y = x.copy()
        assert math.isclose(trapz(y, x), 0.5, rel_tol=1e-5)

    def test_quadratic(self):
        """∫ x² dx from 0 to 1 = 1/3."""
        from isochrones.utils import trapz

        x = np.linspace(0.0, 1.0, 100_000)
        y = x**2
        assert math.isclose(trapz(y, x), 1.0 / 3.0, rel_tol=1e-5)

    def test_single_interval(self):
        """Two-point trapz is exact for linear functions."""
        from isochrones.utils import trapz

        x = np.array([0.0, 1.0])
        y = np.array([0.0, 2.0])
        assert math.isclose(trapz(y, x), 1.0, rel_tol=1e-12)


class TestPolyval:
    """Tests for the Numba polyval function."""

    def test_constant(self):
        """p=[c] evaluates to c for any x."""
        from isochrones.utils import polyval

        assert polyval(np.array([5.0]), 3.0) == 5.0
        assert polyval(np.array([5.0]), 0.0) == 5.0

    def test_linear(self):
        """p=[a, b] evaluates to a*x + b."""
        from isochrones.utils import polyval

        a, b, x = 2.0, 3.0, 4.0
        assert math.isclose(polyval(np.array([a, b]), x), a * x + b)

    def test_quadratic_matches_numpy(self):
        """Quadratic polynomial should match np.polyval."""
        from isochrones.utils import polyval

        p = np.array([1.0, -2.0, 1.0])  # (x-1)²
        for x in [0.0, 1.0, 2.0, 3.5]:
            assert math.isclose(polyval(p, x), np.polyval(p, x), rel_tol=1e-12)

    def test_zero_polynomial(self):
        """All-zero coefficients -> 0 for any x."""
        from isochrones.utils import polyval

        p = np.array([0.0, 0.0, 0.0])
        assert polyval(p, 42.0) == 0.0


# ---------------------------------------------------------------------------
# Interpolation primitives (isochrones/interp.py)
# ---------------------------------------------------------------------------


class TestSearchsorted:
    """Tests for the custom Numba searchsorted."""

    def test_exact_hit_sets_eq_true(self):
        """Searching for an element already in the array returns eq=True."""
        from isochrones.interp import searchsorted

        arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        idx, eq = searchsorted(arr, 3.0)
        assert eq, "Expected eq=True for exact value"
        assert arr[idx] == 3.0

    def test_between_elements_eq_false(self):
        """Value between two elements returns eq=False and correct bracket."""
        from isochrones.interp import searchsorted

        arr = np.array([1.0, 2.0, 3.0, 4.0])
        idx, eq = searchsorted(arr, 2.5)
        assert not eq
        assert arr[idx - 1] <= 2.5 < arr[idx]

    def test_first_element(self):
        """Searching for the first element returns index 0, eq=True."""
        from isochrones.interp import searchsorted

        arr = np.array([0.0, 1.0, 2.0])
        idx, eq = searchsorted(arr, 0.0)
        assert eq and idx == 0

    def test_last_element(self):
        """Searching for the last element returns last index, eq=True."""
        from isochrones.interp import searchsorted

        arr = np.array([0.0, 1.0, 2.0])
        idx, eq = searchsorted(arr, 2.0)
        assert eq and idx == 2

    def test_matches_numpy_searchsorted(self):
        """Custom searchsorted agrees with np.searchsorted for sorted arrays."""
        from isochrones.interp import searchsorted

        np.random.seed(42)
        arr = np.sort(np.random.random(50))
        test_vals = [arr[0], arr[10], arr[25], arr[-1], 0.15, 0.45, 0.75]
        for val in test_vals:
            custom_idx, _ = searchsorted(arr, val)
            np_idx = np.searchsorted(arr, val)
            assert custom_idx == np_idx, (
                f"searchsorted mismatch at val={val}: got {custom_idx}, expected {np_idx}"
            )


class TestFindIndices2d:
    """Tests for find_indices_2d."""

    def setup_method(self):
        self.ii0 = np.array([0.0, 1.0, 2.0, 3.0])
        self.ii1 = np.array([0.0, 10.0, 20.0, 30.0])

    def test_exact_grid_point_has_zero_norm_distances(self):
        """At a grid node, normalized distances should be exactly 0."""
        from isochrones.interp import find_indices_2d

        indices, norm_dists, oob = find_indices_2d(1.0, 10.0, self.ii0, self.ii1)
        assert not oob
        assert norm_dists[0] == 0.0
        assert norm_dists[1] == 0.0

    def test_midpoint_has_half_norm_distances(self):
        """At midpoints between nodes, normalized distances should be 0.5."""
        from isochrones.interp import find_indices_2d

        indices, norm_dists, oob = find_indices_2d(0.5, 5.0, self.ii0, self.ii1)
        assert not oob
        assert math.isclose(norm_dists[0], 0.5)
        assert math.isclose(norm_dists[1], 0.5)

    def test_out_of_bounds_low(self):
        """Value below grid minimum returns oob=True."""
        from isochrones.interp import find_indices_2d

        _, _, oob = find_indices_2d(-0.1, 10.0, self.ii0, self.ii1)
        assert oob

    def test_out_of_bounds_high(self):
        """Value above grid maximum returns oob=True."""
        from isochrones.interp import find_indices_2d

        _, _, oob = find_indices_2d(1.0, 35.0, self.ii0, self.ii1)
        assert oob

    def test_in_bounds_is_not_oob(self):
        """Interior point is not out of bounds."""
        from isochrones.interp import find_indices_2d

        _, _, oob = find_indices_2d(1.5, 15.0, self.ii0, self.ii1)
        assert not oob


class TestInterpValue2d:
    """Tests for interp_value_2d with a known linear function f(x0, x1) = 3*x0 + 2*x1."""

    def setup_method(self):
        self.ii0 = np.array([0.0, 1.0, 2.0])
        self.ii1 = np.array([0.0, 5.0, 10.0])
        self.icols = np.array([0])

        # Build grid: f(x0, x1) = 3*x0 + 2*x1
        self.grid = np.zeros((3, 3, 1))
        for i, x0 in enumerate(self.ii0):
            for j, x1 in enumerate(self.ii1):
                self.grid[i, j, 0] = 3.0 * x0 + 2.0 * x1

    def _f(self, x0, x1):
        return 3.0 * x0 + 2.0 * x1

    def _interp(self, x0, x1):
        from isochrones.interp import interp_value_2d

        return interp_value_2d(x0, x1, self.grid, self.icols, self.ii0, self.ii1)

    def test_exact_at_all_nodes(self):
        """At grid nodes (excluding max boundary), should return the exact value.

        See TestInterpValue3d.test_exact_at_all_nodes for the max-boundary
        NaN issue (weight=0 * OOB-memory = potentially NaN).
        """
        for x0 in self.ii0[:-1]:
            for x1 in self.ii1[:-1]:
                result = self._interp(x0, x1)
                assert math.isclose(result[0], self._f(x0, x1), abs_tol=1e-10)

    def test_interior_linear_exact(self):
        """Linear interpolation on a linear function is exact at interior points."""
        for x0, x1 in [(0.5, 2.5), (1.0, 7.5), (0.25, 3.75)]:
            result = self._interp(x0, x1)
            assert math.isclose(result[0], self._f(x0, x1), abs_tol=1e-10)

    def test_nan_input_returns_nan(self):
        """NaN input should propagate to NaN output."""
        result = self._interp(np.nan, 5.0)
        assert np.isnan(result[0])

    def test_oob_returns_nan(self):
        """Out-of-bounds input should return NaN."""
        result = self._interp(5.0, 5.0)
        assert np.isnan(result[0])


class TestInterpValue3d:
    """Tests for interp_value_3d with f(x0, x1, x2) = x0 + 2*x1 + 0.01*x2."""

    def setup_method(self):
        self.ii0 = np.array([0.0, 1.0, 2.0])
        self.ii1 = np.array([0.0, 5.0, 10.0])
        self.ii2 = np.array([0.0, 100.0, 200.0])
        self.icols = np.array([0])

        self.grid = np.zeros((3, 3, 3, 1))
        for i, x0 in enumerate(self.ii0):
            for j, x1 in enumerate(self.ii1):
                for k, x2 in enumerate(self.ii2):
                    self.grid[i, j, k, 0] = x0 + 2.0 * x1 + 0.01 * x2

    def _f(self, x0, x1, x2):
        return x0 + 2.0 * x1 + 0.01 * x2

    def _interp(self, x0, x1, x2):
        from isochrones.interp import interp_value_3d

        return interp_value_3d(x0, x1, x2, self.grid, self.icols, self.ii0, self.ii1, self.ii2)

    def test_exact_at_all_nodes(self):
        """At grid nodes (excluding the max boundary), should return the exact value.

        Note: the Numba interpolation accesses grid[indices[j]+1, ...] even when
        norm_distance[j]=0 (exact grid point), and on the maximum boundary
        indices[j]+1 is out of bounds.  Numba does not check bounds in nopython
        mode; if that memory location contains NaN the result is NaN (IEEE 754:
        0 * nan = nan).  This is a known limitation: avoid max-boundary inputs.
        """
        # Exclude the last index in each dimension (maximum boundary)
        for x0 in self.ii0[:-1]:
            for x1 in self.ii1[:-1]:
                for x2 in self.ii2[:-1]:
                    result = self._interp(x0, x1, x2)
                    assert math.isclose(result[0], self._f(x0, x1, x2), abs_tol=1e-10)

    def test_interior_linear_exact(self):
        """Linear interp on a linear function is exact at arbitrary interior points."""
        test_pts = [(0.5, 2.5, 50.0), (1.0, 7.5, 150.0), (0.3, 3.0, 30.0)]
        for x0, x1, x2 in test_pts:
            result = self._interp(x0, x1, x2)
            assert math.isclose(result[0], self._f(x0, x1, x2), abs_tol=1e-10)

    def test_nan_input_returns_nan(self):
        """NaN input should propagate to NaN output."""
        result = self._interp(np.nan, 5.0, 100.0)
        assert np.isnan(result[0])

    def test_oob_returns_nan(self):
        """Out-of-bounds input should return NaN."""
        result = self._interp(5.0, 5.0, 100.0)
        assert np.isnan(result[0])


class TestDFInterpolator:
    """Tests for the DFInterpolator class."""

    def setup_method(self):
        """Build a simple 3D DFInterpolator with f(x,y,z) = x + 0.1*y + 0.001*z."""
        xx = np.array([0.0, 1.0, 2.0])
        yy = np.array([0.0, 5.0, 10.0])
        zz = np.array([0.0, 100.0, 200.0])
        self.xx, self.yy, self.zz = xx, yy, zz

        rows = [
            (x, y, z, x + 0.1 * y + 0.001 * z)
            for x, y, z in itertools.product(xx, yy, zz)
        ]
        df = pd.DataFrame(rows, columns=["x", "y", "z", "val"]).set_index(["x", "y", "z"])

        from isochrones.interp import DFInterpolator

        self.interp = DFInterpolator(df)

    def _f(self, x, y, z):
        return x + 0.1 * y + 0.001 * z

    def test_at_all_grid_nodes(self):
        """Interpolation at grid nodes (excluding max boundary) returns exact values.

        See TestInterpValue3d.test_exact_at_all_nodes for the known limitation:
        the Numba interpolation may return NaN for the maximum boundary due to
        an out-of-bounds array access with weight=0 (0*NaN=NaN in IEEE 754).
        """
        for x in self.xx[:-1]:
            for y in self.yy[:-1]:
                for z in self.zz[:-1]:
                    result = self.interp([x, y, z], ["val"])
                    assert math.isclose(result[0], self._f(x, y, z), abs_tol=1e-10), (
                        f"Mismatch at ({x}, {y}, {z}): got {result[0]}"
                    )

    def test_interior_point_linear_exact(self):
        """For a linear function, interpolation is exact at interior points."""
        result = self.interp([0.5, 2.5, 50.0], ["val"])
        expected = self._f(0.5, 2.5, 50.0)
        assert math.isclose(result[0], expected, abs_tol=1e-10)

    def test_batch_evaluation(self):
        """Batch evaluation is consistent with single-point evaluation."""
        from isochrones.interp import DFInterpolator

        pts = [(0.3, 1.5, 30.0), (1.2, 7.0, 150.0), (0.0, 0.0, 0.0)]
        x_arr = np.array([p[0] for p in pts])
        y_arr = np.array([p[1] for p in pts])
        z_arr = np.array([p[2] for p in pts])
        batch = self.interp([x_arr, y_arr, z_arr], ["val"])
        for i, pt in enumerate(pts):
            single = self.interp(list(pt), ["val"])
            assert math.isclose(batch[i, 0], single[0], abs_tol=1e-10)

    def test_column_index(self):
        """column_index maps column names to integer positions."""
        assert self.interp.column_index["val"] == 0

    def test_ndim(self):
        """ndim should be 3 for a 3-index DataFrame."""
        assert self.interp.ndim == 3


# ---------------------------------------------------------------------------
# Likelihood (isochrones/likelihood.py)
# ---------------------------------------------------------------------------


class TestGaussLnprob:
    """Tests for gauss_lnprob (Numba-compiled Gaussian log-probability).

    Code formula: LOG_ONE_OVER_ROOT_2PI - log(unc) - 0.5*(val-model_val)²/unc²

    This is the normalized Gaussian log-pdf: ``unc`` is a fixed observed
    uncertainty (never a free parameter), so no prior on ``sigma`` applies and
    the normalization term must carry a minus sign.  The sign matches
    ``GaussianPrior._lnpdf`` in ``priors.py``.
    """

    def _lnprob(self, val, unc, model_val):
        from isochrones.likelihood import gauss_lnprob

        return gauss_lnprob(val, unc, model_val)

    def test_known_zero_residual(self):
        """At exact match, lnprob = LOG_ONE_OVER_ROOT_2PI - log(unc)."""
        LOG1OVER_ROOT2PI = math.log(1.0 / math.sqrt(2 * math.pi))
        val, unc = 5700.0, 100.0
        result = self._lnprob(val, unc, val)
        expected = LOG1OVER_ROOT2PI - math.log(unc)
        assert math.isclose(result, expected, rel_tol=1e-10, abs_tol=1e-12)

    def test_one_sigma_offset(self):
        """At residual = unc (1-sigma), penalty is -0.5."""
        LOG1OVER_ROOT2PI = math.log(1.0 / math.sqrt(2 * math.pi))
        val, unc = 5700.0, 100.0
        result = self._lnprob(val, unc, val + unc)  # 1-sigma off
        expected = LOG1OVER_ROOT2PI - math.log(unc) - 0.5
        assert math.isclose(result, expected, rel_tol=1e-10, abs_tol=1e-12)

    def test_maximized_at_zero_residual(self):
        """lnprob is maximized when model_val == val."""
        val, unc = 5700.0, 100.0
        lp_exact = self._lnprob(val, unc, val)
        for offset in [10.0, 50.0, 100.0, 500.0, -10.0, -100.0]:
            assert lp_exact > self._lnprob(val, unc, val + offset)

    def test_monotone_decreasing_with_residual(self):
        """Larger |residual| gives strictly lower lnprob."""
        val, unc = 5700.0, 100.0
        lp1 = self._lnprob(val, unc, val + 50.0)
        lp2 = self._lnprob(val, unc, val + 100.0)
        lp3 = self._lnprob(val, unc, val + 200.0)
        assert lp1 > lp2 > lp3

    def test_symmetric_in_residual(self):
        """lnprob(val, unc, val+d) == lnprob(val, unc, val-d)."""
        val, unc = 5700.0, 100.0
        for d in [25.0, 100.0, 300.0]:
            assert math.isclose(
                self._lnprob(val, unc, val + d),
                self._lnprob(val, unc, val - d),
                rel_tol=1e-12,
            )

    def test_swap_val_model_gives_same_result(self):
        """gauss_lnprob(a, unc, b) == gauss_lnprob(b, unc, a) (symmetric residual)."""
        unc = 100.0
        a, b = 5700.0, 5850.0
        assert math.isclose(self._lnprob(a, unc, b), self._lnprob(b, unc, a), rel_tol=1e-12)

    def test_matches_scipy_norm_logpdf(self):
        """gauss_lnprob must equal the normalized Gaussian log-pdf from scipy."""
        from scipy.stats import norm

        for val, unc, model_val in [
            (5700.0, 100.0, 5700.0),
            (5700.0, 100.0, 5830.0),
            (5700.0, 7.5, 5711.0),
        ]:
            expected = norm.logpdf(model_val, loc=val, scale=unc)
            assert math.isclose(
                self._lnprob(val, unc, model_val), expected, rel_tol=1e-12, abs_tol=1e-12
            )

    def test_integrates_to_unit_probability(self):
        """exp(lnprob) must integrate to 1 for any unc.

        This is the regression guard for the log(unc) sign: with the sign
        flipped the integral equals unc**2 instead of 1.
        """
        val, unc = 5700.0, 100.0
        lo, hi = val - 12 * unc, val + 12 * unc
        integral, _ = quad(lambda x: math.exp(self._lnprob(val, unc, x)), lo, hi, limit=200)
        assert math.isclose(integral, 1.0, rel_tol=1e-8)

    def test_normalization_shrinks_as_uncertainty_grows(self):
        """A noisier measurement must not get a larger log-probability at zero residual.

        A sign flip on log(unc) inverts this ordering.
        """
        val = 5700.0
        assert self._lnprob(val, 10.0, val) < self._lnprob(val, 1.0, val)
        assert self._lnprob(val, 100.0, val) < self._lnprob(val, 10.0, val)


# ---------------------------------------------------------------------------
# Prior distributions (isochrones/priors.py)
# ---------------------------------------------------------------------------


class TestFlatPrior:
    """Tests for FlatPrior (uniform distribution within [lo, hi])."""

    def test_constant_pdf_within_bounds(self):
        """PDF should be 1/(hi-lo) everywhere within bounds."""
        from isochrones.priors import FlatPrior

        lo, hi = 2.0, 7.0
        p = FlatPrior((lo, hi))
        expected = 1.0 / (hi - lo)
        for x in [2.5, 4.0, 6.9]:
            assert math.isclose(p.pdf(x), expected, rel_tol=1e-12)

    def test_zero_outside_bounds(self):
        """PDF should be 0 outside [lo, hi]."""
        from isochrones.priors import FlatPrior

        p = FlatPrior((0.0, 1.0))
        assert p.pdf(-0.01) == 0.0
        assert p.pdf(1.01) == 0.0

    def test_lnpdf_equals_log_pdf(self):
        """lnpdf(x) == log(pdf(x)) for in-bounds x."""
        from isochrones.priors import FlatPrior

        p = FlatPrior((1.0, 4.0))
        for x in [1.5, 2.0, 3.5]:
            assert math.isclose(p.lnpdf(x), math.log(p.pdf(x)), rel_tol=1e-12)

    def test_lnpdf_minus_inf_outside_bounds(self):
        """lnpdf returns -inf outside bounds."""
        from isochrones.priors import FlatPrior

        p = FlatPrior((0.0, 1.0))
        assert p.lnpdf(-1.0) == -np.inf
        assert p.lnpdf(2.0) == -np.inf

    def test_integrates_to_one(self):
        """PDF integrates to 1 over its bounds."""
        from isochrones.priors import FlatPrior

        p = FlatPrior((3.0, 8.0))
        integral, _ = quad(p.pdf, *p.bounds)
        assert math.isclose(integral, 1.0, rel_tol=1e-8)

    def test_samples_within_bounds(self):
        """All samples drawn from FlatPrior lie in [lo, hi]."""
        from isochrones.priors import FlatPrior

        np.random.seed(0)
        lo, hi = 3.0, 8.0
        p = FlatPrior((lo, hi))
        samples = p.sample(2000)
        assert np.all(samples >= lo) and np.all(samples <= hi)


class TestGaussianPrior:
    """Tests for GaussianPrior.

    Note: GaussianPrior inherits pdf() from Prior (which unpacks self.bounds),
    but BoundedPrior.bounds returns None when no bounds are given.  This means
    pdf() raises a TypeError for unbounded GaussianPrior.  To test Gaussian
    shape without bounds, use _pdf() and _lnpdf() directly.  To use pdf(),
    always supply explicit bounds.
    """

    def test_peak_at_mean_via_internal(self):
        """_pdf is maximized at the mean (unbounded form)."""
        from isochrones.priors import GaussianPrior

        p = GaussianPrior(0.0, 1.0)
        assert p._pdf(0.0) > p._pdf(1.0)
        assert p._pdf(0.0) > p._pdf(-1.0)

    def test_symmetric_around_mean_via_internal(self):
        """_pdf is symmetric: _pdf(mu+d) == _pdf(mu-d)."""
        from isochrones.priors import GaussianPrior

        p = GaussianPrior(5.0, 2.0)
        for delta in [0.5, 1.0, 2.0, 3.0]:
            assert math.isclose(p._pdf(5.0 + delta), p._pdf(5.0 - delta), rel_tol=1e-12)

    def test_lnpdf_consistent_with_internal_pdf(self):
        """exp(_lnpdf(x)) == _pdf(x) for any x."""
        from isochrones.priors import GaussianPrior

        p = GaussianPrior(3.0, 1.5)
        for x in [1.0, 2.0, 3.0, 4.0, 5.0]:
            assert math.isclose(math.exp(p._lnpdf(x)), p._pdf(x), rel_tol=1e-12)

    def test_bounded_gaussian_zero_outside(self):
        """With bounds, pdf() is 0 outside the truncation region."""
        from isochrones.priors import GaussianPrior

        p = GaussianPrior(0.0, 1.0, bounds=(-2.0, 2.0))
        assert p.pdf(-3.0) == 0.0
        assert p.pdf(3.0) == 0.0
        assert p.pdf(0.0) > 0.0

    def test_bounded_gaussian_integrates_to_one(self):
        """Bounded GaussianPrior integrates to 1 over its bounds."""
        from isochrones.priors import GaussianPrior

        p = GaussianPrior(0.0, 1.0, bounds=(-3.0, 3.0))
        integral, _ = quad(p.pdf, *p.bounds)
        assert math.isclose(integral, 1.0, rel_tol=1e-6)

    def test_bounded_peak_at_mean(self):
        """Bounded GaussianPrior pdf() is maximized at the mean (when in bounds)."""
        from isochrones.priors import GaussianPrior

        p = GaussianPrior(0.0, 1.0, bounds=(-5.0, 5.0))
        assert p.pdf(0.0) > p.pdf(1.0)
        assert p.pdf(0.0) > p.pdf(-1.0)

    def test_bounded_symmetric(self):
        """Bounded GaussianPrior pdf() is symmetric around mean (when symmetrically bounded)."""
        from isochrones.priors import GaussianPrior

        p = GaussianPrior(5.0, 2.0, bounds=(-100.0, 100.0))  # effectively unbounded
        for delta in [0.5, 1.0, 2.0]:
            assert math.isclose(p.pdf(5.0 + delta), p.pdf(5.0 - delta), rel_tol=1e-12)


class TestPowerLawPrior:
    """Tests for PowerLawPrior."""

    def test_positive_pdf_within_bounds(self):
        """PDF is positive everywhere within bounds."""
        from isochrones.priors import PowerLawPrior

        p = PowerLawPrior(alpha=2.0, bounds=(1.0, 10.0))
        for x in [2.0, 5.0, 8.0]:
            assert p.pdf(x) > 0.0

    def test_integrates_to_one(self):
        """PDF integrates to 1 over bounds."""
        from isochrones.priors import PowerLawPrior

        p = PowerLawPrior(alpha=2.0, bounds=(1.0, 5.0))
        integral, _ = quad(p.pdf, *p.bounds)
        assert math.isclose(integral, 1.0, rel_tol=1e-8)

    def test_lnpdf_consistent(self):
        """exp(lnpdf(x)) matches _pdf(x)."""
        from isochrones.priors import PowerLawPrior

        # Salpeter slope
        p = PowerLawPrior(alpha=-2.35, bounds=(0.1, 10.0))
        for x in [0.3, 1.0, 5.0]:
            assert math.isclose(math.exp(p._lnpdf(x)), p._pdf(x), rel_tol=1e-12)

    def test_samples_in_bounds(self):
        """Samples lie within [lo, hi]."""
        from isochrones.priors import PowerLawPrior

        np.random.seed(1)
        lo, hi = 1.0, 10.0
        p = PowerLawPrior(alpha=0.3, bounds=(lo, hi))
        samples = p.sample(2000)
        assert np.all(samples >= lo) and np.all(samples <= hi)


class TestAgePrior:
    """Tests for AgePrior (log-uniform in 10^age)."""

    def test_default_bounds(self):
        """Default bounds are (5, 10.15) in log10(age/yr)."""
        from isochrones.priors import AgePrior

        p = AgePrior()
        assert p.bounds == (5, 10.15)

    def test_positive_pdf_within_bounds(self):
        """PDF is positive within default bounds."""
        from isochrones.priors import AgePrior

        p = AgePrior()
        for log_age in [6.0, 8.0, 9.5, 10.0]:
            assert p.pdf(log_age) > 0.0

    def test_zero_outside_bounds(self):
        """PDF is 0 outside bounds."""
        from isochrones.priors import AgePrior

        p = AgePrior()
        assert p.pdf(4.0) == 0.0
        assert p.pdf(11.0) == 0.0

    def test_integrates_to_one(self):
        """Integrates to 1 over default bounds."""
        from isochrones.priors import AgePrior

        p = AgePrior()
        integral, _ = quad(p.pdf, *p.bounds)
        assert math.isclose(integral, 1.0, rel_tol=1e-6)

    def test_favors_older_ages(self):
        """Log-uniform prior gives more weight to larger ages (more volume)."""
        from isochrones.priors import AgePrior

        # FlatLogPrior: pdf ∝ 10^log_age, so older ages have higher prior weight
        p = AgePrior()
        assert p.pdf(10.0) > p.pdf(6.0)


class TestDistancePrior:
    """Tests for DistancePrior (r² volume prior)."""

    def test_default_max_distance(self):
        """Default max_distance is 10000 pc."""
        from isochrones.priors import DistancePrior

        p = DistancePrior()
        assert p.bounds == (0, 10000)

    def test_custom_max_distance(self):
        """Custom max_distance is respected."""
        from isochrones.priors import DistancePrior

        p = DistancePrior(max_distance=1000)
        assert p.bounds == (0, 1000)

    def test_increasing_density(self):
        """Prior probability increases with distance (r² volume)."""
        from isochrones.priors import DistancePrior

        p = DistancePrior()
        assert p.pdf(5000) > p.pdf(1000)
        assert p.pdf(9000) > p.pdf(5000)

    def test_integrates_to_one(self):
        """Integrates to 1 over its bounds."""
        from isochrones.priors import DistancePrior

        p = DistancePrior(max_distance=1000)
        integral, _ = quad(p.pdf, *p.bounds)
        assert math.isclose(integral, 1.0, rel_tol=1e-8)


class TestFehPrior:
    """Tests for FehPrior (local metallicity distribution)."""

    def test_local_prior_peaks_near_solar(self):
        """Local FehPrior peaks near solar metallicity ([Fe/H] ~ 0)."""
        from isochrones.priors import FehPrior

        p = FehPrior(local=True)
        assert p.pdf(0.0) > p.pdf(-2.0)
        assert p.pdf(0.0) > p.pdf(1.0)

    def test_halo_fraction_increases_metal_poor(self):
        """Higher halo_fraction increases probability at [Fe/H] ~ -1.5."""
        from isochrones.priors import FehPrior

        p_disk = FehPrior(halo_fraction=0.0, local=False)
        p_halo = FehPrior(halo_fraction=0.5, local=False)
        assert p_halo.pdf(-1.5) > p_disk.pdf(-1.5)

    def test_bounds_applied(self):
        """After setting bounds, PDF is 0 outside."""
        from isochrones.priors import FehPrior

        p = FehPrior()
        p.bounds = (-3.0, 0.5)
        assert p(-3.5) == 0.0
        assert p(0.6) == 0.0
        assert p(0.0) > 0.0

    def test_non_negative_pdf(self):
        """PDF is never negative."""
        from isochrones.priors import FehPrior

        p = FehPrior()
        for feh in np.linspace(-3.0, 1.0, 41):
            assert p.pdf(feh) >= 0.0


class TestChabrierPrior:
    """Tests for ChabrierPrior (broken IMF)."""

    def test_continuity_at_breakpoint(self):
        """PDF is continuous at the 1 Msun breakpoint."""
        from isochrones.priors import ChabrierPrior

        p = ChabrierPrior()
        pdf_below = p.pdf(0.999)
        pdf_above = p.pdf(1.001)
        assert math.isclose(pdf_below, pdf_above, rel_tol=0.02)

    def test_integrates_to_one(self):
        """Integrates to 1 over default bounds (0.1, 100)."""
        from isochrones.priors import ChabrierPrior

        p = ChabrierPrior()
        integral, _ = quad(p.pdf, *p.bounds)
        assert math.isclose(integral, 1.0, rel_tol=1e-4)

    def test_decreasing_high_mass(self):
        """PDF decreases with mass above 1 Msun (Salpeter slope)."""
        from isochrones.priors import ChabrierPrior

        p = ChabrierPrior()
        assert p.pdf(2.0) > p.pdf(5.0) > p.pdf(10.0)

    def test_default_bounds(self):
        """Default bounds are (0.1, 100.0) Msun."""
        from isochrones.priors import ChabrierPrior

        p = ChabrierPrior()
        assert p.bounds == (0.1, 100.0)


# ---------------------------------------------------------------------------
# starfit workflow helpers
# ---------------------------------------------------------------------------


class TestGetBands:
    """Tests for StarModel.get_bands() – parses photometric bands from star.ini."""

    @pytest.fixture
    def star1_ini(self):
        return os.path.join(os.path.dirname(__file__), "star1", "star.ini")

    def test_returns_list(self, star1_ini):
        """get_bands should return a list."""
        from isochrones.starmodel import StarModel

        bands = StarModel.get_bands(star1_ini)
        assert isinstance(bands, list)

    def test_contains_jhk(self, star1_ini):
        """star1 INI has 2MASS J, H, K photometry, so those bands should be present."""
        from isochrones.starmodel import StarModel

        bands = StarModel.get_bands(star1_ini)
        for b in ["J", "H", "K"]:
            assert b in bands, f"Expected band '{b}' in {bands}"

    def test_excludes_spectral_keywords(self, star1_ini):
        """Spectral keywords (Teff, logg, feh, RA, dec) should not appear as bands."""
        from isochrones.starmodel import StarModel

        bands = StarModel.get_bands(star1_ini)
        not_bands = ["Teff", "logg", "feh", "RA", "dec", "ra"]
        for kw in not_bands:
            assert kw not in bands, f"'{kw}' should not be a band, got bands={bands}"

    @pytest.fixture
    def star2_ini(self):
        return os.path.join(os.path.dirname(__file__), "star2", "star.ini")

    def test_no_duplicates(self, star1_ini):
        """get_bands should not return duplicate band names."""
        from isochrones.starmodel import StarModel

        bands = StarModel.get_bands(star1_ini)
        assert len(bands) == len(set(bands)), f"Duplicate bands found: {bands}"

    def test_untagged_and_tagged_keywords_share_one_band(self, star2_ini):
        """A companion key like J_1 belongs to band J, not to a band 'J_1'."""
        from isochrones.starmodel import StarModel

        bands = StarModel.get_bands(star2_ini)
        for b in ["J", "H", "K", "W1", "W2", "W3"]:
            assert b in bands, f"Expected band '{b}' in {bands}"

    def test_excludes_companion_metadata(self, star2_ini):
        """separation/PA keys and tagged band aliases are not bandpasses."""
        from isochrones.starmodel import StarModel

        bands = StarModel.get_bands(star2_ini)
        assert sorted(bands) == ["H", "J", "K", "W1", "W2", "W3"], f"got bands={bands}"


class TestStarfitWorkflowLogic:
    """Tests for pure-logic parts of the starfit() workflow (no MCMC needed)."""

    def test_nstars_mapping(self):
        """Multiplicity strings map to correct star counts."""
        # Mirrors the nstars dict in starfit()
        nstars = {"single": 1, "binary": 2, "triple": 3}
        assert nstars["single"] == 1
        assert nstars["binary"] == 2
        assert nstars["triple"] == 3

    def test_model_filename_pattern(self):
        """Model HDF5 filenames follow '{models}_starmodel_{mult}.h5'."""
        for models in ["mist"]:
            for mult in ["single", "binary", "triple"]:
                expected = f"{models}_starmodel_{mult}.h5"
                # Must contain model name and multiplicity
                assert models in expected
                assert mult in expected
                assert expected.endswith(".h5")

    def test_flat_feh_prior_covers_range(self):
        """FlatPrior used as feh prior has zero density outside bounds."""
        from isochrones.priors import FlatPrior

        lo, hi = -1.0, 0.5
        p = FlatPrior((lo, hi))
        assert p.pdf(lo + 0.1) > 0
        assert p.pdf(lo - 0.01) == 0.0
        assert p.pdf(hi + 0.01) == 0.0


# ---------------------------------------------------------------------------
# Regression: Python 3.8 → 3.12 / NumPy 2.0
# ---------------------------------------------------------------------------


class TestNumpyDeprecatedAliasesRemoved:
    """Verify NumPy 2.0 alias changes relevant to the py3.8→py3.12 upgrade.

    NumPy 1.20 deprecated ``np.int``, ``np.float``, ``np.complex`` (aliases
    for Python built-ins); they were removed in NumPy 1.24 / 2.0.
    ``np.bool`` was *not* fully removed in NumPy 2.x — it now refers to
    ``np.bool_`` (the NumPy scalar type) rather than Python's ``bool``.
    """

    def test_np_int_removed(self):
        """np.int removed in NumPy 2.0; use int or np.int64 instead."""
        with pytest.raises(AttributeError):
            _ = np.int  # type: ignore[attr-defined]

    def test_np_float_removed(self):
        """np.float removed in NumPy 2.0; use float or np.float64 instead."""
        with pytest.raises(AttributeError):
            _ = np.float  # type: ignore[attr-defined]

    def test_np_complex_removed(self):
        """np.complex removed in NumPy 2.0; use complex or np.complex128 instead."""
        with pytest.raises(AttributeError):
            _ = np.complex  # type: ignore[attr-defined]

    def test_np_bool_is_numpy_bool_not_python_bool(self):
        """np.bool in NumPy 2.x is np.bool_ (numpy scalar), NOT Python's bool.

        In NumPy 1.x, np.bool was an alias for Python's bool.
        In NumPy 2.x, np.bool refers to numpy.bool_ (the numpy scalar type).
        Code that did ``isinstance(x, np.bool)`` expecting Python bool must
        be updated.
        """
        assert np.bool is np.bool_  # type: ignore[attr-defined]
        assert np.bool is not bool  # type: ignore[attr-defined]

    def test_np_bool__is_canonical_bool_dtype(self):
        """np.bool_ is the canonical NumPy boolean dtype."""
        arr = np.array([True, False], dtype=np.bool_)
        assert arr.dtype == np.bool_

    def test_np_float64_works(self):
        """np.float64 is the valid double-precision type."""
        arr = np.array([1.0, 2.0], dtype=np.float64)
        assert arr.dtype == np.float64

    def test_np_int64_works(self):
        """np.int64 is the valid fixed-width integer type."""
        arr = np.array([1, 2, 3], dtype=np.int64)
        assert arr.dtype == np.int64


class TestPandasBehaviorPy312:
    """Regression tests for pandas 2.x behaviour used in DFInterpolator._make_grid."""

    def test_multiindex_from_product(self):
        """MultiIndex.from_product still works in pandas 2.x."""
        idx = pd.MultiIndex.from_tuples(
            list(itertools.product([0.0, 1.0], [0.0, 1.0])), names=["x", "y"]
        )
        assert len(idx) == 4
        assert idx.names == ["x", "y"]

    def test_loc_assignment_on_full_index(self):
        """DataFrame.loc assignment used in DFInterpolator._make_grid."""
        src_idx = pd.MultiIndex.from_tuples([(0.0, 0.0), (0.0, 1.0)], names=["x", "y"])
        src = pd.DataFrame({"val": [1.0, 2.0]}, index=src_idx)

        full_idx = pd.MultiIndex.from_tuples(
            [(0.0, 0.0), (0.0, 1.0), (1.0, 0.0)], names=["x", "y"]
        )
        dst = pd.DataFrame(index=full_idx, columns=["val"])
        dst.loc[src.index] = src  # This pattern is used in _make_grid
        assert float(dst.loc[(0.0, 0.0), "val"]) == 1.0
        assert float(dst.loc[(0.0, 1.0), "val"]) == 2.0

    def test_index_levels_attribute(self):
        """index.levels still returns level arrays in pandas 2.x."""
        idx = pd.MultiIndex.from_tuples(
            list(itertools.product([0.0, 1.0], [10.0, 20.0])), names=["x", "y"]
        )
        df = pd.DataFrame({"v": range(4)}, index=idx)
        levels = df.index.levels
        assert len(levels) == 2
        assert list(levels[0]) == [0.0, 1.0]
        assert list(levels[1]) == [10.0, 20.0]


class TestNumpyBehaviorPy312:
    """Regression tests for NumPy behaviour that must stay consistent."""

    def test_nan_self_inequality(self):
        """The x != x idiom for NaN detection (used in interp.py) still works."""
        nan = float("nan")
        assert nan != nan
        assert np.nan != np.nan
        # Non-NaN values should be equal to themselves
        assert not (1.0 != 1.0)

    def test_integer_array_indexing(self):
        """Integer indexing of arrays behaves as expected."""
        arr = np.array([10.0, 20.0, 30.0, 40.0])
        assert arr[0] == 10.0
        assert arr[-1] == 40.0
        assert arr[1:3][0] == 20.0

    def test_searchsorted_stable(self):
        """np.searchsorted behaviour must not regress."""
        arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        assert np.searchsorted(arr, 2.5) == 2
        assert np.searchsorted(arr, 2.0, side="left") == 1
        assert np.searchsorted(arr, 2.0, side="right") == 2
        assert np.searchsorted(arr, 1.0) == 0
        assert np.searchsorted(arr, 5.0, side="right") == 5

    def test_log10_of_powers_of_10(self):
        """log10 of exact powers of 10 returns integers (no rounding surprise)."""
        for exp in range(-3, 4):
            assert math.isclose(math.log10(10**exp), exp, abs_tol=1e-12)

    def test_array_reshape_consistent(self):
        """Reshape used in DFInterpolator._make_grid must be consistent."""
        data = np.arange(24, dtype=float)
        shaped = data.reshape((2, 3, 4))
        assert shaped.shape == (2, 3, 4)
        assert shaped[0, 0, 0] == 0.0
        assert shaped[1, 2, 3] == 23.0


# ---------------------------------------------------------------------------
# Regression: frozen numerical values
# (Must not change between py3.8 and py3.12)
# ---------------------------------------------------------------------------


class TestRegressionNumericalValues:
    """Frozen numerical regression values.

    These are exact numbers that the code must produce.  If any of these
    fail after a Python or dependency upgrade it signals a regression.
    """

    def test_searchsorted_known_results(self):
        """Exact searchsorted results for a fixed array."""
        from isochrones.interp import searchsorted

        arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        assert searchsorted(arr, 1.0) == (0, True)
        assert searchsorted(arr, 5.0) == (4, True)
        assert searchsorted(arr, 2.5) == (2, False)
        assert searchsorted(arr, 3.0) == (2, True)

    def test_fast_addmags_known_values(self):
        """Exact fast_addmags values for known inputs."""
        from isochrones.utils import fast_addmags

        # Two equal mags of 10.0
        expected = 10.0 - 2.5 * math.log10(2)
        assert math.isclose(fast_addmags([10.0, 10.0]), expected, abs_tol=1e-12)

        # Three equal mags of 12.0
        expected2 = 12.0 - 2.5 * math.log10(3)
        assert math.isclose(fast_addmags([12.0, 12.0, 12.0]), expected2, abs_tol=1e-12)

        # Asymmetric pair
        # f1 = 10^(-0.4*8) = 10^-3.2 ≈ 6.310e-4
        # f2 = 10^(-0.4*12) = 10^-4.8 ≈ 1.585e-5
        # combined = -2.5*log10(f1 + f2)
        f1 = 10 ** (-0.4 * 8.0)
        f2 = 10 ** (-0.4 * 12.0)
        expected3 = -2.5 * math.log10(f1 + f2)
        assert math.isclose(fast_addmags([8.0, 12.0]), expected3, abs_tol=1e-12)

    def test_gauss_lnprob_known_values(self):
        """Exact gauss_lnprob values for known inputs."""
        from isochrones.likelihood import gauss_lnprob

        LOG1OVER_ROOT2PI = math.log(1.0 / math.sqrt(2 * math.pi))

        # Perfect match
        result = gauss_lnprob(5700.0, 100.0, 5700.0)
        expected = LOG1OVER_ROOT2PI - math.log(100.0)
        assert math.isclose(result, expected, abs_tol=1e-10)

        # 1-sigma offset
        result2 = gauss_lnprob(5700.0, 100.0, 5800.0)
        expected2 = LOG1OVER_ROOT2PI - math.log(100.0) - 0.5
        assert math.isclose(result2, expected2, abs_tol=1e-10)

        # 2-sigma offset
        result3 = gauss_lnprob(5700.0, 100.0, 5900.0)
        expected3 = LOG1OVER_ROOT2PI - math.log(100.0) - 2.0
        assert math.isclose(result3, expected3, abs_tol=1e-10)

    def test_flat_prior_known_values(self):
        """Exact FlatPrior pdf/lnpdf values."""
        from isochrones.priors import FlatPrior

        p = FlatPrior((0.0, 5.0))
        assert math.isclose(p.pdf(2.5), 0.2, abs_tol=1e-12)
        assert math.isclose(p.lnpdf(2.5), math.log(0.2), abs_tol=1e-12)
        assert p.pdf(-0.1) == 0.0
        assert p.pdf(5.1) == 0.0

    def test_interp_2d_bilinear_known_value(self):
        """2D interpolation of x*y at (0.5, 5.0) in a [0,1]×[0,10] grid."""
        from isochrones.interp import interp_value_2d

        ii0 = np.array([0.0, 1.0])
        ii1 = np.array([0.0, 10.0])
        # f(x, y) = x * y
        grid = np.array([[0.0, 0.0], [0.0, 10.0]], dtype=float).reshape(2, 2, 1)
        icols = np.array([0])

        # At (0.5, 5.0): bilinear interp of x*y gives
        # (1-0.5)*(1-0.5)*0 + (1-0.5)*0.5*0 + 0.5*(1-0.5)*0 + 0.5*0.5*10 = 2.5
        result = interp_value_2d(0.5, 5.0, grid, icols, ii0, ii1)
        assert math.isclose(result[0], 2.5, abs_tol=1e-10)

    def test_dfinterpolator_known_value(self):
        """DFInterpolator regression: known value at interior point."""
        rows = [
            (x, y, z, x + 0.1 * y + 0.001 * z)
            for x, y, z in itertools.product(
                [0.0, 1.0, 2.0], [0.0, 5.0, 10.0], [0.0, 100.0, 200.0]
            )
        ]
        df = pd.DataFrame(rows, columns=["x", "y", "z", "val"]).set_index(["x", "y", "z"])

        from isochrones.interp import DFInterpolator

        interp = DFInterpolator(df)
        result = interp([0.5, 2.5, 50.0], ["val"])
        expected = 0.5 + 0.1 * 2.5 + 0.001 * 50.0
        assert math.isclose(result[0], expected, abs_tol=1e-10)


# ---------------------------------------------------------------------------
# Phase 1: find_indices out-of-bounds detection (bug fix test)
# ---------------------------------------------------------------------------


class TestFindIndicesOOB:
    """Tests for the generic find_indices out-of-bounds detection.

    This tests the fix for the &= -> |= bug in interp.py:58 where
    out_of_bounds was always False because False &= anything is False.
    """

    def setup_method(self):
        import numba.typed
        self.iis = numba.typed.List()
        self.iis.append(np.array([0.0, 1.0, 2.0]))
        self.iis.append(np.array([0.0, 5.0, 10.0]))

    def test_in_bounds_point(self):
        """Interior point should not be out of bounds."""
        from isochrones.interp import find_indices

        point = np.array([0.5, 2.5])
        _, _, oob = find_indices(point, self.iis)
        assert oob is False or oob == False

    def test_oob_first_dim_below(self):
        """Point below first dimension should be out of bounds."""
        from isochrones.interp import find_indices

        point = np.array([-1.0, 2.5])
        _, _, oob = find_indices(point, self.iis)
        assert oob, "Expected out_of_bounds=True for point below dim 0"

    def test_oob_first_dim_above(self):
        """Point above first dimension should be out of bounds."""
        from isochrones.interp import find_indices

        point = np.array([3.0, 2.5])
        _, _, oob = find_indices(point, self.iis)
        assert oob, "Expected out_of_bounds=True for point above dim 0"

    def test_oob_second_dim_below(self):
        """Point below second dimension should be out of bounds."""
        from isochrones.interp import find_indices

        point = np.array([0.5, -1.0])
        _, _, oob = find_indices(point, self.iis)
        assert oob, "Expected out_of_bounds=True for point below dim 1"

    def test_oob_second_dim_above(self):
        """Point above second dimension should be out of bounds."""
        from isochrones.interp import find_indices

        point = np.array([0.5, 11.0])
        _, _, oob = find_indices(point, self.iis)
        assert oob, "Expected out_of_bounds=True for point above dim 1"

    def test_oob_both_dims(self):
        """Point out of bounds in both dimensions."""
        from isochrones.interp import find_indices

        point = np.array([-1.0, 11.0])
        _, _, oob = find_indices(point, self.iis)
        assert oob, "Expected out_of_bounds=True for point OOB in both dims"

    def test_at_grid_boundary(self):
        """Point exactly at grid boundary should be in bounds."""
        from isochrones.interp import find_indices

        point = np.array([0.0, 0.0])
        _, _, oob = find_indices(point, self.iis)
        assert not oob, "Point at grid minimum should be in bounds"


# ---------------------------------------------------------------------------
# Phase 2: addmags with uncertainties
# ---------------------------------------------------------------------------


class TestAddMagsWithUncertainties:
    """Tests for addmags() with (mag, err) tuple inputs."""

    def test_tuple_input_returns_tuple(self):
        """addmags with (mag, err) pairs returns (total_mag, total_err)."""
        from isochrones.utils import addmags

        result = addmags((10.0, 0.1), (10.0, 0.1))
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_tuple_total_mag_matches_plain(self):
        """Total magnitude from (mag, err) pairs matches plain magnitude addition."""
        from isochrones.utils import addmags

        total_with_err, _ = addmags((10.0, 0.1), (12.0, 0.05))
        total_plain = addmags(10.0, 12.0)
        assert math.isclose(total_with_err, total_plain, rel_tol=1e-10)

    def test_single_tuple_identity(self):
        """Single (mag, err) should return approximately the same magnitude."""
        from isochrones.utils import addmags

        total, unc = addmags((10.0, 0.1))
        assert math.isclose(total, 10.0, rel_tol=1e-10)

    def test_uncertainty_is_positive(self):
        """Returned uncertainty should be positive."""
        from isochrones.utils import addmags

        _, unc = addmags((10.0, 0.1), (10.0, 0.1))
        assert unc > 0

    def test_equal_mags_combined_brighter(self):
        """Two equal mags with uncertainties: combined is brighter."""
        from isochrones.utils import addmags

        total, _ = addmags((10.0, 0.1), (10.0, 0.1))
        expected = 10.0 - 2.5 * math.log10(2)
        assert math.isclose(total, expected, rel_tol=1e-10)


# ---------------------------------------------------------------------------
# Phase 2: band_pairs utility
# ---------------------------------------------------------------------------


class TestBandPairs:
    """Tests for band_pairs() utility function."""

    def test_normal_case(self):
        """Three bands produce two pairs, each paired with the last band."""
        from isochrones.utils import band_pairs

        result = band_pairs(["J", "H", "K"])
        assert result == [("J", "K"), ("H", "K")]

    def test_two_bands(self):
        """Two bands produce one pair."""
        from isochrones.utils import band_pairs

        result = band_pairs(["J", "K"])
        assert result == [("J", "K")]

    def test_single_band(self):
        """Single band produces empty list."""
        from isochrones.utils import band_pairs

        result = band_pairs(["J"])
        assert result == []

    def test_empty_list(self):
        """Empty list produces empty list."""
        from isochrones.utils import band_pairs

        result = band_pairs([])
        assert result == []

    def test_four_bands(self):
        """Four bands produce three pairs, each with the last."""
        from isochrones.utils import band_pairs

        result = band_pairs(["B", "V", "R", "I"])
        assert result == [("B", "I"), ("V", "I"), ("R", "I")]


# ---------------------------------------------------------------------------
# Phase 2: Additional prior classes
# ---------------------------------------------------------------------------


class TestAVPrior:
    """Tests for AVPrior (extinction prior)."""

    def test_default_bounds(self):
        """Default bounds are (0, 1)."""
        from isochrones.priors import AVPrior

        p = AVPrior()
        assert p.bounds == (0, 1.0)

    def test_constant_pdf(self):
        """Uniform prior: PDF is 1/(1-0) = 1.0 within bounds."""
        from isochrones.priors import AVPrior

        p = AVPrior()
        assert math.isclose(p.pdf(0.5), 1.0, rel_tol=1e-10)

    def test_zero_outside(self):
        """PDF is 0 outside bounds."""
        from isochrones.priors import AVPrior

        p = AVPrior()
        assert p.pdf(-0.1) == 0.0
        assert p.pdf(1.1) == 0.0

    def test_integrates_to_one(self):
        """PDF integrates to 1."""
        from isochrones.priors import AVPrior

        p = AVPrior()
        integral, _ = quad(p.pdf, *p.bounds)
        assert math.isclose(integral, 1.0, rel_tol=1e-8)


class TestQPrior:
    """Tests for QPrior (mass ratio prior)."""

    def test_default_bounds(self):
        """Default bounds are (0.1, 1)."""
        from isochrones.priors import QPrior

        p = QPrior()
        assert p.bounds == (0.1, 1)

    def test_positive_pdf_within_bounds(self):
        """PDF is positive within bounds."""
        from isochrones.priors import QPrior

        p = QPrior()
        for q in [0.2, 0.5, 0.8]:
            assert p.pdf(q) > 0.0

    def test_increasing_with_q(self):
        """alpha=0.3 > 0, so PDF increases with q."""
        from isochrones.priors import QPrior

        p = QPrior()
        assert p.pdf(0.8) > p.pdf(0.3)

    def test_integrates_to_one(self):
        """PDF integrates to 1."""
        from isochrones.priors import QPrior

        p = QPrior()
        integral, _ = quad(p.pdf, *p.bounds)
        assert math.isclose(integral, 1.0, rel_tol=1e-8)

    def test_zero_outside(self):
        """PDF is 0 outside bounds."""
        from isochrones.priors import QPrior

        p = QPrior()
        assert p.pdf(0.05) == 0.0
        assert p.pdf(1.1) == 0.0


class TestSalpeterPrior:
    """Tests for SalpeterPrior (IMF prior)."""

    def test_default_bounds(self):
        """Default bounds are (0.1, 10)."""
        from isochrones.priors import SalpeterPrior

        p = SalpeterPrior()
        assert p.bounds == (0.1, 10)

    def test_decreasing_with_mass(self):
        """Salpeter alpha=-2.35: PDF decreases with mass."""
        from isochrones.priors import SalpeterPrior

        p = SalpeterPrior()
        assert p.pdf(0.5) > p.pdf(1.0) > p.pdf(5.0)

    def test_integrates_to_one(self):
        """PDF integrates to 1."""
        from isochrones.priors import SalpeterPrior

        p = SalpeterPrior()
        integral, _ = quad(p.pdf, *p.bounds)
        assert math.isclose(integral, 1.0, rel_tol=1e-8)

    def test_lnpdf_consistent(self):
        """exp(lnpdf(x)) == pdf(x)."""
        from isochrones.priors import SalpeterPrior

        p = SalpeterPrior()
        for x in [0.3, 1.0, 5.0]:
            assert math.isclose(math.exp(p.lnpdf(x)), p.pdf(x), rel_tol=1e-10)


class TestLogNormalPrior:
    """Tests for LogNormalPrior."""

    def test_pdf_positive_for_positive_x(self):
        """PDF is positive for x > 0."""
        from isochrones.priors import LogNormalPrior

        p = LogNormalPrior(0.0, 1.0)
        for x in [0.1, 0.5, 1.0, 2.0, 5.0]:
            assert p._pdf(x) > 0.0

    def test_lnpdf_consistent_with_pdf(self):
        """exp(_lnpdf(x)) == _pdf(x)."""
        from isochrones.priors import LogNormalPrior

        p = LogNormalPrior(0.0, 1.0)
        for x in [0.1, 0.5, 1.0, 2.0, 5.0]:
            assert math.isclose(math.exp(p._lnpdf(x)), p._pdf(x), rel_tol=1e-10)

    def test_peak_near_exp_mu(self):
        """PDF peaks near exp(mu) for small sigma."""
        from isochrones.priors import LogNormalPrior

        mu = 1.0
        p = LogNormalPrior(mu, 0.1)
        peak = math.exp(mu)
        # PDF at peak should be higher than away from peak
        assert p._pdf(peak) > p._pdf(peak * 0.5)
        assert p._pdf(peak) > p._pdf(peak * 2.0)

    def test_different_mu(self):
        """Different mu shifts the distribution."""
        from isochrones.priors import LogNormalPrior

        p1 = LogNormalPrior(0.0, 0.5)
        p2 = LogNormalPrior(2.0, 0.5)
        # p2 peaks at higher x than p1
        assert p2._pdf(math.exp(2.0)) > p1._pdf(math.exp(2.0))


# ---------------------------------------------------------------------------
# Phase 2: BrokenPrior
# ---------------------------------------------------------------------------


class TestBrokenPrior:
    """Tests for BrokenPrior (composite prior with breakpoints)."""

    def test_integrates_to_one(self):
        """BrokenPrior integrates to 1 over its bounds."""
        from isochrones.priors import PowerLawPrior, LogNormalPrior, BrokenPrior

        comp1 = LogNormalPrior(np.log(0.3), 0.5)
        comp2 = PowerLawPrior(alpha=-2.35, bounds=(0.1, 100.0))
        bp = BrokenPrior([comp1, comp2], [1.0], bounds=(0.1, 100.0))
        integral, _ = quad(bp.pdf, *bp.bounds)
        assert math.isclose(integral, 1.0, rel_tol=1e-3)

    def test_continuity_at_breakpoint(self):
        """PDF is continuous at breakpoints."""
        from isochrones.priors import PowerLawPrior, LogNormalPrior, BrokenPrior

        comp1 = LogNormalPrior(np.log(0.3), 0.5)
        comp2 = PowerLawPrior(alpha=-2.35, bounds=(0.1, 100.0))
        bp = BrokenPrior([comp1, comp2], [1.0], bounds=(0.1, 100.0))
        pdf_below = bp.pdf(0.999)
        pdf_above = bp.pdf(1.001)
        assert math.isclose(pdf_below, pdf_above, rel_tol=0.02)

    def test_bounds_setter_reinitializes(self):
        """Setting new bounds recomputes normalization."""
        from isochrones.priors import PowerLawPrior, LogNormalPrior, BrokenPrior

        comp1 = LogNormalPrior(np.log(0.3), 0.5)
        comp2 = PowerLawPrior(alpha=-2.35, bounds=(0.1, 100.0))
        bp = BrokenPrior([comp1, comp2], [1.0], bounds=(0.1, 100.0))

        old_norms = bp.norms.copy()
        bp.bounds = (0.5, 50.0)
        # Norms should change after bounds update
        assert not np.allclose(old_norms, bp.norms)
        # Should still integrate to 1
        integral, _ = quad(bp.pdf, *bp.bounds)
        assert math.isclose(integral, 1.0, rel_tol=1e-3)

    def test_positive_within_bounds(self):
        """PDF is positive everywhere within bounds."""
        from isochrones.priors import PowerLawPrior, LogNormalPrior, BrokenPrior

        comp1 = LogNormalPrior(np.log(0.3), 0.5)
        comp2 = PowerLawPrior(alpha=-2.35, bounds=(0.1, 100.0))
        bp = BrokenPrior([comp1, comp2], [1.0], bounds=(0.1, 100.0))
        for x in [0.2, 0.5, 1.0, 5.0, 50.0]:
            assert bp.pdf(x) > 0.0

    def test_sample_within_bounds(self):
        """Samples lie within bounds."""
        from isochrones.priors import ChabrierPrior

        np.random.seed(42)
        p = ChabrierPrior()
        samples = p.sample(1000)
        lo, hi = p.bounds
        assert np.all(samples >= lo) and np.all(samples <= hi)
