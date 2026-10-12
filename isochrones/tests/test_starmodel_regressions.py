"""Offline regression tests for small StarModel bugs (no grid files needed)."""

import math
from types import SimpleNamespace
from unittest import mock

import numpy as np

from isochrones import starmodel
from isochrones.likelihood import gauss_lnprob


def _fake_basic_model(kwargs, model_nu_max, model_delta_nu):
    ic = mock.MagicMock()
    ic.interp_value.return_value = (model_nu_max, model_delta_nu)
    return SimpleNamespace(
        N=1,
        spec_props=[(5700.0, 100.0)],
        bands=[],
        kwargs=kwargs,
        ic=ic,
    )


def test_delta_nu_uses_its_own_uncertainty():
    """The delta_nu term must be normalized by delta_nu_unc, not by delta_nu."""
    kwargs = {"nu_max": (1000.0, 50.0), "delta_nu": (60.0, 2.0)}
    model = _fake_basic_model(kwargs, model_nu_max=1010.0, model_delta_nu=62.0)

    with mock.patch.object(starmodel, "star_lnlike", return_value=0.0):
        got = starmodel.BasicStarModel.lnlike(model, [1.0, 9.0, 0.0, 100.0, 0.1])

    expected = gauss_lnprob(1000.0, 50.0, 1010.0) + gauss_lnprob(60.0, 2.0, 62.0)
    assert math.isclose(got, expected, rel_tol=1e-12)


def test_maxlike_starts_from_supplied_p0():
    """maxlike must honour p0 instead of a hardcoded starting point."""
    target = np.array([1.2, 8.0, -0.3, 120.0, 0.4])
    calls = []

    def lnpost(p):
        p = np.asarray(p, dtype=float)
        calls.append(p.copy())
        return -float(np.sum(((p - target) / np.array([1, 1, 1, 100, 1])) ** 2))

    model = SimpleNamespace(lnpost=lnpost)
    p0 = [1.1, 8.1, -0.2, 110.0, 0.3]

    fit = starmodel.StarModel.maxlike(model, p0, options={"xatol": 1e-6, "fatol": 1e-10})

    assert np.allclose(calls[0], p0)
    assert np.allclose(fit.x, target, atol=1e-3)
