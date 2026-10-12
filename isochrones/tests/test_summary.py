import numpy as np
import pandas as pd
import pytest

from isochrones.summary import (
    central_credible_interval,
    get_quantiles,
    hpd,
    mode_of_samples,
    summary_statistics,
)


def test_mode_known_peak():
    rng = np.random.default_rng(42)
    x = rng.normal(4.0, 1.0, 3000)
    m = mode_of_samples(x)
    assert abs(m - 4.0) < 0.15


def test_mode_drops_nan_and_degenerate():
    assert mode_of_samples(np.array([1.0, 1.0, 1.0, np.nan])) == 1.0
    assert np.isnan(mode_of_samples(np.array([np.nan, np.nan])))


def test_cci_matches_quantiles():
    rng = np.random.default_rng(7)
    x = rng.normal(0, 1, 5000)
    lo, hi = central_credible_interval(x)
    assert np.isclose(lo, np.quantile(x, 0.025), rtol=1e-12)
    assert np.isclose(hi, np.quantile(x, 0.975), rtol=1e-12)


def test_hpd_covers_alpha_fraction():
    rng = np.random.default_rng(11)
    x = rng.normal(0, 1, 5000)
    lo, hi = hpd(x)
    n_in = int(((x >= lo) & (x <= hi)).sum())
    assert 4749 <= n_in <= 4750


def test_hpd_symmetric_normal():
    rng = np.random.default_rng(3)
    x = rng.normal(0, 1, 50000)
    lo, hi = hpd(x)
    assert abs(lo + hi) < 0.15
    assert abs(lo + 1.96) < 0.15 and abs(hi - 1.96) < 0.15


def test_hpd_raises_when_alpha_too_large():
    with pytest.raises(ValueError):
        hpd(np.arange(5), alpha=0.9)


def test_summary_statistics_keys():
    rng = np.random.default_rng(5)
    x = rng.normal(0, 1, 2000)
    s = summary_statistics(x)
    expected = ["05", "16", "50", "84", "95", "mode", "cci_lo_95", "cci_hi_95", "mci_lo_95", "mci_hi_95"]
    for key in expected:
        assert key in s
    assert np.isclose(s["cci_lo_95"], np.quantile(x, 0.025), rtol=1e-12)
    assert np.isclose(s["cci_hi_95"], np.quantile(x, 0.975), rtol=1e-12)
    assert all(np.isfinite(list(s.values())))
    s68 = summary_statistics(x, alpha=0.68)
    assert "cci_lo_68" in s68 and "mci_hi_68" in s68


class FakeModel:
    def __init__(self):
        rng = np.random.default_rng(0)
        self.derived_samples = pd.DataFrame(
            {"age": rng.normal(9.0, 0.5, 1000), "mass": rng.normal(1.0, 0.1, 1000)}
        )


def test_get_quantiles_includes_mode_cci_mci(monkeypatch, tmp_path):
    from isochrones import summary as sm

    monkeypatch.setattr(sm.BasicStarModel, "load_hdf", classmethod(lambda cls, f: FakeModel()))
    df = sm.get_quantiles("star", rootdir=str(tmp_path))
    assert list(df.index) == ["star"]
    for c in ["age", "mass"]:
        for suffix in ["mode", "cci_lo_95", "cci_hi_95", "mci_lo_95", "mci_hi_95", "05", "50", "95"]:
            assert c + "_" + suffix in df.columns
    assert np.isfinite(df.loc["star", "age_mode"])
    assert np.isfinite(df.loc["star", "mass_mci_hi_95"])


def test_get_quantiles_missing_file_returns_empty(tmp_path):
    assert get_quantiles("nope", rootdir=str(tmp_path)).empty

def test_summary_statistics_ignores_nan_like_pandas_quantile():
    rng = np.random.default_rng(13)
    x = rng.normal(0, 1, 2000)
    with_nan = np.concatenate([x, [np.nan, np.nan, np.nan]])
    s = summary_statistics(with_nan)
    ref = summary_statistics(x)
    assert all(np.isfinite(list(s.values())))
    for q in [0.05, 0.16, 0.5, 0.84, 0.95]:
        key = "{:02.0f}".format(q * 100)
        assert np.isclose(s[key], pd.Series(with_nan).quantile(q), rtol=1e-12)
    assert s == ref


def test_summary_statistics_all_nan_returns_nan():
    s = summary_statistics(np.array([np.nan, np.nan]))
    assert set(s) == set(summary_statistics(np.arange(100.0)))
    assert all(np.isnan(v) for v in s.values())
