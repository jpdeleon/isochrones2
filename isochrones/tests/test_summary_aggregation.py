"""Offline tests for summary.get_quantiles filtering and get_summary_df."""

import numpy as np
import pandas as pd
import pytest

from isochrones import summary as sm


class _FakeModel:
    def __init__(self, seed):
        rng = np.random.default_rng(seed)
        self.derived_samples = pd.DataFrame(
            {
                "mass_0_0": rng.normal(1.0, 0.1, 500),
                "age_0": rng.normal(9.0, 0.3, 500),
                "Teff_0_0": rng.normal(5700.0, 80.0, 500),  # not requested
            }
        )


@pytest.fixture
def fake_load(monkeypatch):
    def load(cls, path):
        name = path.split("/")[-2]
        if name == "missing":
            raise FileNotFoundError(path)
        return _FakeModel(sum(map(ord, name)))

    monkeypatch.setattr(sm.BasicStarModel, "load_hdf", classmethod(load))


def test_only_requested_columns_are_summarised(fake_load, tmp_path):
    df = sm.get_quantiles("s1", rootdir=str(tmp_path), columns=["mass"])

    assert any(c.startswith("mass_0_0_") for c in df.columns)
    assert not any(c.startswith("age_0_") for c in df.columns)
    assert not any(c.startswith("Teff") for c in df.columns)


def test_custom_quantiles_and_alpha_shape_the_columns(fake_load, tmp_path):
    df = sm.get_quantiles("s1", rootdir=str(tmp_path), columns=["age"], qs=[0.5], alpha=0.68)

    assert "age_0_50" in df.columns
    assert "age_0_05" not in df.columns
    assert {"age_0_cci_lo_68", "age_0_mci_hi_68", "age_0_mode"} <= set(df.columns)


def test_missing_model_returns_empty_frame_by_default(fake_load, tmp_path):
    assert sm.get_quantiles("missing", rootdir=str(tmp_path)).empty


def test_missing_model_raises_when_requested(fake_load, tmp_path):
    with pytest.raises(FileNotFoundError):
        sm.get_quantiles("missing", rootdir=str(tmp_path), raise_exceptions=True)


def test_verbose_reports_load_failure(fake_load, tmp_path, capsys):
    sm.get_quantiles("missing", rootdir=str(tmp_path), verbose=True)
    assert "cannnot load starmodel" in capsys.readouterr().out


def test_summary_df_has_one_row_per_star(fake_load, tmp_path):
    df = sm.get_summary_df(["s1", "s2", "s3"], rootdir=str(tmp_path), columns=["age"])

    assert list(df.index) == ["s1", "s2", "s3"]
    assert "age_0_mode" in df.columns


def test_summary_df_tolerates_a_failed_star(fake_load, tmp_path):
    df = sm.get_summary_df(["s1", "missing"], rootdir=str(tmp_path), columns=["age"])
    assert list(df.index) == ["s1"]


def test_quantile_worker_forwards_keyword_arguments(fake_load, tmp_path):
    worker = sm.quantile_worker(rootdir=str(tmp_path), columns=["mass"])
    out = worker("s9")
    assert list(out.index) == ["s9"]
    assert "mass_0_0_50" in out.columns
