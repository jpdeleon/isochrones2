"""Offline tests for StarCatalog (no model grids needed)."""

import pickle

import numpy as np
import pandas as pd
import pytest

from isochrones.catalog import StarCatalog


def _df(**extra):
    data = {
        "g_mag": [10.0, 11.0, 12.0],
        "g_mag_unc": [0.01, 0.02, 0.03],
        "r_mag": [9.5, 10.5, 11.5],
        "r_mag_unc": [0.01, 0.02, 0.03],
        "Teff": [5700.0, 5800.0, 5900.0],
        "Teff_unc": [50.0, 60.0, 70.0],
    }
    data.update(extra)
    return pd.DataFrame(data, index=["a", "b", "c"])


class TestConstruction:
    def test_bands_inferred_from_mag_columns(self):
        cat = StarCatalog(_df(), props=["Teff"])
        assert cat.bands == ("g", "r")
        assert cat.band_cols == ("g_mag", "r_mag")
        assert cat.props == ("Teff",)

    def test_explicit_bands_restrict_selection(self):
        cat = StarCatalog(_df(), bands=["r"])
        assert cat.bands == ("r",)
        assert cat.band_cols == ("r_mag",)

    def test_len_matches_rows(self):
        assert len(StarCatalog(_df())) == 3

    def test_missing_property_column_raises(self):
        with pytest.raises(ValueError, match="logg not in DataFrame"):
            StarCatalog(_df(), props=["logg"])

    def test_missing_uncertainty_column_raises(self):
        df = _df().drop(columns=["g_mag_unc"])
        with pytest.raises(ValueError, match="g_mag_unc"):
            StarCatalog(df)

    def test_no_uncs_skips_validation(self):
        df = _df().drop(columns=["g_mag_unc", "r_mag_unc"])
        assert len(StarCatalog(df, no_uncs=True)) == 3


class TestAccessors:
    def test_get_measurement_returns_value_and_uncertainty_arrays(self):
        vals, uncs = StarCatalog(_df()).get_measurement("g_mag")
        np.testing.assert_allclose(vals, [10.0, 11.0, 12.0])
        np.testing.assert_allclose(uncs, [0.01, 0.02, 0.03])

    def test_iter_bands_yields_band_with_measurements(self):
        out = {b: m for b, m in StarCatalog(_df()).iter_bands()}
        assert list(out) == ["g", "r"]
        np.testing.assert_allclose(out["r"][0], [9.5, 10.5, 11.5])

    def test_iter_props_yields_property_with_measurements(self):
        out = dict(StarCatalog(_df(), props=["Teff"]).iter_props())
        np.testing.assert_allclose(out["Teff"][1], [50.0, 60.0, 70.0])


class TestStateHandling:
    def test_df_setter_replaces_data_and_clears_caches(self):
        cat = StarCatalog(_df())
        cat._ds, cat._hr = object(), object()

        cat.df = _df().iloc[:2]

        assert len(cat) == 2
        assert cat._ds is None and cat._hr is None

    def test_set_prior_accumulates_settings(self):
        cat = StarCatalog(_df())
        cat.set_prior(mass="a")
        cat.set_prior(age="b", mass="c")
        assert cat._prior_settings == {"mass": "c", "age": "b"}

    def test_unpickling_drops_cached_plot(self):
        cat = StarCatalog(_df())
        cat._hr = object()
        # the cached holoviews layout is not part of the restored state
        restored = StarCatalog.__new__(StarCatalog)
        restored.__setstate__(dict(cat.__dict__))
        assert restored._hr is None

    def test_pickle_roundtrip_keeps_data(self):
        cat = StarCatalog(_df(), props=["Teff"])
        again = pickle.loads(pickle.dumps(cat))
        pd.testing.assert_frame_equal(again.df, cat.df)
        assert again.bands == cat.bands
