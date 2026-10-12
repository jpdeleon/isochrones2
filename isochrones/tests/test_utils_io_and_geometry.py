"""Offline tests for utils.download_file, distance and addmags error handling."""

import math
from unittest import mock

import numpy as np
import pytest

from isochrones import utils
from isochrones.utils import addmags, distance, fast_addmags


class _FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def iter_content(self, chunk_size=1024):
        for i in range(0, len(self._payload), chunk_size):
            yield self._payload[i : i + chunk_size]
        yield b""  # keep-alive chunks are skipped


class TestDownloadFile:
    def test_writes_payload_to_explicit_path(self, tmp_path):
        target = tmp_path / "grid.tar"
        payload = b"x" * 3000

        with mock.patch.object(utils.requests, "get", return_value=_FakeResponse(payload)) as get:
            out = utils.download_file("http://example.invalid/grid.tar", str(target))

        assert out == str(target)
        assert target.read_bytes() == payload
        get.assert_called_once()

    def test_existing_file_is_not_redownloaded(self, tmp_path):
        target = tmp_path / "grid.tar"
        target.write_bytes(b"old")

        with mock.patch.object(utils.requests, "get") as get:
            out = utils.download_file("http://example.invalid/grid.tar", str(target))

        assert out == str(target)
        assert target.read_bytes() == b"old"
        get.assert_not_called()

    def test_clobber_overwrites_existing_file(self, tmp_path):
        target = tmp_path / "grid.tar"
        target.write_bytes(b"old")

        with mock.patch.object(utils.requests, "get", return_value=_FakeResponse(b"new")):
            utils.download_file("http://example.invalid/grid.tar", str(target), clobber=True)

        assert target.read_bytes() == b"new"

    def test_default_path_is_url_basename_in_data_dir(self, tmp_path, monkeypatch):
        monkeypatch.setattr(utils, "ISOCHRONES", str(tmp_path))

        with mock.patch.object(utils.requests, "get", return_value=_FakeResponse(b"abc")):
            out = utils.download_file("http://example.invalid/some/dir/file.bin")

        assert out == str(tmp_path / "file.bin")
        assert (tmp_path / "file.bin").read_bytes() == b"abc"


class TestDistance:
    def test_identical_positions_have_zero_separation(self):
        assert distance((0.5, 30.0), (0.5, 30.0)) == pytest.approx(0.0, abs=1e-15)

    def test_same_pa_separation_is_radial_difference(self):
        assert distance((0.2, 45.0), (0.5, 45.0)) == pytest.approx(0.3)

    def test_perpendicular_offsets(self):
        # PA 0 is +dec, PA 90 is +ra: legs of a right triangle.
        assert distance((1.0, 0.0), (1.0, 90.0)) == pytest.approx(math.sqrt(2.0))

    def test_opposite_sides_add(self):
        assert distance((0.3, 0.0), (0.4, 180.0)) == pytest.approx(0.7)

    def test_symmetric(self):
        a, b = (0.3, 20.0), (1.1, 200.0)
        assert distance(a, b) == pytest.approx(distance(b, a))


class TestAddmagsErrors:
    def test_equal_magnitudes_gain_2p5_log10_2(self):
        assert addmags(10.0, 10.0) == pytest.approx(10.0 - 2.5 * math.log10(2.0))

    def test_returns_scalar_without_errors_and_pair_with_errors(self):
        assert np.isscalar(addmags(10.0, 11.0))
        total, unc = addmags((10.0, 0.01), (11.0, 0.01))
        assert np.isfinite(total) and np.isfinite(unc) and unc > 0

    def test_total_magnitude_ignores_errors(self):
        plain = addmags(10.0, 11.0)
        with_err, _ = addmags((10.0, 0.05), (11.0, 0.05))
        assert with_err == pytest.approx(plain)

    def test_equal_components_error_shrinks_like_root_two(self):
        # Small-error limit: two equal, independent fluxes -> sigma/sqrt(2).
        _, unc = addmags((10.0, 0.001), (10.0, 0.001))
        assert unc == pytest.approx(0.001 / math.sqrt(2.0), rel=2e-2)

    def test_larger_input_error_gives_larger_total_error(self):
        _, small = addmags((10.0, 0.01), (11.0, 0.01))
        _, large = addmags((10.0, 0.05), (11.0, 0.05))
        assert large > small

    def test_fast_addmags_matches_addmags(self):
        mags = np.array([9.5, 10.2, 11.7])
        assert fast_addmags(mags) == pytest.approx(addmags(*mags))
