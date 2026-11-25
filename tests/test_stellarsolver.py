"""StellarSolver CLI flag/output contracts, verified against upstream cli/main.cpp."""

import subprocess
from pathlib import Path

import pytest
from astropy.io import fits
from astropy.wcs import WCS

from seestar_photometry import astrometry, frames
from conftest import write_cube, write_wcs_sidecar


@pytest.fixture
def frame(tmp_path, cube):
    return frames.load_frame(write_cube(tmp_path / "original image.fit", cube))


@pytest.fixture
def cli(monkeypatch, wcs):
    seen = []
    monkeypatch.setattr(astrometry, "_solver_executable", lambda *args: "stellarsolver-cli")

    def run(command, **kwargs):
        seen.append((command, kwargs))
        assert not WCS(fits.getheader(command[-1])).has_celestial
        assert not kwargs.get("shell", False)
        fits.PrimaryHDU(data=fits.getdata(command[-1]), header=wcs.to_header()).writeto(
            command[command.index("--save-fits") + 1])
        return subprocess.CompletedProcess(command, 0, "solved", "")

    monkeypatch.setattr(subprocess, "run", run)
    return seen


def test_reads_solved_fits_and_uses_real_stellarsolver_flags(frame, cli):
    original = frame.path.read_bytes()
    result = astrometry.solve(frame, solver="stellarsolver", solver_options={
        "index_dir": "/indexes with spaces", "config": "ignored.cfg",
        "profile": "4-SmallScaleSolving", "timeout": 23, "radius": 3})
    assert result.has_celestial and frame.ccd.wcs is result
    command, kwargs = cli[0]
    assert command[command.index("--index-files") + 1] == "/indexes with spaces"
    assert "--config" not in command and "--radius" not in command
    assert command[command.index("--degrees") + 1] == "3.0"
    assert kwargs["timeout"] == 23
    assert frame.path.read_bytes() == original
    assert not Path(command[-1]).exists()
    assert astrometry.has_wcs(frame.path)
    astrometry.solve(frame, solver="stellarsolver")
    assert len(cli) == 1
    astrometry.solve(frame, solver="stellarsolver", force=True)
    assert len(cli) == 2


@pytest.mark.parametrize("outcome", ["no-output", "bad-wcs", "rc", "timeout"])
def test_failed_forced_solve_preserves_previous_sidecar(frame, cli, monkeypatch, outcome):
    write_wcs_sidecar(frame.path)
    cache = astrometry.wcs_cache_path(frame.path)
    original = cache.read_bytes()

    def run(command, **kwargs):
        if outcome == "timeout":
            raise subprocess.TimeoutExpired(command, kwargs["timeout"])
        if outcome == "rc":
            return subprocess.CompletedProcess(command, 1, "", "no solution")
        if outcome == "bad-wcs":
            fits.PrimaryHDU().writeto(command[command.index("--save-fits") + 1])
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(subprocess, "run", run)
    with pytest.raises(RuntimeError):
        astrometry.solve(frame, solver="stellarsolver", force=True)
    assert cache.read_bytes() == original


def test_config_forwarded_and_source_list_entry_point_supported(frame, cli):
    result = astrometry.solve_from_sources(frame, [], [], solver="stellarsolver",
                                          solver_options={"config": "indexes.cfg"})
    assert result.has_celestial
    command = cli[0][0]
    assert Path(command[command.index("--config") + 1]).is_absolute()
    assert "--index-files" not in command
