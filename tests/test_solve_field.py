"""CLI contracts use real FITS outputs; no installed solver or indexes are required."""

import subprocess
from pathlib import Path

import numpy as np
import pytest
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.wcs import WCS

from seestar_photometry import astrometry, frames
from conftest import write_cube, write_wcs_sidecar


@pytest.fixture
def frame(tmp_path, cube):
    return frames.load_frame(write_cube(tmp_path / "original image.fit", cube))


@pytest.fixture
def command_log(monkeypatch, wcs):
    seen = []
    monkeypatch.setattr(astrometry, "_solver_executable", lambda *args: "/solver dir/solve-field")

    def run(command, **kwargs):
        seen.append((command, kwargs))
        image = Path(command[-1])
        assert fits.getdata(image).ndim == 2
        assert not WCS(fits.getheader(image)).has_celestial
        assert not kwargs.get("shell", False)
        fits.PrimaryHDU(header=wcs.to_header()).writeto(command[command.index("--wcs") + 1])
        Path(command[command.index("--solved") + 1]).write_bytes(b"\x01")
        return subprocess.CompletedProcess(command, 0, "solved", "")

    monkeypatch.setattr(subprocess, "run", run)
    return seen


def test_solution_cache_hints_and_original_are_preserved(frame, command_log):
    original = frame.path.read_bytes()
    coord = SkyCoord(120 * u.deg, 30 * u.deg)
    result = astrometry.solve(frame, solver="solve-field", solver_options={
        "center": coord, "radius": 30 * u.arcmin, "scale": 2 * u.arcsec / u.pixel,
        "scale_err": 5, "timeout": 17, "band": "R", "config": "indexes.cfg"})
    assert result.has_celestial and frame.ccd.wcs is result
    command, kwargs = command_log[0]
    assert command[command.index("--radius") + 1] == "0.5"
    assert float(command[command.index("--scale-low") + 1]) == pytest.approx(1.9)
    assert kwargs["timeout"] == 17
    assert not Path(command[-1]).exists()  # private working directory removed
    assert frame.path.read_bytes() == original
    assert astrometry.load_wcs(frame).has_celestial
    astrometry.solve(frame, solver="solve-field")
    assert len(command_log) == 1
    astrometry.solve(frame, solver="solve-field", force=True)
    assert len(command_log) == 2


@pytest.mark.parametrize("outcome", ["unsolved", "no-wcs", "bad-wcs", "rc", "timeout"])
def test_failures_do_not_cache_a_solution(frame, command_log, monkeypatch, outcome):
    def run(command, **kwargs):
        if outcome == "timeout":
            raise subprocess.TimeoutExpired(command, kwargs["timeout"])
        if outcome == "rc":
            return subprocess.CompletedProcess(command, 2, "", "bad index")
        if outcome != "unsolved":
            Path(command[command.index("--solved") + 1]).write_bytes(b"\x01")
        if outcome == "bad-wcs":
            fits.PrimaryHDU().writeto(command[command.index("--wcs") + 1])
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(subprocess, "run", run)
    with pytest.raises(RuntimeError):
        astrometry.solve_field(frame)
    assert not astrometry.wcs_cache_path(frame.path).exists()


def test_missing_hints_allow_a_blind_solve(frame, command_log):
    frame.header.clear()
    astrometry.solve_field(frame)
    command = command_log[0][0]
    assert "--ra" not in command and "--scale-low" not in command


def test_solver_from_extraction_dispatches_without_reextracting(frame, command_log):
    from seestar_photometry import photometry

    ext = photometry.extract_sources(frame)
    assert ext.solve_wcs(solver="solve-field", solver_options={"downsample": 1}).has_celestial
    assert command_log[0][0][command_log[0][0].index("--downsample") + 1] == "1"


def test_missing_executable_gives_setup_instructions(monkeypatch):
    import shutil

    monkeypatch.setattr(shutil, "which", lambda name: None)
    with pytest.raises(RuntimeError, match="index files"):
        astrometry._solver_executable("solve-field", "/missing/solver", "ASTROMETRY_SOLVE_FIELD")
