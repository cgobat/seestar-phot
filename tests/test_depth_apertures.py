"""Empirical aperture noise must reflect the processed image, not pixel RMS."""

from types import SimpleNamespace

import numpy as np
import pytest
from scipy.ndimage import gaussian_filter

from seestar_photometry import photometry


def _extraction(data):
    cube = np.repeat(data[None], 3, axis=0)
    return SimpleNamespace(
        frame=SimpleNamespace(data=cube), background=np.zeros_like(cube),
        # These tests prescribe source-free sky; prevent noise peaks being masked.
        rms=np.full(3, 10.0), fwhm=np.ones(3), aperture=np.full(3, 4.5),
    )


@pytest.mark.filterwarnings("ignore:No sources were found")
def test_depth_measures_correlated_noise_and_is_reproducible():
    rng = np.random.default_rng(123)
    white = rng.normal(size=(300, 300))
    correlated = gaussian_filter(white, 1.0)
    correlated /= correlated.std()
    kwargs = dict(n_apertures=1000, n_iters=2, seed=31)
    independent = photometry.aperture_depth(_extraction(white), **kwargs)
    measured = photometry.aperture_depth(_extraction(correlated), **kwargs)
    assert measured == photometry.aperture_depth(_extraction(correlated), **kwargs)
    assert measured["sigma_aper"] > 2 * independent["sigma_aper"]
    assert independent["sigma_aper"] == pytest.approx(np.sqrt(np.pi * 4.5**2), rel=0.15)
    assert measured["flux_limit"] == 5 * measured["sigma_aper"]


@pytest.mark.filterwarnings("ignore:No sources were found")
def test_depth_excludes_invalid_coverage_and_respects_user_mask():
    rng = np.random.default_rng(42)
    sky = rng.normal(size=(200, 200))
    sky[:35] = np.nan
    mask = np.zeros(sky.shape, dtype=bool)
    mask[100:, 100:] = True
    sky[mask] += 1000
    result = photometry.aperture_depth(
        _extraction(sky), mask=mask, n_apertures=200, n_iters=2
    )
    assert 5 < result["sigma_aper"] < 10
    with pytest.raises(ValueError, match="no unmasked"):
        photometry.aperture_depth(_extraction(sky), mask=np.ones(sky.shape, bool))


def test_depth_masks_source_wings(extraction):
    measured = photometry.aperture_depth(extraction)
    assert np.isfinite(measured["flux_limit"])
    assert measured["n_apertures"] == 500
    assert measured["sigma_aper"] > 0
