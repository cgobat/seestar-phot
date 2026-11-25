"""Reference-catalogue handling: a cached oversized Gaia field, and cross-match.

A Seestar dithers through a night, so a dataset covers more sky than any single
FOV. We therefore build one **oversized catalogue once** -- tiling the area into
overlapping cones queried in parallel, then concatenating and de-duplicating -- and
cache it as an ECSV. Every frame then subsets the cached catalogue to its own
footprint (:func:`sources_in_frame`); no per-frame catalogue queries are needed,
which is what makes the per-frame work pure CPU and trivially parallel.

The direct Gaia DR3 TAP query joins GSPC Johnson-Kron-Cousins B/V/R and
SDSS g/r/i photometry, including flux errors and validation flags. Missing
synthetic magnitudes remain masked; the chosen calibration selects its own
reference columns. Proper motions are available for optional epoch propagation.

See ``docs/astrometry-and-gaia.md``.
"""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import astropy.units as u
import numpy as np
from astropy.coordinates import SkyCoord, SkyOffsetFrame
from astropy.table import MaskedColumn, Table, unique, vstack

GSPC_BANDS = ("b_jkc", "v_jkc", "r_jkc", "g_sdss", "r_sdss", "i_sdss")
#: Catalogue columns carried through the field query and the cross-match.
GAIA_COLUMNS = (
    "source_id", "ra", "dec", "ref_epoch", "pmra", "pmdec",
    "phot_g_mean_mag", "phot_bp_mean_mag", "phot_rp_mean_mag", "bp_rp",
    "phot_variable_flag",
) + tuple(f"{band}_{suffix}" for band in GSPC_BANDS
          for suffix in ("mag", "flux", "flux_error", "flag"))


def _cone_query(ra, dec, radius, gmag_limit):
    photometry = ", ".join(f"s.{band}_{suffix}" for band in GSPC_BANDS
                           for suffix in ("mag", "flux", "flux_error", "flag"))
    return f"""
        SELECT g.source_id, g.ra, g.dec, g.ref_epoch, g.pmra, g.pmdec,
               g.phot_g_mean_mag, g.phot_bp_mean_mag, g.phot_rp_mean_mag, g.bp_rp,
               g.phot_variable_flag, {photometry}
        FROM gaiadr3.gaia_source AS g
        LEFT OUTER JOIN gaiadr3.synthetic_photometry_gspc AS s
            ON g.source_id = s.source_id
        WHERE 1 = CONTAINS(POINT('ICRS', g.ra, g.dec),
                           CIRCLE('ICRS', {ra}, {dec}, {radius}))
          AND g.phot_g_mean_mag < {gmag_limit}
    """


def _fetch_cone(ra, dec, radius, gmag_limit):
    """Query the actual Gaia TAP service through astroquery, without a row cap."""
    from astroquery.gaia import GaiaClass

    gaia = GaiaClass()
    gaia.ROW_LIMIT = -1
    return gaia.launch_job_async(_cone_query(ra, dec, radius, gmag_limit)).get_results()


def propagate_epoch(catalogue, epoch, ref_epoch=2016.0):
    """Propagate Gaia coordinates with SkyCoord, preserving degree column units.

    ``pmra`` is Gaia's cos(dec)-weighted proper motion. Sources lacking either
    proper-motion component keep their original coordinates and reference epoch;
    ``epoch_propagated`` records which rows moved. ``epoch`` accepts Time or a
    Julian year. No distance or radial-velocity correction is inferred.
    """
    import warnings
    from astropy.time import Time

    target = epoch if isinstance(epoch, Time) else Time(epoch, format="jyear")
    out = catalogue.copy()
    names = ("ra", "dec", "pmra", "pmdec")
    units = (u.deg, u.deg, u.mas / u.yr, u.mas / u.yr)
    values = []
    for name, unit in zip(names, units):
        column = catalogue[name]
        quantity = u.Quantity(np.ma.filled(column, np.nan),
                              unit=getattr(column, "unit", None) or unit)
        values.append(quantity.to_value(unit))
    good = np.all(np.isfinite(values), axis=0)
    origin = np.asarray(catalogue["ref_epoch"], float).copy() if "ref_epoch" in catalogue.colnames \
        else np.full(len(catalogue), ref_epoch)
    good &= np.isfinite(origin)
    if good.any():
        positions = SkyCoord(
            ra=values[0][good] * u.deg, dec=values[1][good] * u.deg,
            pm_ra_cosdec=values[2][good] * u.mas / u.yr,
            pm_dec=values[3][good] * u.mas / u.yr, obstime=Time(origin[good], format="jyear"),
        )
        with warnings.catch_warnings():
            # ERFA warns about the omitted distance; angular-only propagation is intended.
            from erfa import ErfaWarning
            warnings.simplefilter("ignore", ErfaWarning)
            moved = positions.apply_space_motion(new_obstime=target)
        out["ra"][good] = moved.ra.to_value(getattr(out["ra"], "unit", None) or u.deg)
        out["dec"][good] = moved.dec.to_value(getattr(out["dec"], "unit", None) or u.deg)
    origin[good] = target.jyear
    out["ref_epoch"] = origin
    out["epoch_propagated"] = good
    out.meta["epoch"] = float(target.jyear)
    return out


def _tile_centers(center, half_size_deg, n_tiles):
    """Grid of tile centres (ra, dec) as true angular offsets around ``center``.

    Offsets are taken in a :class:`SkyOffsetFrame` rather than by adding degrees to
    RA, so the tiling stays correct near the pole -- which matters, since MW Cam sits
    at Dec +81 where naive RA steps collapse.
    """
    origin = SkyCoord(center[0] * u.deg, center[1] * u.deg)
    frame = SkyOffsetFrame(origin=origin)
    cell = 2 * half_size_deg / n_tiles
    offsets = [-half_size_deg + cell * (i + 0.5) for i in range(n_tiles)]
    centers = []
    for dx in offsets:
        for dy in offsets:
            p = SkyCoord(lon=dx * u.deg, lat=dy * u.deg, frame=frame).icrs
            centers.append((float(p.ra.deg), float(p.dec.deg)))
    return centers, cell


def fetch_gaia_catalogue(
    center, cache_path, half_size_deg=1.5, n_tiles=1, gmag_limit=17.0,
    max_workers=5, overwrite=False, attempts=4, epoch=None,
):
    """Build (and cache) an oversized Gaia catalogue by tiling a box.

    Covers a ``2 * half_size_deg`` square centred on ``center`` with an
    ``n_tiles x n_tiles`` grid of overlapping cones, then concatenates and
    de-duplicates on ``source_id``. Cached as an ECSV and read back on later calls.

    The Gaia TAP endpoint intermittently truncates an async result
    (``IncompleteRead``), so the whole build is retried a few times. The cache is
    only written on full success -- a partial mosaic silently missing a tile is much
    worse than a failed build, because every downstream frame in that region would
    lose its calibration stars without any error.

    Parameters
    ----------
    center : SkyCoord or tuple of float
        Scalar field centre, or ``(ra, dec)`` in degrees.
    cache_path : path-like
        ECSV cache for the full catalogue.
    half_size_deg : float
        Half-size of the box. 1.5 deg comfortably covers a night's dithering.
    n_tiles : int
        Tiles per axis. Keep at 1 unless the box is large: Gaia TAP is unreliable
        under concurrent jobs, and a single cone is both faster and safer.
    """
    centre = center.icrs if isinstance(center, SkyCoord) else SkyCoord(
        *center, unit=u.deg, frame="icrs")
    if not centre.isscalar or not 0 < half_size_deg < 90 or n_tiles < 1:
        raise ValueError("a scalar centre, positive half-size <90 deg and n_tiles>=1 are required")
    if not np.isfinite(gmag_limit):
        raise ValueError("gmag_limit must be finite")
    center = (float(centre.ra.deg), float(centre.dec.deg))
    cache_path = Path(cache_path)
    if cache_path.exists() and not overwrite:
        catalogue = Table.read(cache_path)
        if epoch is not None:
            catalogue = propagate_epoch(catalogue, epoch,
                                        ref_epoch=catalogue.meta.get("epoch", 2016.0))
            catalogue.write(cache_path, format="ascii.ecsv", overwrite=True)
        return catalogue

    import time

    centers, cell = _tile_centers(center, half_size_deg, n_tiles)
    radius = cell / 2 * np.sqrt(2) + 0.1  # cover each cell (incl. corners) + overlap
    last = None
    for attempt in range(attempts):
        try:
            with ThreadPoolExecutor(max_workers=max_workers) as ex:
                tables = list(
                    ex.map(lambda c: _fetch_cone(c[0], c[1], radius, gmag_limit), centers)
                )
            catalogue = unique(vstack(tables, metadata_conflicts="silent"), keys="source_id")
            for band in GSPC_BANDS:
                mag = f"{band}_mag"
                if mag not in catalogue.colnames:
                    continue
                values = np.asarray(catalogue[mag], float)
                mask = np.ma.getmaskarray(catalogue[mag]) | ~np.isfinite(values)
                catalogue[mag] = MaskedColumn(values, mask=mask, unit=u.mag)
                flux, error = f"{band}_flux", f"{band}_flux_error"
                if flux in catalogue.colnames and error in catalogue.colnames:
                    f = np.asarray(catalogue[flux], float)
                    e = np.asarray(catalogue[error], float)
                    invalid = mask | np.ma.getmaskarray(catalogue[flux]) \
                        | np.ma.getmaskarray(catalogue[error]) \
                        | ~np.isfinite(f) | ~np.isfinite(e) | (f <= 0) | (e < 0)
                    with np.errstate(invalid="ignore", divide="ignore"):
                        sigma = (2.5 / np.log(10)) * e / f
                    catalogue[f"{band}_mag_error"] = MaskedColumn(
                        sigma, mask=invalid, unit=u.mag)
            catalogue.meta.update(catalogue_source="Gaia DR3 TAP GSPC", epoch=2016.0,
                                  catalogue_schema=2, query_ra=center[0], query_dec=center[1],
                                  query_half_size_deg=half_size_deg, query_gmag_limit=gmag_limit)
            if epoch is not None:
                catalogue = propagate_epoch(catalogue, epoch)
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            catalogue.write(cache_path, format="ascii.ecsv", overwrite=True)
            return catalogue
        except Exception as exc:  # transient TAP truncation / dropped connection
            last = exc
            if attempt < attempts - 1:
                time.sleep(5 * (attempt + 1))
    raise last


# Compatibility name: the result is a catalogue, even for a single untiled query.
fetch_gaia_mosaic = fetch_gaia_catalogue


def load_catalogue(cache_path):
    """Read a cached reference-catalogue ECSV."""
    return Table.read(cache_path)


def sources_in_frame(catalogue, wcs, frame, margin=0.0):
    """Subset a catalogue to the sources falling within a frame.

    Maps catalogue sky positions to pixel coordinates via ``wcs`` and keeps those
    inside the image (optionally expanded by ``margin`` pixels). The shape comes
    from the frame data, so both FITS layouts work.
    """
    ny, nx = frame.shape
    coordinates = SkyCoord(catalogue["ra"], catalogue["dec"], unit=u.deg, frame="icrs")
    x, y = wcs.world_to_pixel(coordinates)
    inside = (x >= -margin) & (x < nx + margin) & (y >= -margin) & (y < ny + margin)
    return catalogue[inside]


def crossmatch_table(sources, wcs, catalogue, tol_arcsec=2.0, columns=None):
    """Augment a sources table with sky position and matched catalogue columns.

    ``columns=None`` carries every catalogue column, including custom reference
    magnitudes and colours. An explicit column sequence restricts that set.

    Every input row is kept and gains ``ra``, ``dec`` (from the WCS) and
    ``sep_arcsec`` (distance to its nearest catalogue source). The catalogue columns
    are filled where the match is within ``tol_arcsec`` and **masked** otherwise --
    including where the matched source itself lacks the value (e.g. no synthetic V),
    so a non-masked ``v_jkc_mag`` always means a usable match. Returns a new table.
    """
    det = wcs.pixel_to_world(np.asarray(sources["x"]), np.asarray(sources["y"])).icrs
    ra, dec = det.ra.deg, det.dec.deg
    cat = SkyCoord(catalogue["ra"], catalogue["dec"], unit=u.deg, frame="icrs")
    idx, d2d, _ = det.match_to_catalog_sky(cat)
    unmatched = d2d.arcsec >= tol_arcsec

    out = sources.copy()
    out["ra"] = ra
    out["dec"] = dec
    out["sep_arcsec"] = d2d.arcsec
    for col in catalogue.colnames if columns is None else columns:
        if col not in catalogue.colnames:
            continue
        name = {"ra": "cat_ra", "dec": "cat_dec"}.get(col, col)
        values = catalogue[col][idx]
        mask = unmatched | np.ma.getmaskarray(values)
        out[name] = MaskedColumn(values, mask=mask)
    return out
