# How deep does it go?

> The numerical examples below are historical independent-pixel results. Rerun them
> to obtain empirical depths; they are not updated measurements of the sample data.

Two questions, and they need different answers: *how deep is this frame*, and *how deep would
N hours be*.

## This frame

```python
from seestar_photometry import calibration, examples, photometry, quality

frame = examples.stack()
ext = photometry.extract_sources(frame)
ext.match_gaia(examples.gaia(), wcs=examples.wcs())
cal = calibration.fit_zeropoint(ext.sources, band="G")

row = quality.frame_quality(ext, cal)
print(f"{row['total_exptime']:.0f} s  ->  V = {row['v_lim_5sigma']:.2f} at 5 sigma")
# 760 s  ->  V = 17.15 at 5 sigma
```

The limit comes from the zero point and the aperture noise, not from counting faint
detections:

$$V_\mathrm{lim} = \mathrm{ZP} - 2.5\log_{10}(n\,\sigma_\mathrm{aper})$$

where `sigma_aper` is the measured dispersion of source-free aperture sums.
This needs no detections near the limit; it combines a bright-star calibration
with empirical noise measured at the same aperture radius.

The pipeline now measures `sigma_aper` with Photutils `ImageDepth`: random
source-masked apertures on the final background-subtracted image, using the science
aperture radius. This captures spatial covariance present in the image. The
historical independent-pixel estimate ran roughly 0.7 mag optimistic on the tested
stacks. A blank-aperture limit is still not a recovery/completeness measurement,
and the standard-system magnitude is quoted at the calibration's fiducial colour.

Set `Project(depth_kwargs={"n_apertures": 1000, "n_iters": 10, "seed": 42}, ...)`
to change sampling. Use `build_frame_table(force=True)` to rebuild old or cached
rows. `depth_method`, `depth_n_apertures`, `depth_n_iters`, and `depth_seed` record
how each new result was obtained. A field without usable blank sky fails explicitly.


## Longer exposures

Background-limited photometry deepens by 1.25 mag per decade of exposure. `depth` applies
that to put frames of different integration on a common footing:

```python
from astropy.table import Table
from seestar_photometry import depth

frames_table = Table(rows)        # one row per frame, from pipeline.build_frame_table

print(depth.detection_limit(frames_table, t_ref=900.0))
# model   exptime  n_frames  v_lim_median  v_lim_std
# S50       900.0         3         17.25       ...

print(depth.limit_vs_exptime(frames_table, exptimes=(900, 3600, 14400)))
# 900 s -> 17.25    3600 s -> 18.00    14400 s -> 18.75
```

So a 15-minute S50 stack reaches V ≈ 17.25 and four hours would reach ≈ 18.75 — *if* √t holds
that far.

## Does √t actually hold?

Test it rather than assuming. The two bundled c17 stacks are the same field and unit at two
integrations:

| Exposure | 5σ limit |
|---|---|
| 760 s | 17.15 |
| 1460 s | 17.69 |

Measured gain **+0.53 mag**; the √t law predicts **+0.35 mag**. The real frame did better
than the ideal scaling, because the deeper stack also happened to have slightly better seeing
and a darker sky — which is exactly why exposure alone is not the whole story.

## Conditions, not just time

Depth is set by integration *and* conditions. `fit_depth_model` separates them:

$$V_\mathrm{lim} = a + b\log_{10} t + c\,\mathrm{SQM} + d\,\mathrm{FWHM}$$

```python
model = depth.fit_depth_model(frames_table)
print(model["coeffs"], model["rms"], model["n_frames"])
```

Fit it **per unit** — each Seestar has its own response. Three things to know before using it:

- **It is a noise model, not a zero-point correction.** It acts on the faint-end
  sensitivity, never on the flux scale, so it does *not* change a detected star's magnitude
  or its error bar.
- **It mostly re-derives photon shot noise.** For a sky-limited stack with an aperture ∝ FWHM,
  the expected coefficients are ≈ +0.5 per mag/arcsec² of sky and ≈ −0.3 per pixel of FWHM.
  A real fit landed at +0.46 and −0.20. The instrument-specific signal is in the
  *deviations*, and in the differences *between* units.
- **Beware a narrow range.** A unit whose frames span little variation in sky or seeing has
  badly constrained `c` and `d`, and they will happily take implausible values.

What it is genuinely good for: predicting the depth of a planned observation, setting 5σ
upper limits for non-detections, weighting frames in a co-add, and flagging a frame that
falls far below its condition-predicted depth (cloud, trailing, focus).

One real lever it suggests: since FWHM drives noise through aperture *area*, a tighter
aperture plus an aperture correction can claw back some depth in poor seeing.

## Looking at it

```python
from seestar_photometry import plots

plots.depth_vs_exptime(frames_table)             # with the sqrt(t) line drawn
plots.depth_vs_driver(frames_table, "sky_sb")    # exposure divided out
```

Flattening at the long-exposure end in the first panel means a systematic floor has taken
over and the extrapolation no longer applies. Validate a multi-hour claim by actually
co-adding frames — see [stacking](stacking.md) — rather than by extending the line.

Full detail in [frame-table](../frame-table.md).
