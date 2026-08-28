# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""
TBL logical detector view transform functions.

These transforms are wired to the workflow specs registered in specs.py.
"""

import scipp as sc

#: Largest image published per axis. Cameras read out far more pixels than the
#: dashboard can plot, and ess.livedata may not keep up with the full grid, so
#: images are downsampled to at most this many pixels per axis.
MAX_IMAGE_SIZE = 512


def fold_image(da: sc.DataArray, source_name: str) -> sc.DataArray:
    """Fold detector image dimensions for downsampling to 512x512."""
    # 4096x4096 or 2048x2048 is the actual panel size, but ess.livedata might not be
    #  able to keep up with that so we downsample to 512x512.
    da = da.rename_dims({'dim_0': 'x', 'dim_1': 'y'})
    da = da.fold(dim='x', sizes={'x': MAX_IMAGE_SIZE, 'x_bin': -1})
    da = da.fold(dim='y', sizes={'y': MAX_IMAGE_SIZE, 'y_bin': -1})
    return da


def fold_timepix3_image(da: sc.DataArray, resolution: int) -> sc.DataArray:
    """Cut the Timepix3 pixel grid down to ``resolution`` and fold it for display.

    The Timepix3 readout maps hits onto a square pixel grid whose size is a
    detector setting, while the geometry file declares the largest grid,
    4096x4096. A coarser mode is assumed to enumerate its pixels the same way,
    so its event ids span the first ``resolution**2`` ids of the declared grid.

    Events are grouped by the declared grid before any workflow sees them, in a
    per-source preprocessor that no workflow parameter can reach, which is why
    the mode is applied here as a leading slice rather than by grouping into a
    smaller grid in the first place.
    """
    n_pixels = resolution * resolution
    da = da.flatten(to='detector_number')['detector_number', :n_pixels]
    da = da.fold(dim='detector_number', sizes={'x': resolution, 'y': resolution})
    size = min(MAX_IMAGE_SIZE, resolution)
    da = da.fold(dim='x', sizes={'x': size, 'x_bin': -1})
    da = da.fold(dim='y', sizes={'y': size, 'y_bin': -1})
    return da


def get_multiblade_view(da: sc.DataArray, source_name: str) -> sc.DataArray:
    """Fold detector_number into blade, wire, and strip dimensions.

    A Multi-Blade cassette is a stack of inclined plates, the blades. Each plate
    carries 64 strips across it and 32 wires along it, so ``blade`` enumerates the
    plates and runs along the stacking direction, while ``strip`` and ``wire`` are
    coordinates within a single plate.

    In the TBL vessel the plates are horizontal and stacked vertically: strips run
    horizontally across the beam, wires roughly along it. ESTIA mounts the same
    hardware rotated 90 degrees about the beam, and its geometry file enumerates
    ``detector_number`` in a different order, so ``estia.views.get_multiblade_view``
    folds with a different dimension order. Both orders were verified against their
    geometry files (the fold that makes each pixel coordinate a function of exactly
    one axis is unique); neither should be changed to match the other.
    """
    return da.fold(dim='detector_number', sizes={'blade': 14, 'wire': -1, 'strip': 64})


def get_he3_detector_view(da: sc.DataArray, source_name: str) -> sc.DataArray:
    """Transform to rename dimensions to tube and pixel."""
    return da.rename_dims(dim_0='tube', dim_1='pixel')


def identity(da: sc.DataArray, source_name: str) -> sc.DataArray:
    """Identity transform (no-op)."""
    return da


def get_he3_spectrum(histogram: sc.DataArray) -> sc.DataArray:
    """Per-tube, per-pixel He3 spectra.

    A bank is 4 tubes of 100 pixels, small enough to publish without any spatial
    reduction. The copy decouples the published result from the cumulative
    accumulator's buffer, which the next update mutates in place.
    """
    return histogram.copy()


def get_multiblade_spectrum(histogram: sc.DataArray) -> sc.DataArray:
    """Sum over ``strip``, the coarse axis across each blade.

    The blades are inclined by ~5 degrees, so the 32 wires of one plate project onto
    the stacking direction over slightly more than the 10.5 mm blade pitch. ``blade``
    and ``wire`` together therefore interleave into a comb of 448 channels at
    ~0.33 mm along that direction, more than ten times finer than the 4 mm strip
    pitch measured across a plate. Summing ``strip`` collapses the coarse axis and
    keeps that fine one, which is the resolution the Multi-Blade exists to provide.
    """
    return histogram.sum('strip')
