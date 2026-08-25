# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""
TBL logical detector view transform functions.

These transforms are registered with the instrument via instrument.add_logical_view()
in specs.py.
"""

import scipp as sc


def fold_image(da: sc.DataArray, source_name: str) -> sc.DataArray:
    """Fold detector image dimensions for downsampling to 512x512."""
    # 4096x4096 or 2048x2048 is the actual panel size, but ess.livedata might not be
    #  able to keep up with that so we downsample to 512x512.
    da = da.rename_dims({'dim_0': 'x', 'dim_1': 'y'})
    da = da.fold(dim='x', sizes={'x': 512, 'x_bin': -1})
    da = da.fold(dim='y', sizes={'y': 512, 'y_bin': -1})
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
