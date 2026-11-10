# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""
TBL logical detector view transform functions.

These transforms are registered with the instrument via instrument.add_logical_view()
in specs.py.
"""

import math

import numpy as np
import scipp as sc


def fold_image(da: sc.DataArray, source_name: str) -> sc.DataArray:
    """Fold detector image dimensions for downsampling to 512x512.

    Used with ``reduction_dim=['x_bin', 'y_bin']`` to merge each 8x8 block into
    one output pixel. This is the only option for an area detector such as
    ORCA, which streams whole frames: there are no event ids to coarsen, so the
    full-resolution image must be built and then reduced.

    Event detectors should instead be ingested at reduced resolution, which is
    far cheaper -- see :func:`name_image_dims`.
    """
    # 4096x4096 or 2048x2048 is the actual panel size, but ess.livedata might not be
    #  able to keep up with that so we downsample to 512x512.
    da = da.rename_dims({'dim_0': 'x', 'dim_1': 'y'})
    da = da.fold(dim='x', sizes={'x': 512, 'x_bin': -1})
    da = da.fold(dim='y', sizes={'y': 512, 'y_bin': -1})
    return da


def name_image_dims(da: sc.DataArray, source_name: str) -> sc.DataArray:
    """Name the detector image dimensions, without reshaping.

    For an event detector configured via
    ``Instrument.configure_detector_downsampling``, event ids are remapped onto
    the coarser grid in the preprocessor, before pixel grouping. The array
    already has the target shape here, so there is nothing left to fold or
    reduce -- unlike :func:`fold_image`, which pays for the full-resolution
    grid on every update whether one event arrived or a million.
    """
    return da.rename_dims({'dim_0': 'x', 'dim_1': 'y'})


_MULTIBLADE_SIZES = {'blade': 14, 'wire': 32, 'strip': 64}
#: Permutation of the geometry file's detector_number order that reverses blade and
#: wire; see get_multiblade_view.
_MULTIBLADE_ORDER = (
    np.arange(math.prod(_MULTIBLADE_SIZES.values()))
    .reshape(tuple(_MULTIBLADE_SIZES.values()))[::-1, ::-1, :]
    .ravel()
    .tolist()
)


def get_multiblade_view(da: sc.DataArray, source_name: str) -> sc.DataArray:
    """Fold detector_number into blade, wire, and strip dimensions.

    A Multi-Blade cassette is a stack of inclined plates, the blades. Each plate
    carries 64 strips across it and 32 wires along it, so ``blade`` enumerates the
    plates and runs along the stacking direction, while ``strip`` and ``wire`` are
    coordinates within a single plate.

    In the TBL vessel the plates are horizontal and stacked vertically: strips run
    horizontally across the beam, wires roughly along it. The axes follow how the
    instrument scientists describe the detector, looking along the beam: ``blade``
    counts up from the bottom, ``wire`` runs downstream along the beam, and
    ``strip`` runs from left to right. The geometry file numbers pixels with strip
    fastest and blade slowest, but with blade 0 at the top and wire 0 at the
    downstream end, so both are reversed here. NeXus x points left when looking
    downstream, so strip 0, at +x, is already the leftmost strip. The geometry file
    carries no rotation; this orientation is implicit in its pixel offsets.

    ESTIA mounts the same hardware rotated 90 degrees about the beam, and its
    geometry file enumerates ``detector_number`` in a different order, so
    ``estia.views.get_multiblade_view`` folds with a different dimension order.
    Both orders were verified against their geometry files (the fold that makes each
    pixel coordinate a function of exactly one axis is unique); neither should be
    changed to match the other.
    """
    # Scipp has no negative slice step; one list index permutes all pixels at once.
    return da['detector_number', _MULTIBLADE_ORDER].fold(
        dim='detector_number', sizes=_MULTIBLADE_SIZES
    )


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
    In flattened ``(blade, wire)`` order the channels rise from bottom to top, except
    at each blade boundary, where adjacent blades overlap by about 1.5 wires.
    """
    return histogram.sum('strip')
