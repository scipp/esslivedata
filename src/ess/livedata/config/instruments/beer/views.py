# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""
BEER logical detector view transform functions.

These transforms are registered with the instrument via instrument.add_logical_view()
in specs.py.
"""

import scipp as sc

#: Logical shape of one BEER detector bank, from ``detector_number`` in the
#: NeXus baseline: 12 panels stacked over 19 mm of depth, each 1000x1000 pixels
#: of 1 mm. The panel axis is the one the instrument team sums over to go from
#: per-panel to per-bank images. The two spatial axes are equally sized, so the
#: file does not reveal which of them ``x_pixel_offset`` indexes; a swap would
#: transpose the images and nothing else.
BANK_SIZES = {'panel': 12, 'y': 1000, 'x': 1000}

#: Spatial downsampling factors. They bound each view's screen -- and hence the
#: accumulated per-pixel time-of-arrival histogram behind it -- at a few hundred
#: thousand pixels, in the range of the largest existing detector view (TBL's
#: 512x512 Timepix3). At full 1 mm resolution a single bank would be 12M screen
#: pixels, which is not viable for a live view.
_PANEL_VIEW_FACTOR = 8  # 12 panels of 125x125, i.e. 8 mm pixels
_BANK_VIEW_FACTOR = 4  # one 250x250 image, i.e. 4 mm pixels


def _fold_bank(da: sc.DataArray, *, factor: int) -> sc.DataArray:
    """Fold the flat pixel axis into panels, splitting off spatial binning dims.

    ``factor`` adjacent pixels in each spatial direction end up in the ``x_bin``
    and ``y_bin`` dimensions. Summing those is left to ``reduction_dim`` so that
    ROI index mapping stays intact.
    """
    da = da.fold(dim='detector_number', sizes=BANK_SIZES)
    da = da.fold(dim='y', sizes={'y': -1, 'y_bin': factor})
    return da.fold(dim='x', sizes={'x': -1, 'x_bin': factor})


def get_panel_view(da: sc.DataArray, source_name: str) -> sc.DataArray:
    """Fold a bank into its 12 panels, downsampled spatially."""
    return _fold_bank(da, factor=_PANEL_VIEW_FACTOR)


def get_bank_view(da: sc.DataArray, source_name: str) -> sc.DataArray:
    """Fold a bank for the panel-summed image, downsampled spatially."""
    return _fold_bank(da, factor=_BANK_VIEW_FACTOR)
