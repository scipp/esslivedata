# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""
ODIN logical detector view transform functions.

These transforms are registered with the instrument via instrument.add_logical_view()
in specs.py.
"""

import scipp as sc


def fold_image(da: sc.DataArray, source_name: str) -> sc.DataArray:
    """Fold detector image dimensions for downsampling to 512x512."""
    # 4096x4096 is the actual panel size, but ess.livedata might not be able to keep
    # up with that so we downsample to 512x512.
    # The geometry file has generic dim_0/dim_1 names, so we rename to x/y.
    da = da.rename_dims({'dim_0': 'x', 'dim_1': 'y'})
    da = da.fold(dim='x', sizes={'x': 512, 'x_bin': -1})
    da = da.fold(dim='y', sizes={'y': 512, 'y_bin': -1})
    return da
