# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""
ODIN logical detector view transform functions.

These transforms are registered with the instrument via instrument.add_logical_view()
in specs.py.
"""

import scipp as sc


def name_image_dims(da: sc.DataArray, source_name: str) -> sc.DataArray:
    """Name the detector image dimensions, without reshaping.

    The Timepix3 panel is ingested at reduced resolution via
    ``Instrument.configure_detector_downsampling``: event ids are remapped onto
    the coarser grid in the preprocessor, before pixel grouping, so the
    full-resolution grid is never materialized. The array already has the target
    shape here, so there is nothing left to fold or reduce.

    The geometry file has generic dim_0/dim_1 names, so we rename to x/y.
    """
    return da.rename_dims({'dim_0': 'x', 'dim_1': 'y'})
