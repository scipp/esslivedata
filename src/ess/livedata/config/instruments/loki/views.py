# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""
LOKI logical detector view transform functions.

These transforms are registered with the instrument via instrument.add_logical_view()
in specs.py.
"""

import scipp as sc


def get_tube_view(da: sc.DataArray, source_name: str) -> sc.DataArray:
    """Identity transform — data arrives with correct layer/tube/straw/pixel shape.

    The subsequent summing over 'straw' and 'pixel' dimensions is handled by the
    reduction_dim parameter in add_logical_view to enable proper ROI support.

    Parameters
    ----------
    da:
        Detector data with dimensions (layer, tube, straw, pixel).
    source_name:
        Name of the detector bank (e.g., 'loki_detector_0').

    Returns
    -------
    :
        Unchanged data.
    """
    return da
