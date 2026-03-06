# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""
LOKI logical detector view transform functions.

These transforms are registered with the instrument via instrument.add_logical_view()
in specs.py.
"""

import scipp as sc


def get_tube_view(da: sc.DataArray, source_name: str) -> sc.DataArray:
    """Transform to fold detector data for tube view.

    Folds raw detector data into (layer, straw, tube, pixel) dimensions using
    bank-specific sizes from DETECTOR_BANK_SIZES. The subsequent summing over
    'straw' and 'pixel' dimensions is handled by the reduction_dim parameter
    in add_logical_view to enable proper ROI support.

    Parameters
    ----------
    da:
        Raw detector data with a single dimension.
    source_name:
        Name of the detector bank (e.g., 'loki_detector_0').

    Returns
    -------
    :
        Folded data with dimensions (layer, straw, tube, pixel).
    """
    from ess.loki.workflow import DETECTOR_BANK_SIZES

    sizes = DETECTOR_BANK_SIZES[source_name]
    return da.fold(dim=da.dim, sizes=sizes)
