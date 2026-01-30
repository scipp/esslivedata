# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""
ESTIA logical detector view transform functions.

These transforms are registered with the instrument via instrument.add_logical_view()
in specs.py.
"""

import scipp as sc


def get_multiblade_view(da: sc.DataArray, source_name: str) -> sc.DataArray:
    """Transform to fold detector data into strip, blade, and wire dimensions."""
    from ess.estia.beamline import DETECTOR_BANK_SIZES

    return da.fold(dim=da.dim, sizes=DETECTOR_BANK_SIZES[source_name])
