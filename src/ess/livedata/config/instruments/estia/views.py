# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""
ESTIA logical detector view transform functions.

These transforms are registered with the instrument via instrument.add_logical_view()
in specs.py.
"""

import scipp as sc


def get_multiblade_view(da: sc.DataArray, source_name: str) -> sc.DataArray:
    """Fold detector_number into strip, blade, and wire dimensions.

    ``blade`` enumerates the inclined plates of the Multi-Blade cassette and runs
    along the stacking direction; ``strip`` and ``wire`` are coordinates within a
    single plate. On ESTIA the plates are vertical and stacked horizontally, so
    strips run vertically.

    The dimension order differs from TBL's fold of the same hardware, because the
    two geometry files enumerate ``detector_number`` differently: ESTIA runs wires
    fastest and strips slowest, TBL runs strips fastest and blades slowest. Both
    were verified against their geometry files; see ``tbl.views.get_multiblade_view``.
    """
    from ess.estia.beamline import DETECTOR_BANK_SIZES

    return da.fold(dim=da.dim, sizes=DETECTOR_BANK_SIZES[source_name])
