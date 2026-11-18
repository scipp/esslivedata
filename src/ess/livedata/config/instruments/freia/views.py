# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Scipp contributors (https://github.com/scipp)
"""
FREIA logical detector view transform functions.

These transforms are registered with the instrument via instrument.add_logical_view()
in specs.py.
"""

import scipp as sc


def get_multiblade_view(da: sc.DataArray, source_name: str) -> sc.DataArray:
    """Fold detector_number into blade, wire, and strip dimensions.

    A Multi-Blade cassette is a stack of inclined plates, the blades. Each plate
    carries 64 strips across it and 32 wires along it, so ``blade`` enumerates the
    plates and runs along the stacking direction, while ``strip`` and ``wire`` are
    coordinates within a single plate.

    The geometry file numbers pixels with strip fastest (64), then wire (32), then
    blade slowest (32). This was verified against the pixel offsets of
    ``coda_freia_999999_00030110.hdf``: sorted by ``detector_number`` and reshaped
    to (blade, wire, strip), each offset component depends on exactly the axes
    listed, and on no others:

    - x depends on ``strip`` only, -4 mm per step.
    - y depends on ``blade`` (-10.9 mm per step) and ``wire`` (-0.57 mm per step).
    - z depends on ``wire`` (-3.96 mm per step) and ``blade`` (about +0.5 mm per
      step, from the blade inclination).

    Strips therefore run horizontally across the plate, blades are stacked
    vertically, and the wires run along the beam. The fine vertical direction that
    resolves the scattering angle is carried by ``blade`` and ``wire`` together.
    Blade 0 is at the top, wire 0 at the downstream end and strip 0 at +x. The view
    keeps the file's order and orientation; unlike ``tbl.views.get_multiblade_view``
    it does not reverse blade and wire, since no FREIA numbering convention has been
    given.

    The same hardware is folded the same way on TBL. ESTIA's geometry file numbers
    pixels in the opposite order (strip slowest), see
    ``estia.views.get_multiblade_view``. The order has to be checked against each
    new geometry file: ``ess.freia`` 26.9.1 still folds with strip slowest, which
    matches neither this file nor the previous one's pixel offsets.

    The bank sizes come from ``ess.freia``; only their order is taken from the file.
    """
    from ess.freia.workflow import default_parameters
    from ess.reduce.nexus.types import DetectorBankSizes

    bank_sizes = default_parameters()[DetectorBankSizes][source_name]
    sizes = {dim: bank_sizes[dim] for dim in ('blade', 'wire', 'strip')}
    return da.fold(dim=da.dim, sizes=sizes)


def get_spectrum_view(histogram: sc.DataArray) -> sc.DataArray:
    """Sum over the ``strip`` axis.

    A Multi-Blade detector on a reflectometer resolves the scattering angle
    along the fine blade/wire direction, and strips run across it, so each
    (blade, wire) spectrum is at constant angle. In the pixel offsets of
    ``coda_freia_999999_00030110.hdf`` the strip axis moves along x only, while
    blade and wire move y and z.
    """
    return histogram.sum('strip')
