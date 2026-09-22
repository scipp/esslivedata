# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Scipp contributors (https://github.com/scipp)
"""
FREIA logical detector view transform functions.

These transforms are registered with the instrument via instrument.add_logical_view()
in specs.py.

Detector geometry in the reference file
---------------------------------------

What ``coda_freia_999999_00022031.hdf`` shows, and what it leaves open. Only the
fold depends on this; nothing here consumes positions yet.

- The multiblade detector has ``depends_on='.'`` and absolute pixel offsets:
  x = 2.96..3.12 m, y = -0.126..0.126 m, z = 0.16..0.49 m. Every other
  component (source, choppers, monitors) also sits at the origin, so the file
  gives no beam axis or sample position to compare against.
- Per step along each fold axis: ``strip`` moves 4 mm in y only; ``wire`` moves
  -3.98 mm in x and -0.36 mm in z (along the inclined plate); ``blade`` moves
  -1.1 mm in x and 10.2 mm in z. The fine transverse direction (0.36 mm per wire,
  the one that resolves the scattering angle) is therefore z.
- ESTIA's file, for the same hardware: ``strip`` along y, ``wire`` along z,
  fine direction x, offsets centred on the origin and placed by a
  ``detector_arm`` chain. FREIA's layout is ESTIA's rotated 90 degrees about y.
- FREIA scatters vertically and ESTIA horizontally (``ess.freia`` computes
  ``theta = asin(y)`` of the outgoing direction). In the NeXus frame (z along the
  beam, y up) that difference would be a rotation about z, which would put
  FREIA's fine direction along y. The file has it along z.

It is not settled which frame the file uses. Read with x along the beam and z
up, the detector would sit 3 m downstream and 3-9 degrees above the beam, which
fits a vertical-scattering reflectometer, but nothing else in the file confirms
that reading. Check this again when a file carries real component positions,
before adding a geometric projection or a reduction workflow.
"""

import scipp as sc

#: Fold of ``detector_number`` (1..65536, contiguous) in
#: ``coda_freia_999999_00022031.hdf``: strips slowest, wires fastest, the same
#: order as ESTIA. Derived from the pixel offsets: ``y`` depends on ``strip``
#: alone, and the in-plane coordinates are identical for every strip. Matches
#: ``ess.freia`` from essreflectometry 26.9.1; earlier releases carried ESTIA's
#: 48 blades.
DETECTOR_BANK_SIZES = {'multiblade_detector': {'strip': 64, 'blade': 32, 'wire': 32}}


def get_multiblade_view(da: sc.DataArray, source_name: str) -> sc.DataArray:
    """Fold detector_number into strip, blade, and wire dimensions.

    ``blade`` enumerates the inclined plates of the Multi-Blade cassette and runs
    along the stacking direction; ``strip`` and ``wire`` are coordinates within a
    single plate. See ``estia.views.get_multiblade_view`` for the same hardware
    on ESTIA.
    """
    return da.fold(dim=da.dim, sizes=DETECTOR_BANK_SIZES[source_name])


def get_spectrum_view(histogram: sc.DataArray) -> sc.DataArray:
    """Sum over the ``strip`` axis.

    A Multi-Blade detector on a reflectometer resolves the scattering angle
    along the fine blade/wire direction, and strips run across it, so each
    (blade, wire) spectrum is at constant angle. This follows from the hardware,
    not from the reference file, whose frame is unresolved (see the module
    docstring).
    """
    return histogram.sum('strip')
