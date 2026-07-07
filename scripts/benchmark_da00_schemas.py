# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Scipp contributors (https://github.com/scipp)
"""
Benchmark: generic da00 encode/decode vs. a dedicated result-data schema.

Investigates whether dedicated flatbuffer schemas for result data (mostly 0-D,
1-D, or 2-D with one coord per dim plus scalar start_time/end_time coords)
could substantially reduce encode/decode times compared to the generic da00
schema.

Pipeline under test (production path in esslivedata):

  encode:  sc.DataArray -> scipp_to_da00() -> serialise_da00() -> bytes
  decode:  bytes -> deserialise_da00() -> da00_to_scipp() -> sc.DataArray

Three variants are compared:

1. ``enc``/``dec``: the current production pipeline (streaming_data_types).
2. ``enc-opt``/``dec-opt``: hand-optimized encoder/decoder for the *unchanged*
   da00 wire format (fused scipp conversion, module-level dtype maps, no
   ``flatten()`` copy, direct flatbuffers table access). Isolates how much of
   the cost is implementation overhead vs. inherent to the schema.
3. ``enc-ded``/``dec-ded``: prototype dedicated schema — a single flatbuffer
   table with fixed slots covering 0-D/1-D/2-D:
   - start_time/end_time are scalar int64 fields instead of full Variable
     tables (each carrying name/unit strings, axes/shape/data vectors)
   - signal/errors/coords are typed float64 vectors in fixed slots
   - dim/coord names and units are plain string fields

Representative results (Python 3.11, x86-64, best of 7, microseconds/message):

  case       |    enc  enc-opt  enc-ded  ded |    dec  dec-opt  dec-ded  ded
  -----------|-------------------------------|----------------------------
  0d         |    299      272       65 4.6x |    263      126       54 4.9x
  1d-1k      |    390      365      114 3.4x |    358      175       87 4.1x
  1d-1k+err  |    525      485      130 4.0x |    519      286      107 4.9x
  1d-10k     |    515      479      198 2.6x |    389      200      107 3.6x
  2d-128     |    575      515      201 2.9x |    455      249      127 3.6x
  2d-512     |   2694     2370     1815 1.5x |    530      537      413 1.3x

Conclusions:

- A dedicated schema gives ~3-5x faster encode AND decode for typical small
  result messages (0-D/1-D/2-D up to ~100k elements). Large payloads (2d-512,
  ~2 MB) are bandwidth-bound; the schema barely matters there.
- The win comes from eliminating the generic per-variable table structure
  (name/unit strings, axes string-vector, shape vector, dtype dispatch per
  variable, all built via pure-Python flatbuffers), NOT from specializing per
  dimensionality: this prototype is a single schema covering 0-D..2-D, and
  per-dimensionality splits would only save a couple of empty-slot writes
  (~1-2 us). The scalar start_time/end_time coords are the worst offenders in
  da00: two full Variable tables to move two int64 values.
- Optimizing the implementation while keeping the da00 wire format recovers
  almost nothing on encode (the per-variable tables must still be built) and
  ~2x on decode (fusing deserialise+scipp conversion, skipping generated-code
  wrappers).
- Floor check: bare scipp DataArray construction from ready numpy arrays costs
  ~34 us (1d-1k) / ~13 us (0-D), so the dedicated decoder is within ~2.5x of
  the attainable floor.

Run: python scripts/benchmark_da00_schemas.py  (needs the package installed)
"""

import timeit
from dataclasses import dataclass

import flatbuffers
import numpy as np
import scipp as sc
from flatbuffers import number_types as N
from flatbuffers.table import Table
from streaming_data_types import dataarray_da00

from ess.livedata.kafka.scipp_da00_compat import da00_to_scipp, scipp_to_da00

# ---------------------------------------------------------------------------
# Representative payloads: signal + one coord per dim (bin edges) + scalar
# start_time/end_time coords, as produced by Job._add_time_coords.
# ---------------------------------------------------------------------------


def make_payload(shape: tuple[int, ...], variances: bool = False) -> sc.DataArray:
    rng = np.random.default_rng(42)
    dims = ['x', 'y'][: len(shape)]
    values = rng.random(shape)
    data = sc.array(
        dims=dims,
        values=values,
        variances=values.copy() if variances else None,
        unit='counts',
    )
    coords = {
        dim: sc.linspace(dim, 0.0, 1.0, num=n + 1, unit='mm')
        for dim, n in zip(dims, shape, strict=True)
    }
    coords['start_time'] = sc.scalar(1_000_000_000_000, unit='ns')
    coords['end_time'] = sc.scalar(1_001_000_000_000, unit='ns')
    return sc.DataArray(data, coords=coords, name='intensity')


CASES = {
    '0d': make_payload(()),
    '1d-1k': make_payload((1000,)),
    '1d-1k+err': make_payload((1000,), variances=True),
    '1d-10k': make_payload((10_000,)),
    '2d-128': make_payload((128, 128)),
    '2d-512': make_payload((512, 512)),
}

SOURCE = 'job-1234/detector_view'
TS = 1_000_000_000_000


# ---------------------------------------------------------------------------
# Current pipeline
# ---------------------------------------------------------------------------


def encode_da00(da: sc.DataArray) -> bytes:
    return dataarray_da00.serialise_da00(
        source_name=SOURCE, timestamp_ns=TS, data=scipp_to_da00(da)
    )


def decode_da00(buf: bytes) -> sc.DataArray:
    return da00_to_scipp(dataarray_da00.deserialise_da00(buf).data)


# ---------------------------------------------------------------------------
# Optimized encoder/decoder for the UNCHANGED da00 wire format.
#
# Same bytes on the wire (validated against streaming_data_types), but:
#   - fused with the scipp conversion (no Variable dataclass intermediate)
#   - module-level dtype maps instead of per-call dict construction
#   - np.ravel (no copy for contiguous) instead of .flatten() (always copies)
#   - direct flatbuffers.Table access instead of generated-code wrappers
# ---------------------------------------------------------------------------

_NP_TO_DA00 = {
    np.dtype('int8'): 1,
    np.dtype('uint8'): 2,
    np.dtype('int16'): 3,
    np.dtype('uint16'): 4,
    np.dtype('int32'): 5,
    np.dtype('uint32'): 6,
    np.dtype('int64'): 7,
    np.dtype('uint64'): 8,
    np.dtype('float32'): 9,
    np.dtype('float64'): 10,
}
_DA00_TO_NP = {v: k for k, v in _NP_TO_DA00.items()}


def _pack_variable_opt(
    b: flatbuffers.Builder,
    name: str,
    var: sc.Variable,
    label: str | None = None,
) -> int:
    values = var.values
    if var.dtype == sc.DType.datetime64:
        values = values.view(np.int64)
        unit = f'datetime64[{var.unit}]'
    else:
        unit = None if var.unit is None else str(var.unit)
    label_off = b.CreateString(label) if label is not None else None
    unit_off = b.CreateString(unit) if unit is not None else None
    name_off = b.CreateString(name)
    shape_off = b.CreateNumpyVector(np.asarray(values.shape, dtype=np.int64))
    data_off = b.CreateNumpyVector(np.ravel(values).view(np.uint8))
    axes_offs = [b.CreateString(d) for d in var.dims]
    b.StartVector(4, len(axes_offs), 4)
    for off in reversed(axes_offs):
        b.PrependUOffsetTRelative(off)
    axes_off = b.EndVector()
    b.StartObject(8)
    b.PrependUOffsetTRelativeSlot(0, name_off, 0)
    if unit_off is not None:
        b.PrependUOffsetTRelativeSlot(1, unit_off, 0)
    if label_off is not None:
        b.PrependUOffsetTRelativeSlot(2, label_off, 0)
    b.PrependInt8Slot(4, _NP_TO_DA00[values.dtype], 0)
    b.PrependUOffsetTRelativeSlot(5, axes_off, 0)
    b.PrependUOffsetTRelativeSlot(6, shape_off, 0)
    b.PrependUOffsetTRelativeSlot(7, data_off, 0)
    return b.EndObject()


def encode_da00_opt(da: sc.DataArray) -> bytes:
    """Same da00 wire format, hand-optimized, fused scipp->bytes."""
    b = flatbuffers.Builder(1024)
    b.ForceDefaults(True)
    if da.variances is None:
        offs = [_pack_variable_opt(b, 'signal', da.data, label=da.name)]
    else:
        offs = [
            _pack_variable_opt(b, 'signal', sc.values(da.data), label=da.name),
            _pack_variable_opt(b, 'errors', sc.stddevs(da.data)),
        ]
    offs.extend(
        _pack_variable_opt(b, name, var)
        for name, var in da.coords.items()
        if var.shape == var.values.shape
    )
    b.StartVector(4, len(offs), 4)
    for off in reversed(offs):
        b.PrependUOffsetTRelative(off)
    data_off = b.EndVector()
    source_off = b.CreateString(SOURCE)
    b.StartObject(3)
    b.PrependUOffsetTRelativeSlot(0, source_off, 0)
    b.PrependInt64Slot(1, TS, 0)
    b.PrependUOffsetTRelativeSlot(2, data_off, 0)
    b.Finish(b.EndObject(), file_identifier=b'da00')
    return bytes(b.Output())


def _tab_string(tab: Table, slot: int) -> str | None:
    o = tab.Offset(4 + 2 * slot)
    return tab.String(o + tab.Pos).decode() if o else None


def decode_da00_opt(buf: bytes) -> sc.DataArray:
    """Same da00 wire format, hand-optimized, fused bytes->scipp."""
    root = flatbuffers.encode.Get(flatbuffers.packer.uoffset, bytearray(buf), 0)
    tab = Table(buf, root)
    o = tab.Offset(8)  # DataArray.data (slot 2)
    n_vars = tab.VectorLen(o)
    vec = tab.Vector(o)
    signal = None
    errors = None
    label = ''
    coords: dict[str, sc.Variable] = {}
    for j in range(n_vars):
        vt = Table(buf, tab.Indirect(vec + 4 * j))
        name = _tab_string(vt, 0)
        unit = _tab_string(vt, 1)
        do = vt.Offset(4 + 2 * 7)
        values = vt.GetVectorAsNumpy(N.Uint8Flags, do)
        dto = vt.Offset(4 + 2 * 4)
        dtype = vt.Get(N.Int8Flags, dto + vt.Pos) if dto else 0
        values = values.view(_DA00_TO_NP[dtype])
        ao = vt.Offset(4 + 2 * 5)
        n_axes = vt.VectorLen(ao) if ao else 0
        avec = vt.Vector(ao) if ao else 0
        dims = [vt.String(avec + 4 * i).decode() for i in range(n_axes)]
        if n_axes > 1:
            so = vt.Offset(4 + 2 * 6)
            values = values.reshape(vt.GetVectorAsNumpy(N.Int64Flags, so))
        elif n_axes == 0:
            values = values.reshape(())
        if values.dtype in _DTYPE_MAP_LOCAL:
            values = values.astype(_DTYPE_MAP_LOCAL[values.dtype])
        if unit is not None and unit.startswith('datetime64'):
            u = unit.split('[')[1].rstrip(']')
            var = sc.epoch(unit=u) + sc.array(dims=dims, values=values, unit=u)
        else:
            var = sc.array(dims=dims, values=values, unit=unit)
        if name == 'signal':
            signal = var
            lo = _tab_string(vt, 2)
            label = lo if lo is not None else ''
        elif name == 'errors':
            errors = var
        else:
            coords[name] = var
    data = signal
    if errors is not None:
        data.variances = (errors**2).values
    compatible = {
        k: v for k, v in coords.items() if set(v.dims).issubset(set(data.dims))
    }
    return sc.DataArray(data, coords=compatible, name=label)


_DTYPE_MAP_LOCAL = {
    np.dtype('uint8'): np.int32,
    np.dtype('int8'): np.int32,
    np.dtype('uint16'): np.int32,
    np.dtype('int16'): np.int32,
    np.dtype('uint32'): np.int64,
    np.dtype('uint64'): np.float64,
}


# ---------------------------------------------------------------------------
# Prototype dedicated schema (hand-rolled flatbuffers, one table, fixed slots).
#
# table Result {                       // covers 0-D, 1-D, 2-D
#   source_name: string;               // slot 0
#   timestamp_ns: long;                // slot 1
#   start_time_ns: long;               // slot 2
#   end_time_ns: long;                 // slot 3
#   name: string;                      // slot 4  (DataArray.name)
#   unit: string;                      // slot 5  (signal unit)
#   signal: [double];                  // slot 6
#   errors: [double];                  // slot 7  (stddevs; optional)
#   dim0: string; coord0: [double]; unit0: string;   // slots 8,9,10
#   dim1: string; coord1: [double]; unit1: string;   // slots 11,12,13
#   n0: long; n1: long;                // slots 14,15 (signal shape; 0-D: absent)
# }
# ---------------------------------------------------------------------------

NUM_SLOTS = 16


def encode_dedicated(da: sc.DataArray) -> bytes:
    b = flatbuffers.Builder(1024)
    values = da.values
    ndim = values.ndim
    # vectors and strings must be created before StartObject
    off_source = b.CreateString(SOURCE)
    off_name = b.CreateString(da.name)
    off_unit = b.CreateString(str(da.unit))
    off_signal = b.CreateNumpyVector(np.ravel(values))
    off_errors = (
        b.CreateNumpyVector(np.ravel(np.sqrt(da.variances)))
        if da.variances is not None
        else None
    )
    dim_offs = []
    for i in range(ndim):
        dim = da.dims[i]
        coord = da.coords[dim]
        dim_offs.append(
            (
                b.CreateString(dim),
                b.CreateNumpyVector(coord.values),
                b.CreateString(str(coord.unit)),
            )
        )
    b.StartObject(NUM_SLOTS)
    b.PrependUOffsetTRelativeSlot(0, off_source, 0)
    b.PrependInt64Slot(1, TS, 0)
    b.PrependInt64Slot(2, int(da.coords['start_time'].value), 0)
    b.PrependInt64Slot(3, int(da.coords['end_time'].value), 0)
    b.PrependUOffsetTRelativeSlot(4, off_name, 0)
    b.PrependUOffsetTRelativeSlot(5, off_unit, 0)
    b.PrependUOffsetTRelativeSlot(6, off_signal, 0)
    if off_errors is not None:
        b.PrependUOffsetTRelativeSlot(7, off_errors, 0)
    for i, (od, oc, ou) in enumerate(dim_offs):
        b.PrependUOffsetTRelativeSlot(8 + 3 * i, od, 0)
        b.PrependUOffsetTRelativeSlot(9 + 3 * i, oc, 0)
        b.PrependUOffsetTRelativeSlot(10 + 3 * i, ou, 0)
    if ndim >= 1:
        b.PrependInt64Slot(14, values.shape[0], 0)
    if ndim == 2:
        b.PrependInt64Slot(15, values.shape[1], 0)
    b.Finish(b.EndObject(), file_identifier=b'dr00')
    return bytes(b.Output())


def _field(tab: Table, slot: int) -> int:
    return tab.Offset(4 + 2 * slot)


def _string(tab: Table, slot: int) -> str | None:
    o = _field(tab, slot)
    return tab.String(o + tab.Pos).decode() if o else None


def _int64(tab: Table, slot: int) -> int:
    o = _field(tab, slot)
    return tab.Get(N.Int64Flags, o + tab.Pos) if o else 0


def _f64vec(tab: Table, slot: int) -> np.ndarray | None:
    o = _field(tab, slot)
    return tab.GetVectorAsNumpy(N.Float64Flags, o) if o else None


def decode_dedicated(buf: bytes) -> sc.DataArray:
    n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, bytearray(buf), 0)
    tab = Table(buf, n)
    signal = _f64vec(tab, 6)
    errors = _f64vec(tab, 7)
    n0, n1 = _int64(tab, 14), _int64(tab, 15)
    dims, coords = [], {}
    for i in range(2):
        dim = _string(tab, 8 + 3 * i)
        if dim is None:
            break
        dims.append(dim)
        coords[dim] = sc.array(
            dims=[dim], values=_f64vec(tab, 9 + 3 * i), unit=_string(tab, 10 + 3 * i)
        )
    if n1:
        signal = signal.reshape(n0, n1)
        if errors is not None:
            errors = errors.reshape(n0, n1)
    elif not n0:
        signal = signal[0]
        errors = errors[0] if errors is not None else None
    data = (
        sc.scalar(
            signal,
            variance=errors**2 if errors is not None else None,
            unit=_string(tab, 5),
        )
        if not dims
        else sc.array(
            dims=dims,
            values=signal,
            variances=errors**2 if errors is not None else None,
            unit=_string(tab, 5),
        )
    )
    coords['start_time'] = sc.scalar(_int64(tab, 2), unit='ns')
    coords['end_time'] = sc.scalar(_int64(tab, 3), unit='ns')
    return sc.DataArray(data, coords=coords, name=_string(tab, 4))


# ---------------------------------------------------------------------------
# Benchmark driver
# ---------------------------------------------------------------------------


def bench(fn, *args, repeat=7) -> float:
    """Best-of time per call in microseconds."""
    t = timeit.Timer(lambda: fn(*args))
    number, _ = t.autorange()
    return min(t.repeat(repeat=repeat, number=number)) / number * 1e6


@dataclass
class Row:
    case: str
    enc_da00: float
    enc_opt: float
    enc_ded: float
    dec_da00: float
    dec_opt: float
    dec_ded: float
    size_da00: int
    size_ded: int


def main() -> None:
    rows = []
    for case, da in CASES.items():
        buf = encode_da00(da)
        ded = encode_dedicated(da)
        opt = encode_da00_opt(da)
        # correctness: optimized generic encoder must round-trip through the
        # REFERENCE decoder (same wire format), and both alternative decoders
        # must reproduce the reference pipeline's output.
        orig = decode_da00(buf)
        for label, candidate in (
            ('opt wire', decode_da00(opt)),
            ('opt decode', decode_da00_opt(buf)),
            ('dedicated roundtrip', decode_dedicated(ded)),
        ):
            if not sc.identical(candidate, orig):
                raise AssertionError(f'{label} mismatch for case {case}')

        rows.append(
            Row(
                case=case,
                enc_da00=bench(encode_da00, da),
                enc_opt=bench(encode_da00_opt, da),
                enc_ded=bench(encode_dedicated, da),
                dec_da00=bench(decode_da00, buf),
                dec_opt=bench(decode_da00_opt, buf),
                dec_ded=bench(decode_dedicated, ded),
                size_da00=len(buf),
                size_ded=len(ded),
            )
        )

    print('All timings: microseconds per message, best of 7 (scipp <-> bytes).')
    hdr = (
        f'{"case":<11}| {"enc":>8} {"enc-opt":>8} {"enc-ded":>8} {"ded":>5} | '
        f'{"dec":>8} {"dec-opt":>8} {"dec-ded":>8} {"ded":>5} | '
        f'{"bytes":>7} {"bytes-d":>7}'
    )
    print(hdr)
    print('-' * len(hdr))
    for r in rows:
        print(
            f'{r.case:<11}| {r.enc_da00:>8.1f} {r.enc_opt:>8.1f} '
            f'{r.enc_ded:>8.1f} {r.enc_da00 / r.enc_ded:>4.1f}x | '
            f'{r.dec_da00:>8.1f} {r.dec_opt:>8.1f} {r.dec_ded:>8.1f} '
            f'{r.dec_da00 / r.dec_ded:>4.1f}x | {r.size_da00:>7} {r.size_ded:>7}'
        )


if __name__ == '__main__':
    main()
