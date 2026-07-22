# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Scipp contributors (https://github.com/scipp)
"""
Benchmark: da00 vs. revised dedicated result-data schema ('dr00' v2, issue #1057).

Revision over the first prototype (scripts/benchmark_da00_schemas.py on branch
claude/da00-schema-dimensionality-fcaagq):

- explicit ``shape: [int64]`` vector -- ndim = len(shape), no inference from
  defaulted scalar slots (zero-length dims are unambiguous)
- three dim slots (0-D..3-D), each: dim name, OPTIONAL coord vector (bin-edges
  n+1 or midpoints n), coord dtype byte, coord unit string
- dtype byte for signal and each coord (da00 enum values + datetime64), so
  int32/int64 coords (roi index) and datetime64[ns] axes (timeseries) fit
- ``time``/``start_time``/``end_time`` as presence-checked int64 fields
  (window outputs carry all three, cumulative only start/end, timeseries none)
- unit strings: absent field = unit None; present empty string = dimensionless

Payload cases mirror the wire inventory of all registered workflows
(detector views, monitors, reductions, timeseries) rather than synthetic ones.

  table Result {                                   // slot
    source_name: string;                           //  0
    timestamp_ns: long;                            //  1
    time_ns: long;                                 //  2 (presence-checked)
    start_time_ns: long;                           //  3 (presence-checked)
    end_time_ns: long;                             //  4 (presence-checked)
    name: string;                                  //  5 (DataArray.name)
    unit: string;                                  //  6 (signal unit)
    dtype: byte;                                   //  7
    signal: [ubyte];                               //  8
    errors: [ubyte];                               //  9 (stddevs; optional)
    shape: [long];                                 // 10 (len = ndim, 0..3)
    dim0: string; coord0: [ubyte];                 // 11, 12
    cdtype0: byte; cunit0: string;                 // 13, 14
    dim1: string; coord1: [ubyte];                 // 15, 16
    cdtype1: byte; cunit1: string;                 // 17, 18
    dim2: string; coord2: [ubyte];                 // 19, 20
    cdtype2: byte; cunit2: string;                 // 21, 22
  }

Representative results (Python 3.11, x86-64, best of 7, microseconds/message):

  case          |     enc  enc-ded     x |     dec  dec-ded     x
  --------------|------------------------|-----------------------
  0d-total      |   257.1     60.4  4.3x |   223.5     44.5  5.0x
  1d-monitor    |   262.7    105.4  2.5x |   236.4     64.4  3.7x
  1d-1k+err     |   636.7    125.4  5.1x |   615.9     81.3  7.6x
  2d-nocoord    |   378.1    214.2  1.8x |   344.1    128.7  2.7x
  2d-128        |   406.4    201.1  2.0x |   425.9    168.7  2.5x
  2d-roi        |   351.7    113.7  3.1x |   352.2     93.8  3.8x
  3d-qmap       |  1339.1    659.9  2.0x |   952.1    459.9  2.1x
  1d-timeseries |   380.0    106.7  3.6x |   247.3    126.2  2.0x
  2d-512        |  5307.6   3951.2  1.3x |   606.3    401.9  1.5x

  droppable-extras (2d-128 + transform + foreign-dim coord): enc-ded ~270
  (the ~70 us over plain 2d-128 is the _extras_droppable scan)
  fallback abort overhead (dr00 attempt then da00): roi-readback ~83,
  estia-ioq ~162 -- both amortizable via a per-stream eligibility memo,
  which stays correct in both directions because encode_dr00 self-validates.

Eligibility is fused into the encoder rather than a standalone predicate: a
standalone scan of da.coords costs 40-60 us in scipp property accesses alone,
while the encoder touches the same properties anyway.

Run: .venv/bin/python scripts/benchmark_result_schema.py
"""

from __future__ import annotations

import timeit
from dataclasses import dataclass

import flatbuffers
import numpy as np
import scipp as sc
from flatbuffers import number_types as N
from flatbuffers.table import Table
from streaming_data_types import dataarray_da00

from ess.livedata.kafka.scipp_da00_compat import da00_to_scipp, scipp_to_da00

SOURCE = 'job-1234/detector_view'
TS = 1_000_000_000_000

# ---------------------------------------------------------------------------
# Payload cases from the wire inventory (issue #1057 comments).
# ---------------------------------------------------------------------------


def _with_times(da: sc.DataArray, window: bool = False) -> sc.DataArray:
    coords = {
        'start_time': sc.scalar(1_000_000_000_000, unit='ns'),
        'end_time': sc.scalar(1_001_000_000_000, unit='ns'),
    }
    if window:
        coords['time'] = coords['start_time']
    return da.assign_coords(coords)


def _edges(dim: str, n: int, unit: str = 'mm') -> sc.Variable:
    return sc.linspace(dim, 0.0, 1.0, num=n + 1, unit=unit)


def make_cases() -> dict[str, sc.DataArray]:
    rng = np.random.default_rng(42)

    def counts(shape: tuple[int, ...], dims: tuple[str, ...], variances=False):
        values = rng.random(shape)
        return sc.array(
            dims=dims,
            values=values,
            variances=values.copy() if variances else None,
            unit='counts',
        )

    cases = {}
    # Monitor / detector scalar totals (window output: all three time coords).
    cases['0d-total'] = _with_times(
        sc.DataArray(counts((), ()), name='counts_total'), window=True
    )
    # Monitor histogram: 1-D, bin-edge coord, name==dim.
    cases['1d-monitor'] = _with_times(
        sc.DataArray(
            counts((100,), ('time_of_arrival',)),
            coords={'time_of_arrival': _edges('time_of_arrival', 100, 'ns')},
            name='cumulative',
        )
    )
    # Reduction I(Q)-style: 1-D with errors.
    cases['1d-1k+err'] = _with_times(
        sc.DataArray(
            counts((1000,), ('Q',), variances=True),
            coords={'Q': _edges('Q', 1000, '1/angstrom')},
            name='i_of_q',
        )
    )
    # Logical detector view: 2-D image, NO spatial coords (bare fold dims).
    cases['2d-nocoord'] = _with_times(
        sc.DataArray(counts((64, 448), ('tube', 'pixel')), name='cumulative')
    )
    # Geometric detector view: 2-D image with x/y bin-edge coords.
    cases['2d-128'] = _with_times(
        sc.DataArray(
            counts((128, 128), ('y', 'x')),
            coords={'y': _edges('y', 128), 'x': _edges('x', 128)},
            name='cumulative',
        )
    )
    # ROI spectra: 2-D, int32 midpoint coord + float64 edges.
    cases['2d-roi'] = _with_times(
        sc.DataArray(
            counts((4, 100), ('roi', 'time_of_arrival')),
            coords={
                'roi': sc.array(
                    dims=['roi'], values=np.arange(4, dtype=np.int32), unit=None
                ),
                'time_of_arrival': _edges('time_of_arrival', 100, 'ns'),
            },
            name='roi_spectra_current',
        ),
        window=True,
    )
    # BIFROST qmap cut: 3-D, midpoints + edges + edges.
    cases['3d-qmap'] = _with_times(
        sc.DataArray(
            counts((5, 100, 100), ('arc', 'ΔE', 'Q'), variances=True),
            coords={
                'arc': sc.array(dims=['arc'], values=np.arange(5.0), unit='meV'),
                'ΔE': _edges('ΔE', 100, 'meV'),
                'Q': _edges('Q', 100, '1/angstrom'),
            },
            name='cut_data',
        )
    )
    # Timeseries delta: float64 values, datetime64[ns] time axis, no scalar times.
    time = sc.epoch(unit='ns') + sc.array(
        dims=['time'], values=(TS + np.arange(5) * 10**9), unit='ns'
    )
    cases['1d-timeseries'] = sc.DataArray(
        sc.array(dims=['time'], values=rng.random(5), unit='K'),
        coords={'time': time},
        name='delta',
    )
    # Large image: bandwidth-bound regime.
    cases['2d-512'] = _with_times(
        sc.DataArray(
            counts((512, 512), ('y', 'x')),
            coords={'y': _edges('y', 512), 'x': _edges('x', 512)},
            name='cumulative',
        )
    )
    return cases


def make_droppable_extras_case() -> sc.DataArray:
    """Geometric-view-style payload: fits dr00 only because its extra coords
    (scalar transform, foreign-dim bounds) are exactly those the da00 round
    trip drops today. Exercises the _extras_droppable slow path."""
    da = make_cases()['2d-128'].copy()
    da.coords['detector_transform'] = sc.spatial.translation(
        value=[1.0, 2.0, 3.0], unit='m'
    )
    da.coords['wavelength'] = sc.array(
        dims=['wavelength'], values=[1.0, 2.0], unit='angstrom'
    )
    return da


def make_fallback_cases() -> dict[str, sc.DataArray]:
    rng = np.random.default_rng(7)
    # ROI rectangle readback: payload lives in x/y/roi_index coords.
    n = 4
    roi = sc.DataArray(
        sc.array(dims=['bounds'], values=np.zeros(2 * n, dtype=np.int32), unit=''),
        coords={
            'x': sc.array(dims=['bounds'], values=rng.random(2 * n), unit='mm'),
            'y': sc.array(dims=['bounds'], values=rng.random(2 * n), unit='mm'),
            'roi_index': sc.array(
                dims=['bounds'], values=np.repeat(np.arange(n, dtype=np.int32), 2)
            ),
        },
        name='roi_rectangle',
    )
    # ESTIA reflectometry: extra scalar float geometry coords.
    values = rng.random(100)
    estia = _with_times(
        sc.DataArray(
            sc.array(dims=['Q'], values=values, variances=values, unit='counts'),
            coords={
                'Q': _edges('Q', 100, '1/angstrom'),
                'sample_rotation': sc.scalar(0.4, unit='deg'),
                'detector_rotation': sc.scalar(0.8, unit='deg'),
                'sample_size': sc.scalar(10.0, unit='mm'),
                'L1': sc.scalar(8.5, unit='m'),
            },
            name='i_of_q',
        )
    )
    return {'roi-readback': roi, 'estia-ioq': estia}


def encode_with_fallback(da: sc.DataArray) -> bytes:
    try:
        return encode_dr00(da)
    except Ineligible:
        return encode_da00(da)


# ---------------------------------------------------------------------------
# da00 baseline (production pipeline).
# ---------------------------------------------------------------------------


def encode_da00(da: sc.DataArray) -> bytes:
    return dataarray_da00.serialise_da00(
        source_name=SOURCE, timestamp_ns=TS, data=scipp_to_da00(da)
    )


def decode_da00(buf: bytes) -> sc.DataArray:
    return da00_to_scipp(dataarray_da00.deserialise_da00(buf).data)


# ---------------------------------------------------------------------------
# dr00 v2 codec.
# ---------------------------------------------------------------------------

_NP_TO_DTYPE = {
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
_DTYPE_TO_NP = {v: k for k, v in _NP_TO_DTYPE.items()}
DT_DATETIME64 = 11  # int64 ns-since-epoch payload; coord unit holds epoch unit

TIME_COORDS = ('time', 'start_time', 'end_time')


class Ineligible(Exception):
    """Payload does not fit dr00; caller falls back to da00."""


def _extras_droppable(da: sc.DataArray) -> bool:
    """Slow path, only entered when unaccounted coords exist: True if every
    extra coord is one that today's da00 round trip drops anyway
    (transform/vector3 dtypes at encode, foreign-dim coords at decode)."""
    dims = set(da.dims)
    for name, coord in da.coords.items():
        if name in TIME_COORDS and coord.ndim == 0:
            continue
        if name in dims and coord.dims == (name,):
            continue
        if coord.shape != coord.values.shape:
            continue  # transform/vector3 dtype: dropped by da00 encode today
        if not set(coord.dims).issubset(dims):
            continue  # foreign-dim coord: dropped by da00 decode today
        return False
    return True


def encode_dr00(da: sc.DataArray) -> bytes:
    """Encode to dr00, raising Ineligible for payloads that need da00.

    Eligibility is checked inline: name-keyed lookups cover the regular
    coords, and a full coords scan runs only when unaccounted coords exist.
    """
    values = da.values
    ndim = values.ndim
    if ndim > 3:
        raise Ineligible
    dtype_code = _NP_TO_DTYPE.get(values.dtype)
    if dtype_code is None:
        raise Ineligible
    b = flatbuffers.Builder(1024)
    coords = da.coords
    accounted = 0
    off_source = b.CreateString(SOURCE)
    off_name = b.CreateString(da.name)
    off_unit = b.CreateString(str(da.unit)) if da.unit is not None else None
    off_signal = b.CreateNumpyVector(np.ravel(values).view(np.uint8))
    off_errors = (
        b.CreateNumpyVector(np.ravel(np.sqrt(da.variances)).view(np.uint8))
        if da.variances is not None
        else None
    )
    off_shape = b.CreateNumpyVector(np.asarray(values.shape, dtype=np.int64))
    dim_offs = []
    for i in range(ndim):
        dim = da.dims[i]
        coord = coords.get(dim)
        if coord is None:
            dim_offs.append((b.CreateString(dim), None, 0, None))
            continue
        if coord.dims != (dim,):
            raise Ineligible  # e.g. 2-D coord along this dim
        accounted += 1
        if coord.dtype == sc.DType.datetime64:
            cvals = coord.values.view(np.int64)
            dim_offs.append(
                (
                    b.CreateString(dim),
                    b.CreateNumpyVector(cvals.view(np.uint8)),
                    DT_DATETIME64,
                    b.CreateString(str(coord.unit)),
                )
            )
        else:
            cvals = coord.values
            cdt = _NP_TO_DTYPE.get(cvals.dtype)
            if cdt is None:
                raise Ineligible
            dim_offs.append(
                (
                    b.CreateString(dim),
                    b.CreateNumpyVector(np.ravel(cvals).view(np.uint8)),
                    cdt,
                    b.CreateString(str(coord.unit)) if coord.unit is not None else None,
                )
            )
    b.StartObject(23)
    b.PrependUOffsetTRelativeSlot(0, off_source, 0)
    b.PrependInt64Slot(1, TS, 0)
    if (t := coords.get('time')) is not None and t.ndim == 0:
        b.PrependInt64Slot(2, int(t.value), None)
        accounted += 1
    if (t := coords.get('start_time')) is not None:
        b.PrependInt64Slot(3, int(t.value), None)
        accounted += 1
    if (t := coords.get('end_time')) is not None:
        b.PrependInt64Slot(4, int(t.value), None)
        accounted += 1
    if len(coords) != accounted and not _extras_droppable(da):
        raise Ineligible
    b.PrependUOffsetTRelativeSlot(5, off_name, 0)
    if off_unit is not None:
        b.PrependUOffsetTRelativeSlot(6, off_unit, 0)
    b.PrependInt8Slot(7, _NP_TO_DTYPE[values.dtype], 0)
    b.PrependUOffsetTRelativeSlot(8, off_signal, 0)
    if off_errors is not None:
        b.PrependUOffsetTRelativeSlot(9, off_errors, 0)
    b.PrependUOffsetTRelativeSlot(10, off_shape, 0)
    for i, (od, oc, cdt, ou) in enumerate(dim_offs):
        base = 11 + 4 * i
        b.PrependUOffsetTRelativeSlot(base, od, 0)
        if oc is not None:
            b.PrependUOffsetTRelativeSlot(base + 1, oc, 0)
            b.PrependInt8Slot(base + 2, cdt, 0)
        if ou is not None:
            b.PrependUOffsetTRelativeSlot(base + 3, ou, 0)
    b.Finish(b.EndObject(), file_identifier=b'dr00')
    return bytes(b.Output())


def _field(tab: Table, slot: int) -> int:
    return tab.Offset(4 + 2 * slot)


def _string(tab: Table, slot: int) -> str | None:
    o = _field(tab, slot)
    return tab.String(o + tab.Pos).decode() if o else None


def _opt_int64(tab: Table, slot: int) -> int | None:
    o = _field(tab, slot)
    return tab.Get(N.Int64Flags, o + tab.Pos) if o else None


def _u8vec(tab: Table, slot: int) -> np.ndarray | None:
    o = _field(tab, slot)
    return tab.GetVectorAsNumpy(N.Uint8Flags, o) if o else None


def decode_dr00(buf: bytes) -> sc.DataArray:
    n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, bytearray(buf), 0)
    tab = Table(buf, n)
    dtype = _DTYPE_TO_NP[tab.Get(N.Int8Flags, _field(tab, 7) + tab.Pos)]
    signal = _u8vec(tab, 8).view(dtype)
    errors = _u8vec(tab, 9)
    if errors is not None:
        errors = errors.view(dtype)
    so = _field(tab, 10)
    shape = tab.GetVectorAsNumpy(N.Int64Flags, so)
    ndim = len(shape)
    dims = []
    coords: dict[str, sc.Variable] = {}
    for i in range(ndim):
        base = 11 + 4 * i
        dim = _string(tab, base)
        dims.append(dim)
        cvals = _u8vec(tab, base + 1)
        if cvals is None:
            continue
        cdt = tab.Get(N.Int8Flags, _field(tab, base + 2) + tab.Pos)
        cunit = _string(tab, base + 3)
        if cdt == DT_DATETIME64:
            coords[dim] = sc.epoch(unit=cunit) + sc.array(
                dims=[dim], values=cvals.view(np.int64), unit=cunit
            )
        else:
            coords[dim] = sc.array(
                dims=[dim], values=cvals.view(_DTYPE_TO_NP[cdt]), unit=cunit
            )
    unit = _string(tab, 6)
    if ndim == 0:
        signal = signal.reshape(())
        errors = errors.reshape(()) if errors is not None else None
    elif ndim > 1:
        signal = signal.reshape(shape)
        errors = errors.reshape(shape) if errors is not None else None
    data = sc.array(
        dims=dims,
        values=signal,
        variances=errors**2 if errors is not None else None,
        unit=unit,
    )
    if (t := _opt_int64(tab, 2)) is not None:
        coords['time'] = sc.scalar(t, unit='ns')
    if (t := _opt_int64(tab, 3)) is not None:
        coords['start_time'] = sc.scalar(t, unit='ns')
    if (t := _opt_int64(tab, 4)) is not None:
        coords['end_time'] = sc.scalar(t, unit='ns')
    return sc.DataArray(data, coords=coords, name=_string(tab, 5))


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def bench(fn, *args, repeat=7) -> float:
    t = timeit.Timer(lambda: fn(*args))
    number, _ = t.autorange()
    return min(t.repeat(repeat=repeat, number=number)) / number * 1e6


@dataclass
class Row:
    case: str
    enc: float
    enc_ded: float
    dec: float
    dec_ded: float
    size: int
    size_ded: int


def main() -> None:
    rows = []
    for case, da in make_cases().items():
        buf = encode_da00(da)
        dr00_buf = encode_dr00(da)
        # dr00 round trip must reproduce the da00 pipeline's output exactly.
        orig = decode_da00(buf)
        got = decode_dr00(dr00_buf)
        if not sc.identical(got, orig):
            raise AssertionError(f'dr00 mismatch for {case}:\n{got}\nvs\n{orig}')
        rows.append(
            Row(
                case=case,
                enc=bench(encode_da00, da),
                enc_ded=bench(encode_dr00, da),
                dec=bench(decode_da00, buf),
                dec_ded=bench(decode_dr00, dr00_buf),
                size=len(buf),
                size_ded=len(dr00_buf),
            )
        )

    print('microseconds/message, best of 7 (scipp <-> bytes round trip)')
    hdr = (
        f'{"case":<14}| {"enc":>7} {"enc-ded":>8} {"x":>5} | '
        f'{"dec":>7} {"dec-ded":>8} {"x":>5} | {"bytes":>7} {"bytes-d":>7}'
    )
    print(hdr)
    print('-' * len(hdr))
    for r in rows:
        print(
            f'{r.case:<14}| {r.enc:>7.1f} {r.enc_ded:>8.1f} '
            f'{r.enc / r.enc_ded:>4.1f}x | '
            f'{r.dec:>7.1f} {r.dec_ded:>8.1f} {r.dec / r.dec_ded:>4.1f}x | '
            f'{r.size:>7} {r.size_ded:>7}'
        )

    # Droppable extras (geometric detector_transform + foreign-dim coord):
    # must still take the fast path and reproduce the da00 round-trip output.
    da = make_droppable_extras_case()
    got = decode_dr00(encode_dr00(da))
    if not sc.identical(got, decode_da00(encode_da00(da))):
        raise AssertionError('droppable-extras mismatch')
    print(
        f'\ndroppable-extras (2d-128 + transform + foreign-dim coord): '
        f'enc-ded {bench(encode_dr00, da):.1f} '
        f'(plain 2d-128 above; overhead = _extras_droppable scan)'
    )

    print('\nfallback cases: encode_with_fallback = dr00 attempt + da00 encode')
    for case, da in make_fallback_cases().items():
        try:
            encode_dr00(da)
            raise AssertionError(f'{case} unexpectedly eligible')
        except Ineligible:
            pass
        plain = bench(encode_da00, da)
        with_fb = bench(encode_with_fallback, da)
        print(
            f'{case:<14}| da00 {plain:>7.1f} | via-fallback {with_fb:>7.1f} '
            f'| abort overhead {with_fb - plain:>5.1f}'
        )


if __name__ == '__main__':
    main()
