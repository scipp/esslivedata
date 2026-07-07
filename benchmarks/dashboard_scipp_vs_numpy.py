# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Scipp contributors (https://github.com/scipp)
"""Benchmark: scipp vs plain numpy for the dashboard's da00 hot path.

The dashboard consumes da00-encoded reduction results from Kafka at
O(100) messages/s and displays O(10) plots per session. Per message it
runs (see ``kafka/scipp_da00_compat.py``, ``dashboard/temporal_buffers.py``):

1. flatbuffer decode (``deserialise_da00``) -- independent of scipp
2. ``da00_to_scipp``  -- build sc.DataArray from da00 variables
3. ``TemporalBuffer.add`` -- metadata check (sc.identical) + slice write

Per displayed plot update it runs (``dashboard/extractors.py``, ``plots.py``,
``scipp_to_holoviews.py``):

4. extractor (latest slice, or window slice + nansum/nanmean)
5. plot prep (is_edges/islinspace/midpoints/nanmin/nanmax, .values)

This script times each stage against a functionally equivalent plain
numpy/Python implementation, for representative payload shapes.

Run:  python benchmarks/dashboard_scipp_vs_numpy.py
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import scipp as sc
from streaming_data_types import dataarray_da00

from ess.livedata.dashboard.extractors import (
    LatestValueExtractor,
    WindowAggregatingExtractor,
)
from ess.livedata.dashboard.temporal_buffers import TemporalBuffer
from ess.livedata.kafka.scipp_da00_compat import da00_to_scipp

# ---------------------------------------------------------------------------
# Plain numpy stand-ins for the scipp-based pipeline
# ---------------------------------------------------------------------------

_DTYPE_MAP = {
    np.dtype('uint8'): np.int32,
    np.dtype('int8'): np.int32,
    np.dtype('uint16'): np.int32,
    np.dtype('int16'): np.int32,
    np.dtype('uint32'): np.int64,
    np.dtype('uint64'): np.float64,
}


@dataclass
class PlainVariable:
    dims: tuple[str, ...]
    values: np.ndarray
    unit: str | None


@dataclass
class PlainDataArray:
    dims: tuple[str, ...]
    values: np.ndarray
    unit: str | None
    coords: dict[str, PlainVariable] = field(default_factory=dict)
    variances: np.ndarray | None = None
    name: str = ''


def da00_to_plain(
    variables: list[dataarray_da00.Variable], *, signal_name: str = 'signal'
) -> PlainDataArray:
    """Mirror of da00_to_scipp built on plain numpy (no scipp)."""
    signal = None
    errors = None
    coords: dict[str, PlainVariable] = {}
    label = None
    for var in variables:
        data = np.asarray(var.data)
        if data.dtype in _DTYPE_MAP:
            data = data.astype(_DTYPE_MAP[data.dtype])
        if var.name == signal_name:
            signal = PlainVariable(tuple(var.axes), data, var.unit)
            label = var.label
        elif var.name == 'errors':
            errors = data
        else:
            coords[var.name] = PlainVariable(tuple(var.axes), data, var.unit)
    coords = {
        name: v for name, v in coords.items() if set(v.dims).issubset(signal.dims)
    }
    return PlainDataArray(
        dims=signal.dims,
        values=signal.values,
        unit=signal.unit,
        coords=coords,
        variances=errors**2 if errors is not None else None,
        name=label or '',
    )


class PlainTemporalBuffer:
    """Numpy ring buffer mirroring TemporalBuffer's steady-state add path.

    Implements: metadata-match check against a reference (dims, unit, and
    exact comparison of non-time coord arrays -- the numpy equivalent of the
    sc.identical() call in TemporalBuffer._metadata_matches), append of a
    single time slice into a preallocated array, and windowed retrieval.
    """

    DEFAULT_MAX_MEMORY = 20 * 1024 * 1024  # match TemporalBuffer

    def __init__(self, max_memory: int = DEFAULT_MAX_MEMORY) -> None:
        self._max_memory = max_memory
        self._max_capacity = 0
        self._buffer: np.ndarray | None = None
        self._variances: np.ndarray | None = None
        self._times: np.ndarray | None = None
        self._size = 0
        self._reference: PlainDataArray | None = None

    def _metadata_matches(self, data: PlainDataArray) -> bool:
        ref = self._reference
        if (
            data.values.shape != ref.values.shape
            or data.unit != ref.unit
            or data.dims != ref.dims
            or data.name != ref.name
        ):
            return False
        if data.coords.keys() != ref.coords.keys():
            return False
        for name, coord in ref.coords.items():
            new = data.coords[name]
            if new.unit != coord.unit or new.dims != coord.dims:
                return False
            if not np.array_equal(new.values, coord.values):
                return False
        return True

    def add(self, data: PlainDataArray, time_ns: int) -> None:
        if self._buffer is None or not self._metadata_matches(data):
            self._reference = data
            # Same memory-derived capacity rule as TemporalBuffer.
            bytes_per_slice = data.values.nbytes + 8
            self._max_capacity = max(1, self._max_memory // bytes_per_slice)
            self._buffer = np.empty(
                (self._max_capacity, *data.values.shape), dtype=data.values.dtype
            )
            self._variances = (
                np.empty_like(self._buffer) if data.variances is not None else None
            )
            self._times = np.empty(self._max_capacity, dtype=np.int64)
            self._size = 0
        if self._size == self._max_capacity:
            drop = max(1, self._max_capacity // 10)
            n = self._size - drop
            self._buffer[:n] = self._buffer[drop : self._size]
            if self._variances is not None:
                self._variances[:n] = self._variances[drop : self._size]
            self._times[:n] = self._times[drop : self._size]
            self._size = n
        self._buffer[self._size] = data.values
        if self._variances is not None:
            self._variances[self._size] = data.variances
        self._times[self._size] = time_ns
        self._size += 1

    def get_latest(self) -> PlainDataArray:
        ref = self._reference
        i = self._size - 1
        return PlainDataArray(
            dims=ref.dims,
            values=self._buffer[i],
            unit=ref.unit,
            coords=ref.coords,
            variances=None if self._variances is None else self._variances[i],
            name=ref.name,
        )

    def extract_window_agg(self, duration_ns: int) -> PlainDataArray:
        times = self._times[: self._size]
        # Median frame interval to shift the cutoff, as in
        # WindowAggregatingExtractor.extract().
        if self._size > 1:
            half_median = int(np.median(np.diff(times)) // 2)
        else:
            half_median = 0
        cutoff = times[-1] - duration_ns + half_median
        start = int(np.searchsorted(times, cutoff, side='left'))
        window = self._buffer[start : self._size]
        ref = self._reference
        # Same rule as WindowAggregation.auto: sum counts, average the rest.
        agg = np.nansum if ref.unit == 'counts' else np.nanmean
        return PlainDataArray(
            dims=ref.dims,
            values=agg(window, axis=0),
            unit=ref.unit,
            coords=ref.coords,
            variances=(
                None
                if self._variances is None
                else agg(self._variances[start : self._size], axis=0)
            ),
            name=ref.name,
        )


# ---------------------------------------------------------------------------
# Payloads
# ---------------------------------------------------------------------------


def make_da00_payload(name: str) -> bytes:
    """Serialize a representative reduced-data message to da00 bytes."""
    rng = np.random.default_rng(seed=42)
    if name == 'timeseries (scalar)':
        variables = [
            dataarray_da00.Variable(
                name='signal', data=np.array(1.23), axes=[], shape=[], unit='K'
            ),
            dataarray_da00.Variable(
                name='time',
                data=np.array(1_700_000_000 * 10**9, dtype=np.int64),
                axes=[],
                shape=[],
                unit='datetime64[ns]',
            ),
        ]
    elif name.startswith('1d'):
        n = int(name.split('/')[1])
        values = rng.poisson(1000.0, size=n).astype(np.float64)
        variables = [
            dataarray_da00.Variable(
                name='signal', data=values, axes=['tof'], shape=[n], unit='counts'
            ),
            dataarray_da00.Variable(
                name='errors',
                data=np.sqrt(values),
                axes=['tof'],
                shape=[n],
                unit='counts',
            ),
            dataarray_da00.Variable(
                name='tof',
                data=np.linspace(0.0, 71e6, n + 1),
                axes=['tof'],
                shape=[n + 1],
                unit='us',
            ),
        ]
    else:  # 2d/<n>
        n = int(name.split('/')[1])
        values = rng.poisson(100.0, size=(n, n)).astype(np.float64)
        variables = [
            dataarray_da00.Variable(
                name='signal',
                data=values,
                axes=['y', 'x'],
                shape=[n, n],
                unit='counts',
            ),
            dataarray_da00.Variable(
                name='x',
                data=np.linspace(-0.5, 0.5, n),
                axes=['x'],
                shape=[n],
                unit='m',
            ),
            dataarray_da00.Variable(
                name='y',
                data=np.linspace(-0.5, 0.5, n),
                axes=['y'],
                shape=[n],
                unit='m',
            ),
        ]
    return dataarray_da00.serialise_da00(
        source_name='benchmark', timestamp_ns=0, data=variables
    )


PAYLOADS = ['timeseries (scalar)', '1d/1000', '1d/10000', '2d/128', '2d/512']


# ---------------------------------------------------------------------------
# Timing helper
# ---------------------------------------------------------------------------


def bench(func, *, min_time_s: float = 0.2, repeats: int = 3) -> float:
    """Return best-of-repeats time per call in microseconds."""
    # Warm up and estimate a rep count targeting min_time_s per measurement.
    func()
    t0 = time.perf_counter()
    func()
    dt = time.perf_counter() - t0
    n = max(1, int(min_time_s / max(dt, 1e-9)))
    best = float('inf')
    for _ in range(repeats):
        t0 = time.perf_counter()
        for _ in range(n):
            func()
        best = min(best, (time.perf_counter() - t0) / n)
    return best * 1e6


# ---------------------------------------------------------------------------
# Benchmark stages
# ---------------------------------------------------------------------------


def with_time_coord(da: sc.DataArray, t_ns: int) -> sc.DataArray:
    # Production time coords are int64 ns since epoch; FullHistoryExtractor
    # converts to datetime64 only for display. (TemporalBuffer._trim_to_timespan
    # raises a DTypeError on datetime64 time coords: float64 timespan cannot be
    # subtracted from datetime64.)
    return da.assign_coords(time=sc.scalar(t_ns, unit='ns', dtype='int64'))


def run(payload_name: str) -> dict[str, tuple[float, float]]:
    """Run all stages for one payload; returns {stage: (scipp_us, numpy_us)}."""
    results: dict[str, tuple[float, float]] = {}
    buf = make_da00_payload(payload_name)

    decoded = dataarray_da00.deserialise_da00(buf)
    variables = decoded.data

    results['flatbuffer decode (shared)'] = (
        bench(lambda: dataarray_da00.deserialise_da00(buf)),
        float('nan'),
    )
    results['build DataArray'] = (
        bench(lambda: da00_to_scipp(variables)),
        bench(lambda: da00_to_plain(variables)),
    )

    # --- buffer add (steady state: metadata matches, slice write) ----------
    da = da00_to_scipp(variables)
    plain = da00_to_plain(variables)
    # Cross-check the two deserialization paths agree.
    np.testing.assert_array_equal(np.asarray(da.values), plain.values)
    if da.variances is not None:
        np.testing.assert_allclose(np.asarray(da.variances), plain.variances)
    if 'time' in da.coords:  # timeseries payload carries its own time coord
        da = da.drop_coords('time')
        del plain.coords['time']

    # O(100) msg/s arrive spread over many streams; a single stream/buffer
    # sees ~1 Hz updates (per-job accumulated results).
    period_ns = 10**9  # 1 Hz per stream
    sc_buffer = TemporalBuffer()
    sc_buffer.set_required_timespan(60.0)
    clock = {'t': 1_700_000_000 * 10**9}

    def scipp_add() -> None:
        clock['t'] += period_ns
        sc_buffer.add(with_time_coord(da, clock['t']))

    np_buffer = PlainTemporalBuffer()

    def numpy_add() -> None:
        clock['t'] += period_ns
        np_buffer.add(plain, clock['t'])

    results['buffer add (steady state)'] = (bench(scipp_add), bench(numpy_add))

    # Refill both buffers with a deterministic 10 min history for extraction
    # (both are capped at the same 20 MB memory limit, so large payloads hold
    # correspondingly fewer time slices).
    sc_buffer.clear()
    np_buffer2 = PlainTemporalBuffer()
    t0 = 1_700_000_000 * 10**9
    n_history = 600
    for i in range(n_history):
        sc_buffer.add(with_time_coord(da, t0 + i * period_ns))
        np_buffer2.add(plain, t0 + i * period_ns)

    # --- extract latest -----------------------------------------------------
    latest = LatestValueExtractor()
    results['extract latest'] = (
        bench(lambda: latest.extract(sc_buffer.get())),
        bench(np_buffer2.get_latest),
    )

    # --- window aggregation (60 s window over 1 Hz data) --------------------
    window = WindowAggregatingExtractor(window_duration_seconds=60.0)
    # Cross-check that both implementations compute the same aggregate.
    sc_result = window.extract(sc_buffer.get())
    np_result = np_buffer2.extract_window_agg(60 * 10**9)
    np.testing.assert_allclose(np.asarray(sc_result.values), np_result.values)
    results['extract 60s window + agg'] = (
        bench(lambda: window.extract(sc_buffer.get())),
        bench(lambda: np_buffer2.extract_window_agg(60 * 10**9)),
    )

    # --- plot prep -----------------------------------------------------------
    if payload_name.startswith('1d'):
        hist = da00_to_scipp(variables)

        def scipp_plot_prep() -> None:
            # What HvConverter1d + log-scale autoscale do per update.
            edges = hist.coords.is_edges(hist.dim)
            mids = sc.midpoints(hist.coords[hist.dim]) if edges else None
            _ = (
                mids.values if mids is not None else None,
                hist.values,
                sc.stddevs(hist.data).values,
                float(hist.data.nanmin().value),
                float(hist.data.nanmax().value),
            )

        p = da00_to_plain(variables)
        p_edges = p.coords['tof'].values

        def numpy_plot_prep() -> None:
            edges = p_edges.shape[0] == p.values.shape[0] + 1
            mids = 0.5 * (p_edges[1:] + p_edges[:-1]) if edges else None
            _ = (
                mids,
                p.values,
                np.sqrt(p.variances),
                float(np.nanmin(p.values)),
                float(np.nanmax(p.values)),
            )

        results['plot prep (1d hist)'] = (
            bench(scipp_plot_prep),
            bench(numpy_plot_prep),
        )
    elif payload_name.startswith('2d'):
        img = da00_to_scipp(variables)

        def scipp_plot_prep() -> None:
            # _all_coords_evenly_spaced + _get_midpoints + log-clim autoscale.
            _ = all(sc.islinspace(img.coords[dim]) for dim in img.dims)
            x = img.coords[img.dims[1]].values
            y = img.coords[img.dims[0]].values
            _ = (
                x,
                y,
                img.values,
                float(img.data.nanmin().value),
                float(img.data.nanmax().value),
            )

        p = da00_to_plain(variables)

        def numpy_plot_prep() -> None:
            def islinspace(v: np.ndarray) -> bool:
                d = np.diff(v)
                return bool(np.all(np.isclose(d, d[0])))

            _ = all(islinspace(p.coords[dim].values) for dim in p.dims)
            _ = (
                p.coords['x'].values,
                p.coords['y'].values,
                p.values,
                float(np.nanmin(p.values)),
                float(np.nanmax(p.values)),
            )

        results['plot prep (2d image)'] = (
            bench(scipp_plot_prep),
            bench(numpy_plot_prep),
        )

    return results


def main() -> None:
    print(f"scipp {sc.__version__}, numpy {np.__version__}")
    all_results: dict[str, dict[str, tuple[float, float]]] = {}
    for payload in PAYLOADS:
        print(f"\n=== payload: {payload} ===")
        res = run(payload)
        all_results[payload] = res
        for stage, (t_sc, t_np) in res.items():
            if np.isnan(t_np):
                print(f"  {stage:<32} {t_sc:9.1f} us")
            else:
                ratio = t_sc / t_np if t_np else float('inf')
                print(
                    f"  {stage:<32} scipp {t_sc:9.1f} us   "
                    f"numpy {t_np:9.1f} us   x{ratio:5.1f}"
                )

    # ------------------------------------------------------------------
    # Scenario: 100 msg/s ingest, 10 displayed plots updating at 1 Hz.
    # Ingest mix: 60 timeseries, 30 1d/1000, 10 2d/512 messages per second.
    # ------------------------------------------------------------------
    mix = {'timeseries (scalar)': 60, '1d/1000': 30, '2d/512': 10}
    display = {'1d/1000': 6, '2d/512': 4}

    def ingest_cost(col: int) -> float:
        total = 0.0
        for payload, rate in mix.items():
            r = all_results[payload]
            per_msg = r['build DataArray'][col] + r['buffer add (steady state)'][col]
            total += rate * per_msg
        return total * 1e-6  # us -> s per second of wall time

    decode = (
        sum(
            rate * all_results[p]['flatbuffer decode (shared)'][0]
            for p, rate in mix.items()
        )
        * 1e-6
    )

    def display_cost(col: int) -> float:
        total = 0.0
        for payload, n_plots in display.items():
            r = all_results[payload]
            prep = next(v for k, v in r.items() if k.startswith('plot prep'))
            per_update = r['extract 60s window + agg'][col] + prep[col]
            total += n_plots * per_update
        return total * 1e-6

    print("\n=== scenario: 100 msg/s ingest (60 scalar + 30 1d/1000 + 10 2d/512),")
    print("    10 plots at 1 Hz (6x 1d windowed, 4x 2d windowed) ===")
    print(f"  flatbuffer decode (shared)   {decode * 100:6.2f} % of one core")
    for label, col in [('scipp', 0), ('numpy', 1)]:
        print(
            f"  {label:<6} ingest {ingest_cost(col) * 100:6.2f} %   "
            f"display {display_cost(col) * 100:6.2f} %   "
            f"total {(ingest_cost(col) + display_cost(col)) * 100:6.2f} % of one core"
        )


if __name__ == '__main__':
    main()
