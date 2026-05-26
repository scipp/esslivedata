# 940 — Timeseries downsampling and update throttling

Addresses issue #940: long-running timeseries plots (~1 day at 1 Hz = ~100k points)
cause server-side lag because `Plotter.compute()` rebuilds the full `hv.Curve` and
ships it through `pipe.send` on every tick.

## Decision

Three plot-config knobs on the timeseries plotter, all time-based:

| knob | default | meaning |
|---|---|---|
| `period_seconds` | 1 | plot sample period **and** update period |
| `recent_seconds` | 3600 | trailing window kept at `period_seconds` resolution |
| `floor_period_seconds` | 300 | indefinite older-data band at this coarser period; 0 drops older data |

Total point count is approximately
`recent_seconds / period_seconds + (run_duration − recent_seconds) / floor_period_seconds`.

The defaults keep a fresh dashboard against a multi-day 1 Hz f144 stream in the
~4k-point regime forever, regardless of buffer length.

## Where the logic lives

Downsampling is a **plotter-level** concern (presentation policy, not subscription
shape), so `FullHistoryExtractor` stays unchanged. The timeseries plotter:

1. Owns the three knobs.
2. Overrides `compute()` to (a) throttle on data-time progression, (b)
   downsample the per-key DataArrays, then (c) call the standard plot path.

A pure helper `downsample_timeseries(data, period_seconds, recent_seconds,
floor_period_seconds)` is tested in isolation.

## Throttle semantics

Driven by **data time**, not wall-clock. The plotter tracks
`_last_compute_data_time`, the latest time coord seen at the last `compute()`
that actually ran. New `compute()` calls short-circuit when
`(latest_in_new_data − _last_compute_data_time) < period_seconds`.

This means:
- A `period_seconds=10` plot runs `compute()` once every 10 s of data arrival.
- `pipe.send` only fires on those ticks (presenter stays clean otherwise).
- Idle periods with no new data don't trigger any work.

Lag indicator in the plot title updates at the same cadence — acceptable.

## Downsampling algorithm

Per-band uniform stride, two bands:

1. **Recent band**: data with time ≥ `latest − recent_seconds`, bucketed into
   `period_seconds`-wide bins, first sample of each bin kept.
2. **Floor band**: data older than the recent band, bucketed into
   `floor_period_seconds`-wide bins, first sample of each bin kept.
   Skipped if `floor_period_seconds == 0`.

No min/max binning, no LTTB. f144 scalar logs are slow-changing; a missed spike
between two stride samples is a non-issue in the field. Strategy is a
non-breaking future addition if needed.

Bound coords (`start_time`, `end_time`) are 1-D along the concat dim, so
positional selection preserves them automatically.

## Files

New:
- `src/ess/livedata/dashboard/timeseries_downsample.py` — `downsample_timeseries()`
- `tests/dashboard/timeseries_downsample_test.py`

Modified:
- `src/ess/livedata/dashboard/plot_params.py` — `TimeseriesDownsamplingParams`,
  `PlotParamsTimeseries`
- `src/ess/livedata/dashboard/plots.py` — `LinePlotter.from_timeseries_params`,
  conditional `compute()` override path
- `src/ess/livedata/dashboard/plotter_registry.py` — timeseries factory and
  params_factory swap

Untouched: `FullHistoryExtractor`, correlation histograms (they still want
unaltered full-history input).

## Out of scope

- Per-tier resolution beyond two bands (recent + floor).
- Strategy selection (stride vs min/max vs LTTB).
- Upstream throttling at `ToNXlog` / Kafka — `period_seconds` only changes the
  dashboard's behaviour.
- Server-side response to client zoom/pan.
