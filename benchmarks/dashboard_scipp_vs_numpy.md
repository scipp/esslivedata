# Scipp vs plain numpy in the dashboard's da00 hot path

Question: the dashboard deserializes da00 into `scipp.DataArray` and performs
simple operations on it. Scipp calls carry C++-binding overhead — is it
overkill, and would a plain Python/numpy container be faster? Assumed load:
O(100) results/s ingested, O(10) plots displayed per session.

Companion script: `benchmarks/dashboard_scipp_vs_numpy.py`
(`python benchmarks/dashboard_scipp_vs_numpy.py` in an environment with the
package installed). Numbers below are from a 4-core container, scipp 26.3.1,
numpy 2.4.6; treat them as relative indicators, not absolutes.

## Survey: what the dashboard actually does with scipp

Per **incoming message** (runs at the full O(100)/s rate, for every stream,
whether displayed or not):

| Stage | Code | Scipp operations |
| --- | --- | --- |
| flatbuffer decode | `streaming_data_types.dataarray_da00` | none (shared cost) |
| build DataArray | `kafka/scipp_da00_compat.py::da00_to_scipp` | `sc.array` per variable, `errors**2`, `sc.DataArray(...)`, datetime handling (`sc.epoch + ...`) |
| buffer add | `dashboard/temporal_buffers.py::TemporalBuffer.add` | `sc.identical` metadata check (slices + `assign` + `drop_coords` to build the comparison template), slice write `buf['time', i] = data`, occasional trim (`sc.to_unit`, comparison, `argmax`) |

Per **displayed plot update** (O(10) plots, ~1 Hz each):

| Stage | Code | Scipp operations |
| --- | --- | --- |
| buffer get | `TemporalBuffer.get` | dict assembly + `sc.DataArray(...)` |
| extract | `dashboard/extractors.py` | latest: `data['time', -1]`; window: label-based slice, `sc.median` of intervals, `sc.nansum`/`sc.nanmean`, `assign_coords` |
| plot prep | `dashboard/plots.py`, `scipp_to_holoviews.py` | `coords.is_edges`, `sc.islinspace`, `sc.midpoints`, `sc.stddevs`, `nanmin`/`nanmax` (log clim), `.to(dtype=...)`, `sc.where` (log-scale masking), `.values` |
| timeseries only | `dashboard/timeseries_downsample.py`, `FullHistoryExtractor` | epoch/int arithmetic, comparisons, boolean indexing, `sc.concat` |

Everything downstream of plot prep (HoloViews element construction, Bokeh
serialization, browser rendering) is scipp-independent and is by far the
larger cost per update; it is out of scope here.

## Benchmark design

Each stage is timed against a functionally equivalent plain-numpy
implementation (`PlainDataArray` dataclass: values/dims/unit/coords/variances
+ a preallocated ring buffer with the same 20 MB memory cap, the same
metadata-match-per-add semantics, the same `WindowAggregation.auto` rule).
Cross-checks assert both paths produce identical values. Payloads mimic
reduced-data messages: a scalar timeseries sample, 1-D histograms
(1 000 / 10 000 bins, with errors and bin edges), 2-D images (128², 512²,
with x/y coords). Message rate per *stream* is ~1 Hz — the O(100)/s is spread
across many streams/jobs.

## Results (µs per call, best of 3)

| stage | payload | scipp | numpy | ratio |
| --- | --- | ---: | ---: | ---: |
| flatbuffer decode (shared) | scalar / 1d / 2d | 125–193 | — | — |
| build DataArray | scalar | 46 | 3 | ×16 |
| | 1d/1000 | 100 | 5 | ×22 |
| | 2d/512 | 120 | 3 | ×43 |
| buffer add (steady state) | scalar | 60 | 1 | ×56 |
| | 1d/1000 | 104 | 19 | ×5 |
| | 1d/10000 | 159 | 394 | ×0.4 |
| | 2d/512 | 2 720 | 8 450 | ×0.3 |
| extract latest (via temporal buffer) | any | ~110 | ~1 | ×120–150 |
| extract 60 s window + agg | scalar | 459 | 61 | ×8 |
| | 1d/1000 | 635 | 216 | ×3 |
| | 1d/10000 | 1 214 | 2 529 | ×0.5 |
| | 2d/512 | 7 129 | 7 435 | ×1.0 |
| plot prep | 1d/1000 | 59 | 26 | ×2 |
| | 2d/512 | 502 | 470 | ×1.1 |

Scenario totals — 100 msg/s ingest (60 scalar + 30× 1d/1000 + 10× 2d/512),
10 windowed plots updating at 1 Hz:

| | ingest | display | total (% of one core) |
| --- | ---: | ---: | ---: |
| flatbuffer decode (shared) | 1.5 % | — | 1.5 % |
| scipp | 4.1 % | 3.5 % | 7.6 % |
| numpy | 8.6 % | 3.3 % | 11.9 % |

## Interpretation

1. **The binding overhead is real and large in relative terms.** Every scipp
   operation costs ~5–50 µs regardless of array size, so small-array/scalar
   stages are 5–150× slower than numpy. `extract latest` at ×120–150 is the
   extreme case: it rebuilds a DataArray from the buffer and label-slices it,
   all fixed overhead. (In production this path only runs when a temporal
   buffer exists; latest-only subscriptions use `SingleValueBuffer`, where
   both approaches cost ~nothing.)

2. **In absolute terms it does not matter at the stated rates.** The whole
   scipp pipeline costs ~100–250 µs per message. At 100 msg/s plus 10 plot
   updates/s the entire scipp-touching workload is ~8 % of one core — and
   ~1.5 points of that is the flatbuffer decode, which a numpy rewrite keeps.
   The dashboard's actual per-update cost is dominated by
   HoloViews/Bokeh/WebSocket rendering, which dwarfs both columns.

3. **Numpy does not uniformly win.** For the larger payloads (10 k-bin
   histograms, 2-D images) scipp is at par or ahead: its reductions are
   multithreaded and its overlapping slice-copy (buffer compaction) avoids
   the temporary that numpy makes for same-array assignment. The naive-numpy
   scenario total above is *worse* than scipp because the 512² buffer adds
   are copy-bound. A numpy rewrite would want a true ring buffer (modular
   indexing, no compaction copy) to win everywhere — more code to own, and
   TemporalBuffer's current strategy would need redesign either way.

4. **What a rewrite would forfeit.** The hot path leans on scipp semantics
   that a `PlainDataArray` would have to re-implement and test: unit-aware
   arithmetic and conversion (`to_unit`, window cutoffs in mixed units),
   datetime64 handling, label-based slicing, bin-edge detection
   (`coords.is_edges`), variance propagation, and `sc.identical` for
   metadata-change detection. The benchmark's ~250-line stand-in covers only
   the happy path of these.

## Conclusion

Scipp's per-call overhead is measurable (typically 10–50× on small
operations) but the absolute budget at O(100) msg/s in, O(10) plots out is a
few percent of one core either way — replacing scipp with a plain numpy
container would save at most ~2–4 % of a core in the best case, and the
naive port is actually slower for image-sized payloads. Not worth the loss
of units/variances/labeled-dims semantics at current rates.

If rates grow by ~10× (O(1000) small messages/s into one dashboard), the
fixed per-message scipp cost (~100–250 µs) starts to matter (~10–25 % of a
core). Cheaper than a rewrite, the fixed overhead can be reduced within the
current design, in order of measured impact:

- `extract latest` (~110 µs): skip the `TemporalBuffer.get()` DataArray
  rebuild + label slice when only the newest slice is needed (e.g. keep the
  last message alongside the buffer, or a dedicated fast path).
- `TemporalBuffer.add` (~60–100 µs small payloads): the `sc.identical`
  metadata check rebuilds a comparison template (slice + `assign` +
  `drop_coords`) on every add; caching the template or comparing coords
  directly would cut most of it.
- `WindowAggregatingExtractor` (~460 µs scalar): re-derives the median frame
  interval and re-slices per extract; the median could be cached/updated
  incrementally.

## Incidental observations

- `TemporalBuffer._trim_to_timespan` raises `DTypeError` when the `time`
  coord is datetime64 (float64 timespan subtracted from datetime64; it
  converts the timespan with `sc.to_unit` but never to int64). Reduced-data
  streams use int64 time coords so this is normally dormant, but any da00
  input whose `time` coord round-trips as datetime64 (see
  `da00_to_scipp`'s datetime path) would crash once the buffer needs
  trimming. Casting the timespan to int64 for datetime coords would fix it.
- The 20 MB default buffer cap means a 512×512 float64 image buffer holds
  only 10 frames — a 60 s window plot over such a stream silently aggregates
  ~10 s of data. Worth keeping in mind when interpreting windowed 2-D plots.
