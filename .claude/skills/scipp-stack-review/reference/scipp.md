# scipp — reviewer reference

scipp is a labeled, unit-aware N-D array library (think xarray plus physical
units, propagated uncertainties, and event/binned data). Its semantics differ
from numpy/xarray in ways that produce silent bugs downstream.

## Data structures

- `Variable`: N-D array with `.dims` (string labels), `.unit`, `.values`,
  optional `.variances`. No coords. First dim is outermost (C order).
- `DataArray`: `.data` (Variable) + `.coords` + `.masks` (dicts of
  Variables). Coords are independent variables and carry their own units.
- `Dataset`: dict of DataArrays sharing aligned coords; insertion with
  mismatched coords/extents fails. Masks live on items, not the Dataset.
- `DataGroup`: dict of arbitrary objects (nested allowed); no coord or dim
  consistency enforced; ops broadcast over items and fail on non-scipp items.
- Scalar (0-D) is distinct from length-1 array. `.value`/`.variance`
  (singular) raise unless 0-D; use `.values`/`.variances` otherwise.
- `attrs` was **removed** (24.12.0). The replacement is unaligned coords:
  `da.coords.set_aligned('x', False)`. Any code (or review suggestion) using
  `da.attrs` targets a dead API.

## Units

- Add/subtract/compare require identical units (`UnitError` otherwise);
  multiply/divide combine units. Trig requires `rad`/`deg`.
- Numeric default unit is `dimensionless` (`sc.units.one`); strings get
  unit `None`.
- `unit=None` means "not a quantity" — reviewers reject overloading it to
  mean "unknown/mixed unit". For index-like data use `unit=None` unless it
  enters arithmetic with physical quantities, then `dimensionless`
  (essreduce#211 convention).
- Convert with `da.to(unit='mm')` / `sc.to_unit(...)`, never manual factors.
  `.to()` on ints picks a safe intermediate dtype; still watch precision.
- External boundaries (Kafka messages, JSON, UI) often carry no units —
  require the code to recover units from an authoritative source (e.g.
  parsed NeXus file) instead of omitting them, and require unit-explicit
  API names/factories (`from_ns`, `to_seconds`) over bare numbers.

## Variances

- Propagation assumes **uncorrelated** operands.
- Broadcasting an operand with variances raises `sc.VariancesError` on
  purpose (correlated duplicates → later `sum`/`mean` underestimates
  uncertainty; see doi:10.3233/JNR-220049). See essreduce reference for the
  sanctioned workaround.
- `sc.values(x)`, `sc.variances(x)`, `sc.stddevs(x)` are the unit-safe ways
  to strip/inspect uncertainty. NeXus files store stddevs; scipp holds
  variances — watch for missing square/sqrt at file boundaries.
- Domain-edge cases (0/0, asin/acos outside [-1,1]) should yield NaN
  variance, not abs() fudges (scipp#3854).
- Statistical variance ≠ systematic resolution: storing instrument
  resolution (e.g. SAS Q-resolution) in `.variances` conflates the two and
  corrupts downstream fits (scippnexus#63).

## Slicing and indexing

- `da['x', 0]` positional; `da['x', sc.scalar(2023)]` label-based. Mixing
  these up is a classic silent-wrong-selection bug. `Variable.__index__`
  forbids float positional indexing, but a *label* float scalar is rounded
  at Python-literal level before lookup (`11.9999999999999999 == 12.0`).
- Integer slice (`da['x', 0]`) drops the dim and demotes that dim's coords to
  **unaligned**; range slice (`da['x', 0:1]`) keeps dim and alignment. Hence
  `a - a['x', 1]` works while `a - a['x', 1:2]` raises on coord mismatch.
- Positional/label slices are **views** — in-place ops on them mutate the
  parent. Integer-array (`da['x', [1,2,5]]`) and boolean indexing always
  return **copies**. Boolean condition must be 1-D (no auto-flatten).
- Exact float label lookup raises `IndexError` (no fuzzy match) — use an
  interval `da['x', lo*unit:hi*unit]` (half-open, `[lo, hi)`). For bin-edge
  coords a single value selects the containing bin. Label indexing requires
  1-D monotonic coords.
- Slicing marks metadata not depending on the slice dim read-only, and
  assigning to a slice fails if it would inconsistently update a shared
  mask. Negative step is unsupported.

## Copy vs view, ownership

- `sc.array(values=np_arr)` copies the numpy buffer. But inserting a
  Variable into `coords`/`masks`/`.data`, or a DataArray into a Dataset,
  **shares ownership**: mutate the source later and the container changes.
  `ds['a']` returns a view. `.copy()` (deep by default) breaks sharing.
- `fold`/`flatten` return views; `concat` copies; `transform_coords` returns
  a shallow copy with added coords.
- In-place ops (`+=`, `out=`) cannot broadcast the LHS or change its
  dtype/shape; RHS-only broadcast.
- Don't add `.copy()` before ops that already copy (`bin`, `concat`) —
  wasted allocation.

## Masks

- Dict of boolean Variables; each mask has its own dims (a 1-D mask on 3-D
  data is fine and preferred over broadcasting it). `True` = excluded.
- Binary ops: union of masks; same names OR-combined.
- Reductions over a dim **apply and drop** masks that depend on that dim;
  independent masks pass through. `mean` excludes masked from the count;
  `sum` treats them as 0; `bin` treats masked source bins as empty; `rebin`
  as zero.
- Removing: `del da.masks[name]`, `da.drop_masks(...)`.

## Binned (event) data

- Binned ≠ histogrammed: binned data holds per-bin event tables; `hist` /
  `bins.sum()` collapses to dense. `da.bin(x=edges)` reorders a **copy** of
  the event table; `da.hist(x=...)` on binned data re-histograms at any
  resolution; `rebin` resamples dense histograms.
- Event-level metadata: `da.bins.coords`, `da.bins.masks`; bin-level:
  `da.coords`, `da.masks`. Both coexist — check which level an operation
  should target.
- Slicing binned data by an outer dim drops out-of-range events;
  `da.bins['coord', lo:hi]` filters by event coord but reprocesses all
  events per call — for repeated extraction, `bin` once then slice, or
  `da.group('label')`.
- Check for binned data with `x.is_binned` (`x.bins is not None` is
  deprecated). `da.bins` of a non-binned array is `None` today.
- `sc.bins_like(da, var)` broadcasts a per-bin value to events.

## transform_coords, lookup, groupby

- `da.transform_coords(['out'], graph=...)`: consumed inputs become
  unaligned (kept unless `keep_inputs=False`); intermediates kept unaligned;
  dim-rename happens automatically when unambiguous (`rename_dims=False` to
  disable). Works through to event coords of binned data.
- `sc.lookup(series, mode='previous')` + `transform_coords` is the idiom for
  mapping event timestamps to time-series log values — flag hand-rolled
  searchsorted/interp equivalents.
- `groupby` groups along one dim only and is slow for millions of tiny
  groups; float labels need `bins=`. For event data prefer `da.group(...)` /
  `da.bin(...)`.

## dtype and precision

- dtype deduced from input; ints stay ints — truncation and overflow are
  real ((a+b)/2 patterns; prefer forms that cannot overflow). Derived float
  results should be float64; float32 accumulation loses counts
  (scipp#3767: "float32 is too lossy... especially when it propagates").
- Datetimes: `sc.datetime`/`sc.datetimes`, epoch-based, int64 ns at ESS.
  Keep ns precision end-to-end; convert at the edge.
- Vectors/rotations have dedicated dtypes (`sc.vectors`, `.fields.x`).
- Comparisons: `sc.isclose`/`allclose` for computed floats, but **exact**
  comparison where exactness is the contract (grid-edge clamping — a
  reviewer explicitly rejected `np.isclose` there). `sc.identical` respects
  coord alignment flags.

## Anti-patterns (wrong → right)

```python
# Unit/variance-dropping numpy round-trip
np.sqrt(da.values)                 # → sc.sqrt(da)
a.values / b.values                # → a / b  (also handles dim broadcast)
da.values * 1000  # "m to mm"      # → da.to(unit='mm')

# Manual histogramming
np.histogram(da.coords['x'].values, weights=da.values, bins=40)
# → da.hist(x=sc.linspace('x', lo, hi, num_edges, unit=...))

# Python loops over slices (very slow; scipp docs have a section on this)
for i in range(n-1): img['t', i] -= img['t', i+1]
# → img['t', :-1] -= img['t', 1:]        (or fold/flatten helper dims)

# Positional index where label intended
da['year', 2023]                   # → da['year', sc.scalar(2023)]

# Mutating a view / shared buffer unintentionally
sub = da['x', :2]; sub += 1        # mutates da → use .copy() if independence needed
da.coords['x'] = x; x += 1         # mutates da's coord → insert x.copy()

# Broadcasting variances "fixed" by stripping them
det / sc.values(norm)              # silently drops error term
# → ess.reduce.uncertainty.broadcast_uncertainties(norm, prototype=det,
#     mode=UncertaintyBroadcastMode.upper_bound), then divide

# Removed/deprecated APIs
da.attrs['x']                      # → da.coords['x'] + set_aligned(..., False)
da.bins is not None                # → da.is_binned
sc.Variable(dims=..., values=...)  # → sc.array / sc.scalar / sc.zeros ...
```

## Idiom cheat-sheet

- Create: `sc.array/scalar/zeros/ones/arange/linspace/index/datetimes/vectors`.
  Prefer `sc.zeros` over `sc.empty` unless full overwrite is provable.
- Convert: `x.to(unit=..., dtype=...)`; strip uncertainty: `sc.values(x)`.
- Histogram: `da.hist(x=edges)`; keep events: `da.bin(x=edges)`; resample
  dense: `sc.rebin`; group discrete: `da.group('k')` / `sc.groupby`.
- Coord transform: `da.transform_coords([...], graph=...)`;
  time-series lookup: `sc.lookup(...)`.
- Select by value: `da['x', lo*u:hi*u]`; gather: `da[indices_var]`.
- Reshape: `fold`/`flatten` (views) instead of loops.
- Concat/stack: `sc.concat([a, b], dim)` (existing dim concatenates, new dim
  stacks).
- Test equality: `sc.testing.assert_identical` (checks unit + dtype);
  `sc.isclose/allclose` for numerics.
