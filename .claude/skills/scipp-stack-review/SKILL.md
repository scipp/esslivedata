---
name: scipp-stack-review
description: >
  Domain knowledge for reviewing and writing code in projects built on the
  scipp stack: scipp, scippnexus, sciline, essreduce (ess.reduce), and
  downstream ESS packages (esssans, essdiffraction, esslivedata, ...). Use
  when reviewing a PR or diff, hunting bugs, or writing non-trivial code that
  imports scipp, scippnexus, sciline, or ess.reduce. Covers stack-specific
  anti-patterns and bug classes (units, variances, coords, binned data,
  provider graphs, NeXus loading) that generic Python review misses.
---

# Reviewing code on the scipp stack

This skill distills library semantics and several years of human code-review
history across the scipp organization. The stack is uncommon in training
data, so intuitions from numpy/xarray/pandas often mislead. When reviewing,
check the diff against the bug classes below, then consult the per-library
reference for anything the diff touches:

- [reference/scipp.md](reference/scipp.md) — Variable/DataArray semantics: units,
  variances, coords, masks, slicing, views, binned data.
- [reference/sciline.md](reference/sciline.md) — Pipeline/provider graphs: type-keyed
  DI, generics, map/reduce, mutation and caching semantics.
- [reference/scippnexus.md](reference/scippnexus.md) — NeXus file access: lazy loading,
  class-based lookup, transformations, fallback behavior.
- [reference/essreduce.md](reference/essreduce.md) — Generic reduction workflows,
  uncertainty broadcasting, tof, StreamProcessor.
- [reference/conventions.md](reference/conventions.md) — What human reviewers enforce:
  naming, tests, errors, API surface, dependencies, process.

Findings should name the failure mode concretely (which unit/coord/variance is
wrong, and what result silently corrupts). Maintainers push back on review
suggestions with technical rationale — expect reasoned disagreement, and when
suggesting a "simplification", check it does not change guard semantics.

## Highest-signal bug classes

These recur across all repos. Each has bitten real PRs.

### 1. Variance correlation via broadcasting

scipp propagates variances assuming *uncorrelated* operands and deliberately
raises `VariancesError` when an operand with variances would be broadcast
(the duplicated values are correlated; later reductions would silently
underestimate uncertainties). **Never "fix" this with `.values`, `sc.values()`
or a manual `broadcast()`** — that silently drops or corrupts the error term.
The correct tool is `ess.reduce.uncertainty.broadcast_uncertainties(data,
prototype=..., mode=...)` with `UncertaintyBroadcastMode.upper_bound` as the
default; `drop` only as an explicit user choice. Flag any diff that strips
variances from a normalization term or broadcasts variance-carrying data.

### 2. `.values` / numpy round-trips

`da.values` bypasses units, variances, and dim-label broadcasting. Flag
`np.sqrt(da.values)`, `a.values / b.values`, `np.histogram(...)`, manual unit
factors (`* 1000` instead of `.to(unit='mm')`). The idiomatic form is almost
always a direct scipp op: `sc.sqrt(da)`, `a / b`, `da.hist(x=edges)`,
`da.to(unit=...)`. Exception: intentional hot-path numpy interop — then units
and dtype must be checked/converted explicitly at the boundary first.

### 3. Sciline: silent last-wins keys and shared-pipeline mutation

- The graph is keyed by provider **return type** only. Inserting a second
  provider or param for the same type silently *replaces* the first — no
  error. Plain types (`float`, `str`, `dict`) as keys collide across the
  workflow; require distinct domain `NewType`s / `Scope` subclasses.
- `insert` and `pipeline[Key] = value` **mutate in place**. Mutating a
  pipeline obtained from a library or shared object corrupts other users —
  require `.copy()` first. `map`/`reduce`/`copy` return new objects
  (an unassigned `pl.map(...)` is a no-op).
- **No caching**: two `compute()` calls re-run shared ancestors. Use
  `compute((A, B))` for multiple targets, or pin an intermediate with
  `pl[Key] = computed_value`.
- Sciline ignores default argument values and gives `Optional`/`|` no special
  treatment — a default does not make an input optional.

### 4. Event data vs histograms; bins vs edges

- **Streaming/accumulation: accumulate histograms, never events.** A
  `.hist()` placed *after* an accumulator means per-batch cost and memory grow
  linearly with run time. The `hist` belongs upstream of the accumulator.
- `bin` reorders events (expensive copy), `hist` sums into a dense array,
  `rebin` resamples an existing histogram, `groupby` is single-dim
  split-apply-combine. Re-histogramming binned data at a different resolution
  is `.hist(x=...)`, not another `bin`.
- Off-by-one: N bins means N+1 edges; check `num_bins` vs `num_edges` at
  every `linspace` call. Watch bin-edge max vs event max (edges include min,
  exclude max), repeated coordinate values, and periodic/frame wrapping.
- Extracting many event subranges via repeated `da.bins['p', a:b]` is
  O(events) per call — `da.bin(p=edges)` once, then slice.

### 5. View-vs-copy and shared ownership

- Positional/label slices are **views**: `sub = da['x', 0:2]; sub += 1`
  mutates `da`. Integer-array and boolean indexing return copies.
- Inserting a Variable into `coords`/`masks`/`data`, or a DataArray into a
  Dataset, **shares the buffer** — mutating the source later mutates the
  container. Flag mutation-after-insertion; require `.copy()` to break sharing.
- Function arguments (dicts, DataGroups, DataArrays) must not be mutated —
  copy first. This is one of the most-repeated review comments in the org.

### 6. Positional vs label-based indexing

`da['x', 0]` is positional; `da['x', sc.scalar(0.5, unit='m')]` is label-based.
An integer where a scalar Variable was intended (or vice versa) selects the
wrong data. Also: `da['x', 0]` drops the dim and demotes the sliced coords to
unaligned, while `da['x', 0:1]` keeps the dim and alignment — this changes
whether a later binary op raises on coord mismatch. Exact float label lookup
raises; use an interval. But grid-edge/index comparisons that must be exact
should NOT be replaced with `isclose`.

### 7. Coord alignment and propagation

Binary ops compare aligned coords and abort on mismatch; **unaligned coords
are silently dropped on mismatch**. `transform_coords` demotes consumed
inputs to unaligned. After a coordinate transformation, check the output is
self-contained (e.g. tof data should carry the Ltotal used to compute it) and
that stale coords that no longer describe the data are dropped.

### 8. Mask semantics

Masks combine with OR in binary ops. Reductions (`sum`, `mean`, `bin`,
`rebin`) over a dim **apply-and-drop** masks depending on that dim; `mean`
excludes masked elements from the count (not "treats as zero"). In
normalization, the result needs the **union** of detector and monitor masks,
zero/nonfinite monitor bins must be masked (not divided by, producing `inf`
weights), and a valid data range must not be inferred from zero counts.

### 9. Fail loudly — no silent defaults or broad fallbacks

- No `defaultdict` or implicit default for configuration that a user might
  forget to set (missing threshold, unknown detector name) — validate and
  raise with the offending name. Workflow inputs fed by live streams or files
  must have **no defaults** (a stale fallback silently corrupts results).
- Narrow `try/except`: cover only the statement that can legitimately fail,
  catch specific types, and never let a fallback hide a genuine error.
- Raise the semantically right type (`sc.UnitError`, `sc.DimensionError`,
  `KeyError`...) at the point of user error, not a cryptic failure later.
  `assert` is stripped under `-O` — library code uses `if ... raise`.
- Warnings must name the operation that actually failed and the file/node
  path that caused it.

### 10. NeXus access (scippnexus)

- `group['name']` is lazy; `group[...]` loads. Slice **before** loading
  (`f['entry/bank1']['event_time_zero', :1000]`), never load-then-slice.
  Pixel-range slicing of event detectors is NOT cheap (events are stored in
  time order); only pulse (`event_time_zero`) slicing is.
- Find components by class (`instrument[snx.NXdetector]`), not hard-coded
  paths. HDF5 paths are POSIX — `posixpath`, never `os.path`.
- Use `snx.compute_positions()`; don't re-derive geometry from
  `depends_on` chains or legacy `distance`/`polar_angle` fields.
- On malformed groups scippnexus warns and falls back to a `DataGroup` —
  code assuming a `DataArray` must handle that.
- Degrade per-item: one broken entry must not prevent loading intact siblings.

### 11. dtype and precision

Derived float quantities default to float64; float32 loses counts when
accumulating and int arithmetic truncates (`int(-3/2) == -1` but
`-3//2 == -2`) or overflows (`(a+b)/2`). ESS timestamps are int64 nanoseconds
— keep native precision end-to-end, convert explicitly and late. `sc.zeros`
over `sc.empty` unless overwrite-everything is provable. Indices: `unit=None`
unless used in arithmetic with physical quantities (then dimensionless).

### 12. Right layer, no reimplementation

Before accepting new code, ask what upstream already provides: essreduce has
the generic NeXus loading workflow, monitor normalization, uncertainty
broadcasting, tof conversion; scipp has `lookup`, `transform_coords`,
integer-array indexing, `fold/flatten`. Generic reduction logic belongs
upstream of instrument packages ("shouldn't this be in essreduce/ESSsans?").
Repeatedly asked and repeatedly fruitful: an "obviously new" feature often
reduces to an existing workflow with different parameters.

### 13. Provider purity and streaming

Sciline providers must be pure: no file writes, no mutation of inputs, no
wall-clock/random state — the scheduler decides when/if/how often they run.
For `ess.reduce.streaming.StreamProcessor`: dynamic and context keys must not
overlap or depend on each other; accumulators are not cleared on context
change (the context must be compatible with accumulated data); default
accumulators histogram — they do not support event data.

## Review workflow

1. Map which stack layers the diff touches (imports of `scipp`,
   `scippnexus`, `sciline`, `ess.reduce`) and read the matching reference
   file(s) before judging idioms.
2. Sweep the diff against the bug classes above; for each hit, verify against
   the actual semantics (don't assume numpy/xarray behavior).
3. Check tests: do they assert values with `sc.testing.assert_identical`
   (covers unit and dtype), not counts or shapes? Do "what happens if ..."
   edge cases (empty selection, repeated values, single bin, zero events)
   have tests? See [reference/conventions.md](reference/conventions.md).
4. Check the change against downstream consumers: renaming a public domain
   type or provider breaks sibling packages — require deprecated aliases or
   an org-wide usage check.
