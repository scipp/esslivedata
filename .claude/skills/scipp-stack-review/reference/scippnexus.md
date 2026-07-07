# scippnexus — reviewer reference

scippnexus gives h5py-like access to NeXus files, returning scipp objects
with units, dim labels, and variances. Its central design points are
laziness, class-based lookup, and lenient degradation on malformed files.

## The three meanings of `group[...]`

1. `group['name']` (str, may be a nested/absolute path) → lazy child
   (`Group` or `Field`); **no array data read**.
2. `group[snx.NXdetector]` (class, or list of classes; `snx.Field` for all
   datasets) → dict of matching children. Preferred over hard-coded paths:
   group *names* are arbitrary, the class attribute is the contract.
3. `group[()]` / `group[...]` / `group['dim', 5:100]` (scipp index) →
   **loads** and assembles into `DataArray`/`Dataset`/`DataGroup`.

A `Field` loads as `sc.Variable` — or a plain Python scalar for unit-less
scalar datasets. Unrecognized units warn and fall back to dimensionless.

## Performance model

- Navigation (`keys()`, `group['x']`) is cheap and cached; only scipp-index
  subscripts read data. Slice **before** loading:
  `f['entry/bank1']['x_pixel_offset', :10]`, not `f['entry/bank1'][...]`
  followed by slicing.
- `NXevent_data` loads as a DataArray **binned by pulse**
  (`event_time_zero` dim; event coords `event_time_offset`, `event_id`;
  synthetic unit weights). Slicing by `event_time_zero` before load reads
  only those pulses — the key lever.
- Inside an `NXdetector`, events are grouped by `detector_number` on load
  and the pulse dim is not preserved. **Pixel-range slicing still reads the
  whole event table** (events are stored in time order) — only pulse
  slicing is cheap.
- Because of caching, writes via the same handle are not visible to
  subsequent reads on that handle.

## Loading semantics worth knowing in review

- `X_errors` sibling fields merge into variances of `X` (stddevs in file →
  squared to variances on load).
- `NXdetector`/`NXmonitor` load as a **DataGroup wrapping** the signal
  DataArray (plus geometry, `depends_on`, masks...) — `det['data']`, not
  `det.data`. NXlog with sublogs also loads as a DataGroup and disables
  positional time selection.
- Malformed groups: assembly failures **warn and fall back to returning a
  raw `DataGroup`** instead of raising. Downstream code that assumes a
  DataArray must tolerate this (and must not suppress the warning).
- Per-item degradation is the policy: one broken entry (dead link, broken
  transformation chain) must not prevent loading intact siblings.
- Unknown NX classes load as plain DataGroups; `cue_*` fields are dropped.

## Transformations and positions

- `depends_on` loads as a `DependsOn`/`TransformationChain` object, not a
  string. Positions are **not** computed at load time.
- `snx.compute_positions(dg)` resolves chains (including time-dependent
  transforms, interpolated `kind='previous'`), stores `position`, and
  applies detector pixel offsets — but pixel offsets only for
  time-independent transforms. Legacy `distance`/`polar_angle`/
  `azimuthal_angle` fields are ignored.
- Flag any hand-rolled matrix walking of `depends_on` chains — it misses
  chain links, units, rotation-vs-translation, and time dependence.

## Anti-patterns (wrong → right)

```python
# Load-everything then slice
da = f['entry/bank1'][...]; da['event_time_zero', :1000]
# → f['entry/bank1']['event_time_zero', :1000]

# Pixel slicing to "save memory" (reads everything anyway)
f['entry/bank1']['detector_number', :100]      # slow, full event read

# Raw h5py where scippnexus applies (loses units/dims/variances/assembly)
h5py.File(fn)['entry/log/value'][:]
# → snx.load(fn, root='entry/log')  →  DataArray with time coord + units

# Hard-coded paths
f['entry/instrument/bank102']
# → f['entry/instrument'][snx.NXdetector]   # dict name → Group

# os.path on HDF5 node paths (breaks on Windows)
os.path.join(parent, name)          # → posixpath.join(parent, name)

# Reimplementing geometry
# manual depends_on walking / rotation matrices → snx.compute_positions(dg)
```

## Review-history signals (scippnexus repo)

- **Narrow try/except** is the single most repeated demand: the except must
  cover only the statement that can legitimately fail, list concrete
  exception types, and never let a fallback hide a genuine error. Fallback
  warnings must state *which* operation failed and *which* node/file caused
  it.
- Don't mutate arguments (`children`, `dg` dicts) — copy first.
- Right exception types: `sc.DimensionError` for bad dims, `IndexError`
  only for out-of-range; raise at the point of user error, not downstream.
- Layer discipline: don't ping-pong between h5py and snx.Group objects in
  hot loops; downstream policy (default units, reduction choices) does not
  belong in scippnexus.
- Caching (`lru_cache` on methods leaks; `cached_property` vs dataclass
  `__eq__`/`__hash__`, deadlocks) only with benchmarks to justify it.
- Test the *values* of everything loaded, not counts: a classic bug
  (scippnexus#137) returned the full event buffer when 0 pulses were
  selected, and the test only asserted the pulse count. Use
  `sc.testing.assert_identical`. In-memory test files:
  `h5py.File(name, driver='core', backing_store=False)`.
- Failure-case tests expected: nonexistent paths, trailing slashes,
  `depends_on` cycles, length-0 datasets, unit-mismatched `_errors`.
- Bit-mask handling: prefer `mask & (1 << bit)` over arithmetic bit checks
  (sign-bit hazard); watch dtype up-casting interacting with bit positions.
