# essreduce (ess.reduce) — reviewer reference

essreduce provides the shared building blocks for ESS reduction packages:
the generic NeXus-loading workflow, tof conversion, uncertainty handling,
monitor normalization, and streaming. The cardinal review question is
"does this diff reimplement something essreduce already provides?"

## Generic NeXus workflow

- `GenericNeXusWorkflow(run_types=[...], monitor_types=[...])` returns a
  sciline Pipeline; `GenericTofWorkflow` extends it with providers that
  convert event_time_offset to tof.
  `run_types`/`monitor_types` are mandatory: they constrain the `RunType`/
  `MonitorType` TypeVars, and a forgotten type simply has no branch in the
  graph. Nested typevars (`Outer[Inner[RunType]]`) don't survive filtering.
- Chain per component/run: `Filename[RunType] → NeXusFileSpec → location
  specs → NeXusComponent → EmptyDetector/Monitor → RawDetector/RawMonitor`.
- Event data is deliberately **stripped at metadata load** and loaded
  separately per component (`NeXusData[...]`), then merged in
  `assemble_*_data` — the memory-control design. A provider that loads
  geometry via a path that also reads counts loads data twice
  (real bug, essreduce#274).
- Poisson variances (`variances = values`) are added to counts at assembly
  — raw data downstream already carries uncertainties.
- Pixel folding to logical bank dims comes from the `DetectorBankSizes`
  param; positions from `transform * pixel_offset + offset`. Component
  `offset` and pixel `offsets` are different things — the composition is
  `T*(0,0,0) + offset`, not `T*offset` (caught in essreduce#290).
- Extend, don't rebuild: start from the generic workflow, `wf.insert(...)`
  instrument providers (same return type overrides — last wins), set typed
  params (`NeXusDetectorName`, `DetectorBankSizes`, offsets, thresholds).
  Reuse essreduce's domain types (`Filename[SampleRun]`,
  `RawDetector[RunType]`...) instead of minting parallel ones — new
  lookalike types fragment interop across ess packages.

## Uncertainty broadcasting (critical)

scipp raises on broadcasting variance-carrying data because the duplicated
values are correlated (doi:10.3233/JNR-220049). essreduce provides the
sanctioned handling:

- `ess.reduce.uncertainty.broadcast_uncertainties(data, prototype, mode)`
  with `UncertaintyBroadcastMode` ∈ {`drop`, `upper_bound`, `fail`}.
- `upper_bound` scales variances by the broadcast subspace volume —
  conservative over-estimate, the recommended default. `drop` discards the
  term (explicit user choice only). `fail` lets scipp raise.
- Typical use: monitor-normalization terms broadcast onto detector pixels.
  Flag: raw `.broadcast(...)` on variance-carrying data, `sc.values(norm)`
  to sneak past the error, or an `UncertaintyBroadcastMode` param accepted
  but never plumbed to `broadcast_uncertainties`.

## Monitor normalization

Use `ess.reduce.normalization` (`normalize_by_monitor_histogram` accounts
for the wavelength profile; `..._integrated` for flux only) rather than
`detector / monitor.sum()`. The review history of essreduce#252 defines
correctness here:

- The result needs the **union** of detector and monitor masks (and
  NaN-range masks); integration must respect masks (`nansum`).
- Zero/nonfinite monitor bins must be masked, not divided by (else `inf`
  event weights that blow up later, far from the cause).
- The valid range comes from the monitor, not from zero detector counts
  (pixels can have true zeros); for binned detectors without a bin coord
  the range must come from the **events**, not bin edges.
- Detector–monitor unit/range mismatch is a setup error → raise.

## Time-of-flight

- `GenericTofWorkflow` + `TofLookupTableFilename`; per-component
  `LookupTableRelativeErrorThreshold` masks unreliable table regions.
- Pulse-skipping: reference times must be epoch-based, never computed per
  chunk (per-chunk `tmin` flips pulse parity depending on chunk boundaries
  — essreduce#180). `PulseStrideOffset` cannot be guessed from data.
- Frame wrapping / `event_time_offset` modulo arithmetic is a recurring
  bug source: check edge-exact values, repeated values, and that any
  double-modulo trick carries an explanatory comment.
- Outputs must be self-contained: tof only means something with the Ltotal
  that produced it — keep it as a coord; drop stale coords
  (`event_time_offset`) after conversion. Don't accept a transform graph
  whose entries are partially ignored — a false promise that customization
  works (essreduce#302).

## Streaming (`ess.reduce.streaming.StreamProcessor`)

- `dynamic_keys` (updated every chunk) and `context_keys` (change rarely)
  must not overlap or depend on each other.
- **Accumulate histograms, never event data**: cost and memory of
  event-mode accumulation grow linearly with run time. `hist` goes
  upstream of the accumulator; define one accumulator per output histogram
  (esslivedata#900). Default accumulators histogram their input and do not
  support event data; `Min`/`Max` are scalar-only; a `clear()` must reset
  *all* state (a `MeanAccumulator` inheriting `clear` without resetting
  `_count` corrupted means — essreduce#241).
- Accumulators are **not cleared on context change** — the context must be
  compatible with what is accumulated (accumulating I(Q) across detector
  angles: fine; across sample temperatures: wrong — accumulate I(Q,T)).
  Prefer making the context a coord so incompatibility raises.
- Workflow must be linear in dynamic keys up to the accumulation point —
  StreamProcessor cannot verify this; misplaced accumulators are silently
  wrong. `allow_bypass=True` is a sharp knife.
- Outputs depending only on context must recompute on context update.
- Sciline workflow inputs fed from live streams must have **no defaults**
  (a stale NeXus fallback silently corrupts geometry — stated as design law
  in esslivedata review).

## Conventions specific to this layer

- Domain-type naming follows the C.6 guideline: spell abbreviations out in
  public names (`TofLookupTable`, not `LUT`), consistent
  `Raw*/Tof*/Wavelength*` coordinate-space progression, names reflect
  approximations ("straight-line Ltotal").
- Never rename public domain types casually: sibling packages
  (esssans/essdiffraction/essspectroscopy/esslivedata) and user notebooks
  break. Check org-wide usage first; keep deprecated aliases; keep loaders
  reading old serialized formats.
- No silent defaults for per-instrument configuration: thresholds and
  names are validated with diagnostics; acceptable defaults are explicit
  no-ops (`inf`, `None`), not arbitrary numbers. Prefer per-instrument
  defaults over one-size-fits-all.
- Fixed-structure containers: prefer a dataclass over `DataGroup`+NewType
  when fields are fixed (docstrings, typo safety); convert to DataGroup
  only for HDF5 serialization.
- Prefer a user-supplied callable over a config-object mini-language of
  operations applied in fixed order.
- Instrument specifics (bank sizes, detector names, distances) are typed
  parameters, not constants inside providers.
