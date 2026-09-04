# ADR 0011: Wavelength lookup tables laid out as blocks and selected by flight path

- Status: proposed
- Deciders: Simon
- Date: 2026-08-11

Split out of ADR 0010, which decides the generic mechanism this rides on: a workflow
output republished as a context input stream, requested by key and gated from the built
graph. This ADR decides what the lookup-table stream carries and how a job finds its rows.

## Context

Every workflow converting time-of-arrival to wavelength loaded a table from a file fixed
at import time. `LookupTableFilename` was assigned bare-generic per instrument, so one
simulation-derived table served every component in a job. Three properties of those
tables shape the decision.

**Out-of-range lookups fail silently.** A table is indexed by `distance`; a lookup outside
its range yields `NaN`, those events fall outside every histogram bin, and the component
renders empty with no error anywhere. The shipped tables already missed the monitors on
DREAM, LOKI and BIFROST, which therefore produced nothing.

**One table per job wastes almost all of its rows.** Per-component `Ltotal` spreads are
under a metre, often a scalar, while the shipped tables span tens of metres. The cost is
not bandwidth but recompute: the polygon rasterization runs over every distance row on
every chopper change.

**The range was a user parameter** whose default covered no instrument correctly, so an
operator starting the LUT workflow with defaults could silently blank every detector.

## Decision

### One LUT job, two tables, laid out as blocks

The LUT workflow keeps its single `chopper_cascade` source and publishes two outputs, a
detector table and a monitor table, as context streams. The range parameter is removed.

Two, rather than one per component, because a table is a function of `distance` and
`event_time_offset` alone. It carries no component identity: a per-component table is a
*restriction* of the same function to that component's rows, so components can share a
table with nothing lost. But a single uniform grid spanning every component is mostly
empty, and empty is not free: BIFROST's monitors sit 155 m from its detectors, which at a
fine resolution is a message an order of magnitude over a broker's default limit. A table
is therefore a **concatenation of uniform blocks**, one per group of components that sit
close together. Detectors share one dense block, since they cluster around the sample.
Each monitor gets its own block of a handful of rows. The wire states which rows form a
block in a `block` coord, so no consumer has to infer boundaries from a jump in row
spacing. The layout, its wire format and the reason a multi-block table must never reach
essreduce's interpolator are documented in `workflows/lut_blocks.py`.

One job, not one per component. A job's identity is `(workflow_id, source_name)` where
`source_name` is the stream it consumes; the LUT job consumes the chopper cascade and
nothing per-component. Per-component jobs would need synthetic trigger streams to give
each a source name, recompute the same cascade once per component, and leave one chance
per job to drift on the shared parameters. Consistency across components is the point.

The tables are ordinary outputs that a user can plot. "Why is this component empty" is
answered by its table, which was unreadable while the relevant rows were buried in tens
of metres of empty distance. Two outputs, rather than one per component, is also what
keeps the plot picker readable.

### Ranges are derived generically from the geometry, padded for motion

Each component's flight-path range is computed from the registered geometry artifact,
with the same essreduce providers the consumer runs at lookup time (`DetectorLtotal`,
`MonitorLtotal`). Producer and consumer derive the ranges in different services without
ever exchanging them; that agreement is what the end-to-end test pins. Padding is
deliberately generous: over-padding costs recompute in the LUT job and nothing else,
under-padding yields `NaN` silently, which is the failure this ADR exists to remove. The
`Ltotal` definition does vary with geometry, but by metres against flight paths of tens
to hundreds of metres, and padding absorbs it; a per-component rule would buy exactness
the distance resolution does not reward.

Motion is the one thing the artifact cannot supply: a live f144-driven transform is stored
as an *empty* NXlog. Instruments therefore declare one `AxisRange` per moving axis, keyed
by NeXus transform path, with both bounds as axis values in the axis's own units, so the
transform keeps supplying the direction of travel. Which components ride an axis stays
derived from the `depends_on` chain. Translations are bracketed by evaluating the geometry
at the corners of the box the bounds span; rotations are refused, since `Ltotal` is not
monotonic in an angle, and the first instrument with a live rotation has to design for it
deliberately (`workflows/lut_ranges.py`).

A component nobody can place gets no block. A job on such a source gates like any other
and then fails at its first recompute, reporting its flight path against the table's
coverage. Where the component is an aux selection, the factory rejects it at job creation
instead. A group with no placeable component publishes no table, so its stream is never
offered and a job asking for it fails at creation.

**The range is static and covers the full envelope.** The LUT workflow does not consume
motion. A live range would gate the LUT job on motion, couple motion to LUT-driven
clearing, and lag the value the consumer patches into its own geometry, so padding would
still be needed. A guard test recomputes each range from the artifact and fails if a
declared envelope no longer contains it, so regenerating an artifact cannot silently
move a component out of its table.

### A consumer selects its block by flight path

The bound key is essreduce's public `LookupTable[RunType, Component]`, so `Component`
distinguishes a job's detector and monitor tables with no new type parameter. A provider
takes the job's own `DetectorLtotal` or `MonitorLtotal`, already in its graph and
computed from geometry rather than stream data, and takes the block containing its
midpoint. Selecting by flight path rather than by name keeps component identity off the
wire entirely: the job knows its `Ltotal`, the table carries its distances, and neither
side names a bank or a monitor.

The wire value is one `DataArray` carrying the table with its scalar fields as coords,
taken from the built table rather than the job's parameters, since the two can differ.
The provider reassembles the dataclass from the block rather than routing through
essreduce's file loader, whose matching branch is a compatibility shim that cannot carry
`choppers` (`workflows/lut_context.py`).

A reduction needs one table per sciline `Component`: the detector plus the incident and
transmission monitor roles. Which physical monitor fills a role is a per-job aux
selection that an import-time declaration cannot name. Under a shared monitor table it
does not have to: one provider generic in `MonitorType` serves every monitor role, and
each instance picks its block via that role's own `MonitorLtotal`. This is what made the
aux-templated stream names of an earlier shape unnecessary.

### Consumers clear when the table's identity changes (deferred)

A new table means the chopper phasing changed, so data accumulated on either side are not
the same measurement. Consumers do not compare tables: the producer stamps an identity
coord derived from the inputs that determine the table, each rounded to a declared
precision, and a consumer clears when that scalar changes. Clearing on every received
table is rejected because restarting the LUT job re-emits an unchanged table, and that
restart is the recovery action. Not implemented yet; the design is tracked in issue 1248.

## Alternatives considered

| Option | Notes |
|---|---|
| **Two tables laid out as blocks, detectors dense and a block per monitor (chosen)** | Removes the range parameter, keeps the rows a per-component table would have had, and reduces outputs, streams and sciline keys to one per group. |
| One table per component, one job, many outputs | The first shape, and it worked. Generates the outputs model per instrument, multiplies streams by the component count, and forces per-job table selection into *stream names*. Superseded. |
| One instrument-wide uniform grid | Far less plumbing, but almost every row empty; the block layout exists to avoid it. Rejected. |
| Merge monitors into the detectors' block set | One stream instead of two, but a monitor job would receive the detectors' megabyte-scale block to read a few rows. Rejected. |
| Cluster detectors by gap rather than one dense block | Would suit an instrument with far-apart banks, but the layout would silently reshuffle when an artifact is regenerated, and no instrument needs it. Rejected for now. |
| One row per monitor | `WavelengthInterpolator` needs two nodes bracketing the flight path; a single node returns `NaN` for every lookup. Rejected. |
| One job per component | Synthetic trigger streams, the cascade recomputed per component, shared parameters free to drift. Rejected. |
| **Derive every range from the geometry artifact, padded (chosen)** | One derivation, no per-component declarations, agreement with the consumer by construction. |
| Declare the `Ltotal` rule per component, or hand-declare every range | Exact where padding is merely sufficient, and a dozen-plus declarations nobody re-checks when an artifact is regenerated. Rejected. |
| **LUT workflow consumes component motion streams** | Tables would always describe where the component is, and the travel envelope would disappear from the instrument declaration. Costs: motion joins the LUT job's gate, so a dead motion PV stops all wavelength reduction; every sample during a move re-emits; and the LUT job's value can lag the consumer's, so padding stays. The strongest alternative; recorded to revisit. |
| **Both monitor roles read the shared table and select by `MonitorLtotal` (chosen)** | One stream, one generic provider, no per-job identity on the wire. |
| Aux-templated stream names (`wavelength_lut/{incident_monitor}`) | Gate resolution rendered the template from the job's selection and route derivation expanded it over the field's choices. It worked and was the most intricate mechanism on the branch; sharing the monitor table removed the problem instead of solving it. Superseded. |
| Bind every candidate monitor to its own per-component key | Leaves every unselected key a dead parameter and restates the candidate list in the factory. Superseded. |
| Reset on any received table, with no identity | Puts the decision at the producer, but a restart re-emits an unchanged table and would wipe every consumer. Rejected. |
| ADR 0006's `start_time` generation marker as identity | Changes on a restart with identical configuration. Rejected. |
| Content comparison at the consumer | One comparison per consumer instead of one stamp, with the noise tolerance far from the setpoints it filters. `sc.identical` is unusable outright: it returns `False` for bit-identical arrays containing `NaN`, which every table has. Rejected. |
| Identity hashed from the table's bytes | Not stable across a library or platform change, and says nothing about *why* the table changed. Rejected. |

## Consequences

- The LUT workflow loses its range parameter and gains two outputs; the outputs class is
  static and shared by every instrument.
- `Instrument` gains `axis_ranges` and, once the LUT factory is attached, the set of
  components it could place. That set decides which group tables are published, and
  therefore which context streams are offered at all.
- Handing a multi-block table to essreduce's interpolator is silently wrong under numba
  and correct under the scipy fallback. `numba` is not a declared dependency, so the test
  suite exercises the tolerant path: the guard is the block selection itself plus a test
  that the selected block is uniform, not a test that would fail on the mistake.
- The aux-templated stream-name mechanism lost its only user and is removed; stream names
  are plain on both sides of the mirror.
- On a migrated instrument the file-based table is unreachable, so the streamed table
  cannot be A/B'd against it side by side; the published tables are the comparison
  surface instead.
- Motion and LUT clearing are orthogonal by construction: because the range is static, a
  carriage move does not change the table. Whether motion should clear accumulated data
  is a question about motion bindings, left to be decided on its own.
