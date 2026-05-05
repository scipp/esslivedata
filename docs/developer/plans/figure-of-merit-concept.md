# Figure of Merit (FOM) — Concept and Operational Picture

This document captures the conceptual understanding of the Figure of Merit mechanism — what it is for, who uses it, and how it fits into the experiment.
It is deliberately free of implementation detail.
A separate plan document covers the technical design.

## The setting

An ESS instrument runs experiments under the control of NICOS.
A typical experiment includes scans:
NICOS moves the sample (or some other device) through a sequence of points, measuring at each one.
The natural question at every scan point is *"have we measured enough yet, can we move on?"*

Two strategies exist for answering this:

1. **Time-based**: measure for a fixed duration.
   The user must guess in advance how long is enough at each point.
   This is robust but wasteful:
   times are usually overestimated to be safe, and the strategy cannot adapt to the actual incoming neutron rate (which varies with sample, beam, instrument state).
2. **Statistics-based**: measure until some quantity reaches a target — for example, "until the integrated counts in this detector region exceed 10^6".
   This adapts naturally to whatever rate the instrument delivers and is a much better use of beam time.

NICOS can already do strategy 2 when the relevant quantity is a *raw* counter — a number that some device publishes directly.
The problem arises when the quantity is *derived* — produced by a non-trivial reduction step that combines, filters, or fits incoming neutron data.
That reduction is exactly what ESSlivedata workflows do.
The FOM mechanism exists to make a workflow output look, to NICOS, like a simple counter.

## What a Figure of Merit is here

For our purposes, a FOM is almost always a single scalar (with or without a physical unit) — for example:

- counts in a region of interest on a detector,
- amplitude of a fitted Bragg peak,
- a contrast or ratio derived from two regions.

The condition that turns this number into a decision (`> threshold`, `stable to within X%`, etc.) lives on the NICOS side, inside the scan command.
The FOM mechanism does not encode conditions, thresholds, or fitting choices that NICOS needs to know about — NICOS sees a stream of numbers and applies its own scan logic to them.

This may seem to leave fitting and other "richer" cases unaddressed, but the design intent is the opposite:
any complexity is absorbed into the workflow itself, so that what reaches NICOS is still a single number.
If we want NICOS to scan "until the fitted peak amplitude is stable", the fit could happen inside the workflow and the amplitude is what NICOS sees.
But if the FOM is, e.g., 1D curve then NICOS could still decide to define a more complex derived FOM itself, without affecting the general line reasoning here.

A slight generalisation is allowed for the common case where the same workflow runs across multiple source streams (e.g., the same reduction on every detector bank).
ESSlivedata runs one job per primary source, so the natural backend output is N parallel scalar substreams rather than a single pre-combined number.
The slot's alias then carries all N substreams (interleaved on the wire, distinguished by the embedded ``ResultKey``); NICOS aggregates them with a trivial reduction (typically a sum).
The "single number" framing still applies *logically* — what NICOS feeds into its scan condition is one scalar per scan tick — but the aggregation step lives in NICOS, not in a bespoke backend combiner.

## Roles during an experiment

Three actors are involved in a FOM-driven scan:

- **NICOS** owns the experiment. It drives motion, decides when each scan point starts and ends, applies the threshold/condition, and records the experiment metadata for reproducibility.
- **ESSlivedata backend** runs the workflow that produces the FOM and publishes its output continuously.
- **The ESSlivedata dashboard** is where a human operator chooses *which* workflow to use as the FOM, *which* of its outputs to expose, and what parameters to use. Once configured, the dashboard steps out of the control loop.

The intent of the FOM concept is to treat ESSlivedata as part of the extended experiment control interface in the hutch. The instrument scientist configures the FOM through familiar dashboard machinery rather than having to teach NICOS about specific workflows, their outputs, or their parameters.

## Lifecycle and ownership

In normal operation an experiment is almost always running, and the FOM job runs continuously alongside it. Configuring the FOM is a relatively rare, deliberate act; consuming and resetting it happens many times per scan.

- **Configuration** is done from the dashboard before (or between) scans. The scientist picks a workflow, an output, and parameters.
- **Resetting** happens at every scan point. NICOS issues a reset to zero the accumulator before measuring at the new point. When a slot covers multiple substreams, the dashboard fans the reset out to each source-job; the resets do not land atomically, so during the brief skew window the aggregated value is partially-reset. For monotonic accumulators (counts) this manifests as a transient under-threshold reading, never a false over-threshold, which is the safe direction for "have we measured enough?" scans.
- **Reading** happens continuously: the FOM stream produces a fresh number whenever new neutron data arrives.
- **Stopping or reconfiguring** the FOM mid-experiment is possible but should be guarded. Both actions cause a brief outage in the output stream, and a reconfigure mid-scan can silently change the meaning of the number NICOS is reading. The dashboard should warn the operator before allowing either, but should not forbid it — there are legitimate cases (e.g., the operator realises a few scan points in that the parameters could be improved and wants to fix them without aborting the scan).

Multiple parallel FOMs are not the common case. The mechanism should allow a small number of indexed slots (`fom-0`, `fom-1`, ...) to keep the door open for cases like "scan until the *ratio* between two quantities stabilises", but slots are not auto-allocated and the default expectation is that only `fom-0` is in use.

## Timing and synchronisation

Initially the precise timing of reset versus sample motion does not matter — a few stale counts during sample movement are negligible compared to a typical measurement. As source power and instrument performance improve, measurement times shrink and timing alignment between motion and reset will become more important. The mechanism should be built such that scheduled reset/start/stop semantics can be added later without redesigning the data flow.

## Failure modes and degraded operation

The FOM mechanism makes ESSlivedata's backend services a soft real-time dependency of the experiment control loop. The implications are:

- If the **dashboard** crashes or is closed after the FOM is configured, scans continue normally. The dashboard is not in the control path at runtime.
- If the **backend services** are unavailable, FOM-driven scans cannot proceed. NICOS falls back to time-based scans, which is inefficient but always available. This is acceptable: ESSlivedata becoming part of experiment control is a deliberate design choice, and the fallback bounds the operational risk.

## What FOM is *not*

The FOM is an operational signal, not a scientific record. Its sole purpose is to absorb the unknown incoming rate so that scan points take an appropriate amount of time. It is a proxy for measurement duration.

Reproducibility of the experiment does not depend on the FOM:

- NICOS records sample positions, instrument state, and experiment metadata.
- Beam-monitor readings (independent of the FOM) are used for normalisation in subsequent analysis.
- The raw neutron data is preserved and can be re-reduced offline.

Consequently neither NICOS nor the NeXus file needs to know what the FOM is configured to compute. The FOM workflow can be opaque to NICOS — just a number on a known stream.

## Boundaries with existing mechanisms

NICOS can already steer scans on raw counters. The FOM mechanism is specifically for derived quantities that need workflow-level reduction (ROI integration, fits, combinations of streams). Cases where a single detector or monitor publishes a usable scalar directly should continue to be handled by NICOS without involving FOM, to avoid unnecessary coupling.

The FOM concept also does not aim to invert the relationship between NICOS and ESSlivedata. NICOS does not become a workflow author or parameter-setter; it consumes a number and issues resets. Authoring workflows and choosing their parameters remains a dashboard / instrument scientist concern.

## Open conceptual decisions captured

- **Single number, condition in NICOS.** Confirmed.
- **One active FOM normally, with indexed slots reserved.** Confirmed, no auto-increment.
- **Continuous job, reset per scan point.** Confirmed.
- **Reconfigure and stop guarded but not forbidden.** Confirmed.
- **Backend hard dependency, dashboard soft.** Confirmed; time-based fallback exists.
- **FOM not part of reproducibility record.** Confirmed.
- **No auth beyond generic topic-level auth.** Confirmed; in practice the dashboard is the only configurator.
