# Proposal: unified workflow-spec surface across ESS data pipelines and UIs

**Status:** discussion document. Not an implementation plan and not a commitment to a direction. Its purpose is to make the tradeoffs of a specific architectural direction legible enough that we can decide to adopt it, modify it, or reject it on the basis of what it actually costs rather than how it sounds on paper.

## Problem statement

ESS scientific-data software executes and presents reduction workflows across several contexts that today do not share a common definition of what a workflow *is*. The same DREAM powder reduction, the same LoKI SANS, the same BIFROST spectroscopy — as scientific ideas they are each a single thing. In code, each is implemented, described, and configured in multiple places, inconsistently, by different teams, with independent drift.

The contexts in play:

- **Streaming execution during live beam time.** Kafka-driven, message-by-message accumulation, updating a dashboard in near real time. Today this lives in `ess.livedata`.
- **Batch execution on stored NeXus files.** Sciline-based pipelines running once over a complete run, producing final results for publication or further analysis. Today this lives in the technique packages (`ess.powder`, `ess.sans`, `ess.reflectometry`, `ess.diffraction`, `ess.spectroscopy`, ...) in the `scipp/ess` monorepo.
- **Interactive Jupyter / ipywidget UIs.** Scientist-facing notebook UIs that wrap technique-package workflows. Today these live alongside the technique packages.
- **Live dashboard UIs.** Panel/Bokeh-based interactive surfaces shown during beam time. Today these live in `ess.livedata`.
- **CLI wrappers and scripted replay.** Programmatic entry points for operations and testing.
- **External frontends.** NICOS (instrument-control software) is the concrete candidate; anything else that wants to know which workflows exist and how to invoke them falls in the same class.

Every one of these contexts needs to know: *what workflows exist for instrument X*, *what parameters does workflow Y accept*, *what outputs does it produce and in what shape*, *how should it be rendered*. Every one answers those questions by reaching into wherever the workflow is currently defined — which today means a different place per context, reimplemented from scratch per consumer, with no authoritative source.

The concrete consequences, each a real failure mode rather than a design aesthetic:

- **Parameter defaults drift.** The powder-reduction d-spacing default in the livedata dashboard is not guaranteed to match the default in the ess.powder ipywidget UI. Scientists configure the same workflow differently in different tools because the tools disagree.
- **Output-shape disagreements.** Livedata's `I(d)` may carry different coordinate units or dim names than ess.powder's; consumers of both have to reconcile silently.
- **Validation duplication and inconsistency.** "This value must be positive" is declared in one place, copied to another with minor variation, drifts.
- **Duplicated UI effort.** Every UI that wants to render a workflow builds its own form logic from scratch, because there is no shared spec to render against.
- **NICOS has no path in.** There is no single source of truth for "these are the workflows available on DREAM right now" that an out-of-process frontend can query.
- **Contribution overhead for workflow changes.** A scientist adding a parameter to a powder workflow edits the technique package, then separately updates livedata, then separately updates the ipywidget UI. Three changes for one concept.

## What a workflow spec is (and is not)

A **workflow spec** is vocabulary: a declarative description of a reduction pipeline's interface. It answers four questions and no more:

1. **Identity.** Instrument, technique/name, version. Sufficient to refer to the workflow unambiguously across UIs, backends, and releases.
2. **Inputs.** The parameter model (typed fields, ranges, defaults, units), plus the set of data sources the workflow consumes (detector streams, monitor streams, auxiliary log channels, or equivalently NeXus groups in the batch case).
3. **Outputs.** The set of result arrays the workflow produces, each described structurally: dim names, unit, coordinate names and their units. Enough for a UI to pick a plotter, for a consumer to understand the shape, for a downstream consumer to route the output.
4. **Display metadata.** Titles, descriptions, grouping labels, plot hints. Not part of identity but part of the spec, because UIs need it and it has nowhere else sensible to live.

A workflow spec is deliberately *not*:

- **The algorithm.** The reduction code — the sciline graph, the scipp transformations — lives with the technique package, not with the spec. A spec describes an interface; an implementation satisfies it.
- **Runtime machinery.** Kafka topics, accumulator state, stream-to-source bindings, sciline graph construction, session state, job scheduling — all runtime concerns, none of them in the spec.
- **UI rendering code.** Forms, plot layouts, widget wiring — consumers of the spec, not part of it.
- **Fine-grained data-layout detail.** Bin-edge vs bin-center, dtype, chunk shape — runtime or per-result detail, not schema.

The separation is load-bearing. The unification premise rests on specs being cheap to share across Python packages with very different dependency stacks and release cadences. Anything that pulls scipp into the spec layer, or sciline, or a specific UI toolkit, forecloses one of the consumers the unification was supposed to serve.

## The goal

One workflow-spec vocabulary, declared once, consumed uniformly by every UI in the ESS ecosystem and satisfied uniformly by every compute backend. Concretely, two unification axes that together form a matrix:

**Cross-UI unification.** The same workflow spec drives the livedata dashboard, the ess.powder ipywidget UI, any future CLI wrapper, and (if it ever integrates) NICOS. A parameter form rendered in any of them uses the same field names, validations, defaults, descriptions. A user who has configured a powder workflow in one UI recognizes it immediately in another.

**Cross-compute unification.** The same workflow spec describes both the streaming pipeline run by esslivedata and the batch pipeline run by ess.powder on a NeXus file. The parameters a scientist enters at a beam-time dashboard are the same parameters they would enter in their post-run analysis notebook; the outputs they see live are the same outputs they see in the final batch result. Replay of live data as batch and streaming of a NeXus file through the live pipeline become natural operations because both backends speak the same vocabulary.

Expressed as a matrix:

```
                Streaming (esslivedata)       Batch (NeXus + technique package)
Livedata dashboard            ✓                           (replay)
Ipywidget UI            (live notebook?)                       ✓
CLI wrapper                   ✓                              ✓
NICOS                         ✓                              ?
```

Each cell in the matrix consumes or implements the same spec. The cells that today do not exist (ipywidget UIs over live data, CLI wrappers that work for both, NICOS in either column) become tractable because the spec-level integration cost collapses to zero.

This is the claim the rest of the document is evaluating.

## The ecosystem today

Stated as facts, without assuming any particular direction:

`ess.livedata` (this repository) is a streaming framework and dashboard. It defines its own workflow-spec types inline — `WorkflowSpec`, `WorkflowId`, parameter models, output templates — scattered across `config/`, `handlers/`, and per-instrument subpackages under `config/instruments/`. These types today carry `sc.DataArray` default factories as output templates, validators that are pydantic-based, and instrument-specific subclasses that reach into the livedata configuration layer. The specs, the factories that build runnable workflows from them, and the streaming adapters that wire them to Kafka are intermixed.

The **technique packages** (`ess.powder`, `ess.sans`, `ess.reflectometry`, `ess.diffraction`, `ess.spectroscopy`, and similar) live in the `scipp/ess` monorepo. Each owns the reduction algorithm for its technique, implemented as a sciline pipeline over scipp types, operating on NeXus-file inputs. Each has its own convention for how parameters are declared (function signatures? typed dicts? pydantic models? varies). Their UIs (notebook-level ipywidgets where they exist) are built ad-hoc against those conventions.

`ess.reduce` (same monorepo) provides cross-cutting reduction primitives — rebinning, focusing, correction utilities — shared across techniques. It owns compute; it does not today own vocabulary.

`scippneutron` is the low-level neutron-specific scipp layer underpinning the technique packages.

**NICOS** is the ESS instrument-control system. It is not a Python consumer of any of the above today. Whether it ever will be is a matter of active conversation rather than commitment.

The seams where duplication and drift happen are visible in every direction:

- Workflow parameters for DREAM powder exist both as whatever-the-technique-package-uses *and* as livedata's `DreamPowderWorkflowParams` pydantic model. These have to be kept in sync manually.
- Output-shape conventions differ: livedata's pydantic `*Outputs` models use `sc.DataArray` default factories; technique packages return scipp objects directly from sciline. "What does this workflow produce?" has two different-shaped answers depending on who is asking.
- Validators live in livedata only. The technique package does not know that a parameter has an allowed range until the user's notebook crashes.
- Display metadata (titles, groupings) exists only in livedata. The ipywidget UI invents its own.

None of this is pathological today because the duplication is small and the teams are close. It is pathological *eventually* because the surface grows with every instrument and every technique; the coordination cost is super-linear; and the "let's reconcile" projects that would otherwise close the gap keep being lower-priority than the next beam-time fix.

## Architectural direction under discussion

The direction this document is evaluating is a three-way split with a deliberate choice about where the vocabulary lives:

```
┌──────────────────────────────────────────────────────────────────────┐
│ UI surfaces                                                          │
│   livedata dashboard │ ipywidget UIs │ CLI │ NICOS (future)          │
└──────────────────┬───────────────────┬───────────────────┬───────────┘
                   │ consumes specs    │                   │
                   ▼                   ▼                   ▼
┌──────────────────────────────────────────────────────────────────────┐
│ ess.schemas                                                          │
│   pydantic + stdlib vocabulary                                       │
│   WorkflowSpec, WorkflowId, parameter models, output schemas,        │
│   instrument specs, display metadata                                 │
│   (possibly: instrument-specific and technique-specific subpackages) │
└──────────────────┬────────────────────────────────────┬──────────────┘
                   │ implemented by                     │
                   ▼                                    ▼
┌──────────────────────────────────┐   ┌───────────────────────────────┐
│ Streaming backend (ess.livedata) │   │ Batch backend (ess.<technique>)│
│   Kafka source/sink              │   │   sciline over NeXus files     │
│   accumulators, orchestration    │   │   technique-specific algorithm │
│   live-only parameter overlays   │   │                                │
│   dashboard                      │   │                                │
│        ↓ uses                    │   │        ↓ uses                  │
│ ess.reduce (transforms)          │   │ ess.reduce (transforms)        │
└──────────────────────────────────┘   └───────────────────────────────┘
```

Three load-bearing structural claims sit inside this picture, each of which can be evaluated on its own:

1. **Vocabulary is separable from compute.** A workflow-spec type can carry enough information to be useful to every UI (and to act as a contract for every backend) without importing scipp, sciline, or any streaming machinery. This claim is empirically testable: if a spec can be declared in pydantic + stdlib and still answer the four questions above, the claim holds. The output-shape question is the hardest — today outputs are declared via `sc.DataArray` factories; replacing that with a shape-only schema (dims, unit, coord units, no array construction) is the key design move that makes the separation possible.

2. **Streaming is an adapter over shared compute.** The cross-cutting data-shaping primitives that today live in esslivedata (detector-view pixel-to-logical binning, monitor histogramming, focus, ROI reduction) are equally applicable to batch pipelines. They belong in `ess.reduce`, not in livedata. What stays in livedata is the reactive/streaming half: accumulators, Kafka plumbing, orchestration, incremental-update behaviour. This is the claim that says livedata shrinks to its actual job.

3. **Technique packages become spec-conformant.** Each technique package declares its workflows as specs drawn from the shared vocabulary, or equivalently the vocabulary mirrors the technique-package interface. Either way, the technique package is no longer a separate island — its workflows are first-class entries in the shared catalog, consumable by any UI and executable by either compute backend.

Each claim is an argument. Each is also where the main objections live. The rest of this document pulls them apart.

## Why this direction (the upsides)

Not to persuade — to make the tradeoffs legible. Each of these has a cost counted later.

**One vocabulary, many consumers.** Every UI that wants to render a workflow reads from one place. `WorkflowSpec` has one shape, one changelog, one compatibility story. The n-by-m problem (n UIs × m places workflows are defined) collapses to n-plus-m.

**Technique packages gain a UI path.** Today a technique package that wants an interactive UI has to build it. Post-unification, any UI built against the shared vocabulary works for any technique package whose workflows are expressed in it. The ipywidget UI, the livedata dashboard, and the CLI become interchangeable frontends.

**Livedata gains reduction correctness for free.** Today livedata's powder-reduction adapter is a livedata-side reimplementation of what ess.powder does in batch. Post-unification, livedata calls ess.powder directly (or ess.reduce transforms that ess.powder also calls). "The live value differs from the batch value" becomes a class of bug that is structurally hard rather than routine.

**The parallel interface declaration in livedata disappears.** Livedata today maintains its own pydantic declaration of every workflow's parameters and outputs, in parallel with whatever the technique package declares implicitly through its sciline providers and scipp newtypes. The two declarations are unlinked by construction; keeping them in step is a manual *ongoing* burden, not a one-time migration — even with nobody actively changing anything, the two surfaces can drift as upstream types evolve. When a technique package's reduction signature shifts, livedata's pydantic model has no way to know; the mismatch surfaces at integration time, or less happily as a silently wrong result. Post-unification the two declarations collapse to one: technique packages and livedata reference the same parameter and output types, and the translation layer livedata currently maintains (pydantic → sciline inputs, sciline outputs → display metadata) shrinks or disappears. The drift class of bug is gone not because it is patched but because the surface that permitted it ceases to exist. This is distinct from the contribution-overhead upside below: that one is about the cost of making a change; this one is about the cost of *not* making one.

**Replay and streaming of stored data become natural.** Both backends speak the same spec. Running the streaming pipeline over a NeXus file (for testing, for reprocessing) requires no new glue. Replaying a recorded Kafka stream through the batch pipeline (for validation) does not exist as a separate pathway.

**NICOS integration has a surface.** Whether or not NICOS is ever wired in, the fact that a single source of truth exists — "these workflows are available, here are their parameters" — makes the integration *possible*. Today it is not.

**Contribution overhead drops for workflow changes.** A scientist adding a parameter to a powder workflow edits the spec once. Every UI that reads it picks up the change; both backends that implement it see the new field. The three-places-edit-in-sync failure mode disappears.

**Testability improves at every layer.** Shared vocabulary is testable in isolation (pydantic + stdlib test job, no scipp). Cross-cutting primitives in ess.reduce are testable as batch transforms, without Kafka setup. Streaming machinery in livedata is testable as Kafka plumbing over known transforms.

**Design discipline is forced.** Unification requires settling questions that today can be perpetually deferred — versioning contracts, validator semantics, streaming-vs-batch parameter boundaries. Some of these commitments are good for the code regardless of whether the unification lands fully.

## Why this direction is hard (the downsides)

Symmetric honesty. Every upside has a cost that accrues to somebody.

**Beam-time fix latency.** Today an instrument scientist who notices a wrong parameter default at 2am during a run opens a one-line PR in livedata, gets on-call review, redeploys, total ~15 minutes. Post-unification, if the authoritative spec lives in the scipp/ess monorepo, the same fix is a PR to that monorepo → a different reviewer pool → release cut → livedata dependency bump → PR to livedata → review → deploy. Hours at best, days at worst. One overly-strict validator is now an hour of beam time.

**Version-mismatch failures.** Once the vocabulary is a shared package pinned by multiple consumers, the consumers can disagree on version. Backend at vocabulary v1.2, dashboard at v1.3: the dashboard renders fields the backend does not emit, users see ghost inputs. Backend at v1.3 with a new required field, dashboard at v1.2: the dashboard cannot validate what arrives. During rolling deploys, the mismatch is transient but not zero. This class of failure does not exist today because there is only one version of anything.

**Rollback asymmetry.** Rolling livedata back while the vocabulary stays at a newer version is usually unsafe: livedata may rely on a field the older vocabulary did not have. Rolling the vocabulary back while livedata stays is usually unsafe for the opposite reason. Today neither exists as a concern because the code moves together. Post-unification, on-call needs a rehearsed playbook and a non-obvious instinct for which way to turn the crank.

**Cascading upstream interface evolution.** Every non-cosmetic change in a technique package — new parameter, renamed output, changed semantics — must propagate through the shared vocabulary. A change that today lives in a single PR in a single repo becomes a coordinated release train across two or three. For routine feature work in technique packages (not emergencies) this is the dominant cost: every release of every technique package incurs coordination overhead. This multiplies across the number of technique packages (currently five to six) and across their release cadences. Additive/non-breaking changes are absorbable; breaking changes force the full ceremony.

**Breaking-change blast radius.** A DREAM-only concept change today touches DREAM's files. Post-unification, a change to the shared vocabulary affects every instrument and every technique package atomically — even when the change is semantically scoped to one. Every vocabulary release implicitly obligates every downstream consumer to re-verify.

**Ownership dilution.** A bug that lives in livedata today is a livedata bug. Post-unification, a bug in the dashboard that traces back through shared vocabulary to a technique package is a three-team bug: who owns the fix, who reviews, who releases, who deploys. Without explicit cross-team coordination, this fails exactly under pressure.

**Release-cadence coupling.** Livedata today ships when livedata is ready. Post-unification, livedata ships when the slowest of {vocabulary, reduce, technique packages, livedata} is ready, for any change that crosses the boundary. Cadence becomes a coordination problem, not just a per-repo discipline.

**Dev-loop friction.** Reproducing a bug that spans the boundary requires editable installs across multiple checkouts. The natural inner loop — edit, test, deploy — grows joints. A devcontainer that worked in one repo no longer works end-to-end.

**Cognitive overhead for contributors.** A new contributor fixing a monitor-histogramming bug today edits livedata. Post-unification, they must figure out which of {vocabulary, reduce, livedata, technique package} owns the bug, how to get a change landed in that repo, what release flow applies, how to cut over. Onboarding gets harder before it gets easier.

**Each technique package becomes a stakeholder whose constraints matter.** Today livedata makes its decisions in isolation. Post-unification, every vocabulary-touching decision is a decision that affects teams that do not answer to livedata's priorities and whose priorities do not answer to livedata's either. "Let's change this" becomes a conversation across teams.

The costs are not reasons not to attempt this direction. They are reasons to take the architecture seriously as an organizational commitment, not just a code commitment, and to front-load the decisions that determine how survivable the costs are.

## Decision points

Each subsection is a decision that must be made (or consciously deferred) for the direction to be actionable. They are not independent — choices compound — but they are separable enough to have distinct tradeoffs. The options listed are not exhaustive; they are the ones actively on the table.

### 1. Vocabulary scope

What, exactly, lives in the shared `ess.schemas` package?

**Options:**

- **(a) Framework only.** Just the cross-cutting types: `WorkflowSpec`, `WorkflowId`, `InstrumentSpec`, `ComponentSpec`, the output-shape schema (`ArraySpec`-like), the `AuxInput`/`AuxSources` structure. Parameter models, instrument specializations, and technique specializations live elsewhere.
- **(b) Framework plus generic workflow parameter models.** Add the technique-agnostic base parameter and output models — generic monitor-data params, generic detector-view params, generic timeseries params — that are not owned by any one technique package.
- **(c) Framework plus generic plus instrument-specific subpackages.** Add per-instrument subpackages holding instrument-scoped enums, subclassed parameter models, component metadata. The vocabulary is the full catalog of workflows, organized by instrument.
- **(d) Framework plus generic plus instrument plus technique subpackages.** Also absorb per-technique schemas (powder, SANS, reflectometry, ...) so the shared vocabulary is the authoritative declaration of every workflow in the ecosystem. Technique packages then import from vocabulary rather than defining their own conventions.

**Tradeoffs:**

(a) is the cleanest boundary but forces every consumer of parameter models to go somewhere else — where? The vocabulary then has no home for the `MonitorDataParams` that livedata, ess.powder, ess.sans, and a NICOS frontend all want to share. Pushing those into each technique package defeats the unification; pushing them into `ess.reduce` drags compute into what should be vocabulary.

(b) is the minimum that delivers on the cross-UI claim. Cross-technique shared models (monitor normalization, detector views) live here; technique-specific specializations stay with the technique package. Coherent but requires the technique packages to import from the vocabulary, which is exactly the direction this plan wants anyway.

(c) is where the instrument specializations become first-class citizens of the vocabulary. Livedata's existing instrument-subpackage pattern relocates here. Attractive because it matches the shape of today's code. Costly because instrument-specific changes — which today are local to livedata — become vocabulary changes that every consumer re-verifies.

(d) is the maximum unification position. The shared vocabulary is the canonical declaration of every workflow anywhere. Technique packages conform rather than define. This is architecturally clean and organizationally aggressive — it requires technique-package maintainers to accept the shared vocabulary as the source of truth for their interfaces.

**Recommendation tilt:** (b) is the honest minimum; (c) matches the current code shape and gives the full cross-UI benefit; (d) is the long-term endpoint but not reachable as a starting position. Start at (c), aim at (d), accept that (d) requires organizational alignment not yet in place.

### 2. Physical home and ownership

Where does the shared vocabulary live as a code artifact, and who owns it?

**Options:**

- **(a) In the `scipp/ess` monorepo as a new subpackage.** Alongside ess.powder, ess.sans, ess.reduce, etc. Maintained by the same team (in whatever form that team exists) that maintains the technique packages. Released as part of the monorepo's cadence.
- **(b) In the `scipp/ess` monorepo but with an independently-scoped ownership.** Formal sub-ownership within the monorepo — a small group is responsible for vocabulary decisions, distinct from technique-package maintainers.
- **(c) As a separate repository with its own release cycle.** `scipp/ess-schemas` or similar. Pinned by consumers (technique packages, livedata). Released on its own schedule.
- **(d) Inside `ess.livedata` for now; extract once a second concrete consumer exists.** Keep it in-repo, enforce the dependency boundary (pydantic + stdlib only, no scipp, no livedata internals), extract only when the demand is demonstrated.

**Tradeoffs:**

(a) maximizes proximity to the technique-package maintainers but exposes the vocabulary to monorepo cadence. Any fix to vocabulary requires a monorepo release. For a vocabulary that is the input-validation surface of the livedata dashboard during beam time, this cadence may be incompatible with operational reality.

(b) is an organizational refinement of (a): the same physical location, but vocabulary changes go through a different reviewer pool with different priorities. Helps with the fix-latency problem only if the vocabulary-owning team has cadence autonomy, which monorepos typically do not grant.

(c) decouples release cadence. Vocabulary ships when it is ready. Consumers pin. This is the "cleanest" option on paper and the most operationally complex — now every consumer has a version matrix, dependency bumps become routine PRs, and the failure modes of pinning (version mismatch, rollback asymmetry) show up in full.

(d) is the cautious staging strategy: do the work, enforce the boundary, prove the shape, and extract only when the payoff is concrete. Keeps fix latency at livedata-deploy speed until extraction. The cost is that other consumers (ess.powder ipywidget UI, NICOS) have to wait until extraction to participate — and (a) through (c) would also have to wait on their own readiness, so "wait for extraction" is not obviously worse than "wait for everything".

**Recommendation tilt:** (d) as the immediate move, (a) or (c) as the long-term home contingent on organizational buy-in and on a demonstrated second consumer. The worst option is "extract early because it will be harder later" — extraction gets no easier if the boundary is unclear; extraction gets much easier if the boundary has already held under real use.

### 3. Authority for interface evolution

Who decides what goes in the vocabulary, and who decides how it changes?

This is the question that most directly determines whether the unification is sustainable. It is also the question most easily deferred, which is precisely what makes it dangerous.

**Options:**

- **(a) Technique packages lead.** Each technique package owns the authoritative declaration of its workflows' interfaces. The shared vocabulary mirrors what technique packages publish. Vocabulary PRs follow technique-package PRs.
- **(b) Vocabulary leads.** The shared vocabulary declares the contract; technique packages conform. New workflows are proposed at the vocabulary level; technique packages implement them.
- **(c) Vocabulary is the only place interfaces are declared.** Technique packages own algorithms; interfaces live only in the shared vocabulary. Technique-package reduction code imports its own parameter types from vocabulary.
- **(d) Each instrument team owns its slice.** The DREAM team owns DREAM specs wherever they live; the LoKI team owns LoKI; a small framework-maintenance group owns cross-cutting types.

**Tradeoffs:**

(a) is the natural default and by far the most expensive at the failure mode that matters. Every non-trivial technique-package change becomes a two-release dance: technique package release, then vocabulary release, then livedata bump. For routine feature work (not emergencies) this multiplies the coordination cost of every change in every technique package. It also makes vocabulary a *lagging* surface — livedata cannot adopt a new technique-package feature until vocabulary has released to describe it.

(b) inverts the authority arrow and is unrealistic in practice. Domain expertise lives in the technique packages; asking vocabulary maintainers to gate reduction-interface design second-guesses the people with the expertise. It also has no leverage: technique packages have their own users who are not livedata, and those users will not tolerate "you must propose to vocabulary first" as the development path.

(c) consolidates interface ownership in vocabulary while leaving algorithms with technique packages. This is the architecturally correct answer and requires organizational alignment — technique-package maintainers must accept vocabulary as their canonical interface declaration site. Reachable in the long term, not as a starting position, and only if the vocabulary team is responsive enough that it does not become a bottleneck. The risk is that if vocabulary cannot keep up, technique packages develop shadow interfaces and the unification quietly dies.

(d) distributes ownership along instrument-team lines, which matches how ESS is actually organized. Crosscuts technique-package ownership because most instruments use multiple techniques. Works if coordinated; a mess otherwise.

**The generative observation** is that the authority question is not just "who" but also "how fast". A process where the vocabulary-owning group reviews every vocabulary change within hours is survivable under (a) or (c); the same process with week-long review cycles is fatal under both. The decision is therefore really about *responsiveness commitments*, not just nominal authority.

### 4. Compute-backend contract

What contract does a workflow spec impose on a compute backend that wants to satisfy it?

**Options:**

- **(a) Spec plus factory.** The spec declares the interface; somewhere (runtime-side, not in vocabulary) there is a factory function that consumes a spec and returns a runnable pipeline. One pattern shared between streaming and batch.
- **(b) Spec plus two adapters.** Explicit streaming adapter and batch adapter per workflow, each implementing the spec. Clearer boundaries; more code.
- **(c) Spec plus callbacks.** Spec declares not just interface but also callback hooks the backend invokes (e.g. "on new message batch, call this"). Moves behaviour into the spec; weakens the vocabulary-vs-compute separation.
- **(d) No explicit contract.** Each backend interprets specs on its own terms; the vocabulary describes interface only. Consumers coordinate by convention.

**Tradeoffs:**

(a) is the natural generalization of livedata's current factory pattern. A `WorkflowFactory` lookup by `WorkflowId` returns something runnable. Streaming and batch backends each maintain their own factory but key off the same IDs. Minimal vocabulary surface.

(b) is explicit: both backends write their own adapter per workflow. Less magic, more repetition. For a world with dozens of workflows and two backends, that repetition adds up.

(c) pushes logic into vocabulary — at the cost of vocabulary carrying streaming-shaped concepts. Violates the dependency-minimal premise.

(d) is "vocabulary is strictly vocabulary"; everything about how to execute a spec is left to each backend. Maximally flexible, minimally unified — backends can drift because nothing enforces that they interpret the spec the same way.

**Recommendation tilt:** (a), with the factory lookup living in each backend rather than in vocabulary. Vocabulary stays vocabulary; backends share a lookup pattern by convention, not by enforcement.

### 5. UI-surface contract

How do UIs consume specs, and what does the vocabulary promise them?

**Options:**

- **(a) Specs are self-describing; UIs auto-render.** Parameter models carry enough metadata (pydantic `Field` descriptions, units, ranges) for a generic rendering layer to build a form. Output schemas carry enough for a generic plotter to pick a visualization.
- **(b) Specs describe; UIs render bespoke.** Each UI writes its own rendering logic per spec type. Vocabulary is consumed for structure, not for rendering.
- **(c) Shared widget library layered over specs.** A dedicated rendering package (separate from vocabulary) provides UI components that know how to render specs. UIs import from this library.

**Tradeoffs:**

(a) is the lowest-friction path for new UIs — a dashboard and an ipywidget UI both get forms for free. The cost is that the generic renderer has to handle every parameter shape well, or it falls back to ugly-but-functional, and every UI that wants non-generic behaviour has to override the generic. Effectively: simple for 80% of cases, painful for the 20% where a custom widget was wanted.

(b) is the current state. Every UI does its own thing. The unification gives them a common vocabulary to render from; they still do their own rendering. No progress on UI-code reuse.

(c) is the maximum reuse position: a shared widget library means the livedata dashboard and the ipywidget UIs literally use the same components. The cost is a fourth package to maintain with its own release cadence and its own set of UI-framework compatibility concerns (Panel vs ipywidgets vs whatever NICOS uses is not trivial).

**Recommendation tilt:** (a) for immediate unification; (c) as an aspiration for long-term consolidation of the dashboard/ipywidget UI code but not in initial scope. The schemas plan currently proposes (a) by default — pydantic `Field` metadata is the rendering input.

### 6. Validator policy and wire format

Pydantic validators are Python callables. They do not round-trip through JSON. Yet validators are exactly where the interesting input-validation logic lives today (range checks, consistency checks, instrument-configuration-specific rules).

**Options:**

- **(a) Validators are Python-side enrichment only.** The wire format is the declarative fields; out-of-process consumers (NICOS over Kafka, JSON-Schema exports) either reimplement validation in their own language or tolerate its absence. Validators are a convenience for Python consumers, not part of the contract.
- **(b) All consumers are Python.** The contract is the Python classes, not a wire format. Anyone consuming a spec imports the classes; wire representation is implementation detail.
- **(c) Dual-mode per validator.** Some validators are declared wire-translatable (reimplementable from schema metadata alone); others are explicitly Python-side. A tagging discipline on every validator.

**Tradeoffs:**

(a) is operationally friendly: validators can evolve at deploy speed without forcing cross-repo releases. It is also weakly validated from an out-of-process consumer's perspective — NICOS receives a spec, cannot run the Python validator, and must either reimplement or accept invalid inputs through.

(b) preserves validator authority but forecloses any consumer that cannot import the Python classes. Locks the ecosystem into a Python-only integration story.

(c) is correct in principle and painful in practice. Every validator needs a tag. The tagging discipline must survive years of drift. In practice, "which category does this validator fall in?" becomes a per-validator design decision that slows down validator writing.

**Recommendation tilt:** (a), with an explicit acknowledgement that it closes the door on Python-free consumers for the foreseeable future. NICOS integration, if it happens, accepts this. If it does not accept this, the decision must be reopened — but committing to (b) because NICOS is hypothetical trades a concrete cost (beam-time latency) for a hypothetical benefit.

### 7. Versioning and cadence

How do consumers pin the vocabulary, and how do vocabulary releases relate to consumer releases?

**Options:**

- **(a) Unified cadence.** Vocabulary, reduce, technique packages, livedata all release together. Monorepo-style. Eliminates version mismatch; adds coordination friction.
- **(b) Independent cadence with pinned dependencies.** Each consumer pins a vocabulary version. Mismatches possible during rolling deploys; requires backward-compat discipline in vocabulary releases.
- **(c) Head-tracking.** Consumers always pin the latest vocabulary; vocabulary maintains aggressive backward compatibility. Lowest coordination cost; highest discipline requirement.
- **(d) Vocabulary is always N+0 with consumers.** No backward compatibility guaranteed; consumers rebuild against each vocabulary release. Fast for vocabulary development, brittle for operations.

**Tradeoffs:**

(a) works for what is already a monorepo (scipp/ess). If livedata is in a separate repository, livedata's cadence does not sync naturally with the monorepo's — consumers of a monorepo-released vocabulary pin a specific version, and we are back to (b).

(b) is the realistic default for a cross-repository setup. It requires a compatibility matrix (which vocabulary versions each livedata release supports), CI discipline to prevent mismatched installs, and backward-compat conventions in vocabulary (new fields optional with defaults for at least one release).

(c) is simpler than (b) if the discipline holds. Every vocabulary release is a non-breaking superset; consumers auto-upgrade on their next release. The risk is that the discipline fails once, silently, and consumers break mysteriously.

(d) trades operational stability for development velocity. Acceptable for pre-1.0 vocabulary; unacceptable for a production-critical surface.

**Recommendation tilt:** (b) with backward-compat discipline (new fields optional, deprecation cycles for breaking changes) during initial releases; aim for (c) once the surface stabilizes.

### 8. Instrument-specific and technique-specific specialization

Where do `DreamPowderWorkflowParams(PowderWorkflowParams)` and similar specializations live, given that they cross instrument × technique?

**Options:**

- **(a) All in vocabulary.** Vocabulary has per-instrument and per-technique subpackages; the full specialization hierarchy lives there.
- **(b) Split by ownership.** Generic parameter models live in vocabulary. Technique-specific specializations (`PowderWorkflowParams`) live with the technique package. Instrument-specific specializations of those (`DreamPowderWorkflowParams`) live in vocabulary's instrument subpackage.
- **(c) Per-instrument lives with the instrument team.** Vocabulary carries generic framework types only. Each instrument's specializations live in an instrument-specific module — possibly in vocabulary, possibly in a separate package, owned by the instrument team.
- **(d) Composition over inheritance.** Avoid the specialization hierarchy entirely. Parameter models carry instrument- and technique-specific fields via composition; the hierarchy is flat.

**Tradeoffs:**

(a) keeps everything co-located but forces vocabulary to carry per-instrument, per-technique knowledge. The vocabulary grows large. Every technique-package change that affects specializations becomes a vocabulary change.

(b) follows ownership: vocabulary owns cross-cutting types; technique packages own technique-specific shapes; instrument specializations live where instrument identity lives (vocabulary). This introduces a dependency edge: vocabulary's instrument subpackages import from technique packages. If technique packages import from vocabulary (as they must for the generic types), we have a potential circular dependency unless carefully broken.

(c) pushes per-instrument specializations to the edge. Vocabulary is smallest; instrument teams can move fast within their subpackages. The cost is that cross-instrument consistency becomes unenforceable — if DREAM's specialization pattern differs from LoKI's, nobody catches it at vocabulary-review time.

(d) sidesteps hierarchy at the cost of duplication. Flat parameter models per (instrument, technique) combination. Easy to reason about locally, tedious across the catalog, and no shared base means a bulk change to all "powder params" requires touching every instance.

**Recommendation tilt:** (b), with the dependency edge (vocabulary → technique packages) accepted as the direction the arrow goes. The alternative — technique packages not importing from vocabulary, or vocabulary not knowing about technique specializations — foregoes the unification.

### 9. Streaming-specific fields

Parameters that exist only because a workflow runs live — accumulation window, rate limit, cadence, incremental-binning resolution — do not apply to batch execution. Where do they live?

**Options:**

- **(a) Vocabulary is batch-native; livedata adds streaming overlays via subclasses.** `MonitorDataParams` in vocabulary has no streaming fields; `LiveMonitorDataParams(MonitorDataParams)` in livedata adds them.
- **(b) Vocabulary admits optional streaming fields.** Fields marked as "live-only"; batch consumers ignore them. Single hierarchy.
- **(c) Parallel hierarchies.** `BatchMonitorDataParams` and `LiveMonitorDataParams` both subclass a shared abstract base.

**Tradeoffs:**

(a) is clean but doubles the subclass tree: vocabulary already uses subclassing for instrument specialization (D1), and streaming adds a second axis. `DreamMonitorDataParams(MonitorDataParams)` in vocabulary and `LiveDreamMonitorDataParams(DreamMonitorDataParams, LiveMonitorParams)` in livedata — multiple inheritance and linearization problems may follow.

(b) is smallest-diff but leaks: batch consumers carry unused fields, vocabulary releases force re-verify on live-only changes, and the vocabulary contract becomes polluted with streaming concepts.

(c) is most explicit but most verbose. Three hierarchies for a single concept.

**Recommendation tilt:** (a). Accept the subclass complexity as the price of the separation. Livedata owns the "Live" subclasses; vocabulary does not know they exist.

### 10. Cross-cutting reduction primitives (detector views, histogramming)

Detector-view binning (pixel indices to logical coordinates), monitor histogramming, ROI reduction, TOF binning — currently live in livedata. These are cross-instrument data-shaping primitives; they apply equally to batch and streaming.

**Options:**

- **(a) Move to `ess.reduce`.** Audit each for streaming entanglement, decompose into pure transform + reactive wrapper, move the pure half.
- **(b) Leave in livedata.** Accept that livedata is the home for these transforms; batch consumers do without or reimplement.
- **(c) Per-technique ownership.** Detector views specific to a technique move to that technique package; generic ones to ess.reduce.

**Tradeoffs:**

(a) is the ambitious position and requires real work: not every transform factors cleanly (ROI feedback, dynamic detector geometry for e.g. LoKI's moving carriage, accumulator state are genuinely streaming-shaped and do not move). The reactive half stays in livedata as thin wrappers over pure transforms.

(b) leaves a known duplication in place. Technique packages that want detector views reimplement. Acceptable short-term; contradicts the unification goal long-term.

(c) distributes ownership along technique lines. Works if each detector view is technique-scoped (often not — DREAM has both powder and spectroscopy-shaped views).

**Recommendation tilt:** (a), as a separate migration plan with its own audit, not a consequence of vocabulary extraction. Requires the ess.reduce maintainers to accept the scope expansion.

### 11. Migration path

How do we get from here (duplicated specs, no shared vocabulary) to there (unified, cross-consumer)?

**Options:**

- **(a) Big bang.** One coordinated release: vocabulary ships, livedata adopts, technique packages adopt, ipywidget UIs adopt. Eliminates intermediate-state complexity; maximum coordination risk.
- **(b) Vocabulary first, inside livedata.** Carve the vocabulary subpackage out inside livedata with a strictly enforced dependency boundary. Prove the shape, prove the CI discipline, prove the test isolation. Only then extract as a separate package, only then onboard other consumers.
- **(c) Per-consumer incremental.** Extract vocabulary early; onboard one consumer at a time (start with livedata, add ess.powder ipywidget, then NICOS, etc.). Long-running parallel state; each consumer's onboarding is its own project.
- **(d) Per-technique incremental.** Extract vocabulary early; unify one technique at a time (powder first, prove end-to-end across livedata and ess.powder, then SANS, etc.). Each technique stresses a different shape.
- **(e) Consolidate but do not extract.** Do (b) but never take step 2 — vocabulary lives inside livedata indefinitely, with the boundary held. Other consumers pay the cross-repo cost only if and when they become concrete.

**Tradeoffs:**

(a) avoids intermediate states but requires every team's agreement simultaneously. Unreachable in practice unless there is a single body of authority over all stakeholders, which ESS does not have.

(b) is the pragmatic staging: internal separation captures most of the benefit (testability, boundary discipline, forcing design decisions) without the coordination cost of extraction. Extraction happens when the payoff is concrete.

(c) and (d) are both "extract early, onboard slowly" strategies. Differ in which axis drives. (c) is consumer-driven (who wants in first?); (d) is workflow-driven (which shape stresses the design?). Either is slow and both create long-running parallel state — multiple ways the same workflow might be described at once.

(e) is the honest null hypothesis. If the payoff of extraction never becomes concrete, consolidation without extraction is a complete outcome. Not a failure.

**Recommendation tilt:** (b) as the immediate move; (d) with powder as the first technique if extraction proceeds; (e) as the acceptable endpoint if the second consumer never materializes.

### 12. Release gates and breakage discipline

How do we prevent a vocabulary change from silently breaking a consumer?

**Options:**

- **(a) Integration tests per consumer.** Every consumer runs its own integration suite against pinned vocabulary. Vocabulary releases that break any consumer's suite are blocked.
- **(b) Contract tests in vocabulary.** Vocabulary package itself contains contract tests that every supported consumer shape must satisfy. Maintained by vocabulary team.
- **(c) Consumer CI pulls vocabulary head.** Consumers run a nightly CI against vocabulary head and report back. Doesn't block vocabulary releases but provides early warning.
- **(d) No gates beyond pydantic's own schema-compatibility.** Consumers pin versions; if they break, they un-pin.

**Tradeoffs:**

(a) is the most thorough and most expensive. Vocabulary's release process must wait on every consumer's CI, which means vocabulary moves at the speed of its slowest consumer.

(b) inverts the ownership: the vocabulary team writes and maintains tests that represent consumer needs. Requires that the vocabulary team know what consumers need, which they may not.

(c) is early-warning without blocking. Less reliable but operationally lightest.

(d) accepts that mismatches will happen and trusts version pinning to contain them. Works if backward-compat discipline is tight; fails silently if it is not.

**Recommendation tilt:** (c) initially, with (a) reserved for the critical consumers (livedata, any technique package with live-beam-time use).

## Development implications

Concrete scenarios that illustrate what changes in the developer experience end-to-end. Each is stated as "today" vs "post-unification" to make the delta visible.

**Adding a parameter to a workflow.** Today, a scientist adding a new parameter to DREAM powder reduction edits livedata's `DreamPowderWorkflowParams`, the livedata factory, and associated tests, in one PR. Post-unification, the scientist edits the shared vocabulary's declaration (which may live in a different repo), waits for a vocabulary release, bumps the livedata dependency, adjusts the livedata factory, and lands. In parallel, the same scientist (or someone coordinated with them) updates ess.powder's implementation to accept the new parameter, or discovers that the parameter change actually requires reworking the ess.powder signature, which extends the coordination window. Best case: a few days. Worst case: a week or more of cross-repo review and release.

**Fixing a validator during beam time.** Today, an instrument scientist notices at 2am that a validator is rejecting valid inputs; the fix is a one-line PR to livedata, reviewed by on-call, redeployed. ~15 minutes. Post-unification, the same fix goes to wherever the validator lives — if it is in vocabulary, the fix is a cross-repo PR, a release, and a dependency bump before it can deploy to the running instrument. Mitigation is to keep validators light (structural only) or to introduce a livedata-side override mechanism for temporary relaxation. Both cost something: light validators move constraint logic into runtime config, override mechanisms introduce a second authority over the same invariants.

**Adding a new instrument.** Today, a new instrument's addition is a new subpackage under livedata's `config/instruments/`. Post-unification, it is a new subpackage in vocabulary (or wherever per-instrument specs live), possibly new transforms in ess.reduce, possibly new streaming overlays in livedata, possibly new technique-package entries if the instrument's techniques need specialization. Three-repo coordination instead of one-repo. Instrument-scientist contribution path becomes harder exactly at the moment it matters most (new instrument onboarding is already a high-coordination moment).

**A technique package bumps a reduction interface.** Today, if ess.powder renames a parameter or adds a required field, livedata's adapter may or may not even notice — livedata has its own parameter model and its own reduction entry point. Post-unification, the vocabulary's `PowderWorkflowParams` tracks ess.powder's signature directly; the change propagates as technique-package release → vocabulary release → livedata bump. The routine-feature cost of every technique-package release doubles or triples. Over a year, across five or six technique packages, the cumulative coordination cost is where the budget actually goes.

**A new contributor fixes a monitor-histogramming bug.** Today, they grep in livedata, find the bug, fix, test, PR. Post-unification, they must first figure out which layer owns the bug (vocabulary? ess.reduce? livedata? a technique package?), clone the relevant repo, set up its editable install alongside livedata, run cross-layer tests, figure out release flow, and land a cross-repo change. Hours of first-time overhead. Mitigation is documentation; non-trivial documentation that has to be kept current; the on-ramp has to be tolerable or contribution volume drops.

**Cross-layer debugging.** Today, a dashboard bug that shows wrong values traces through one codebase. Post-unification, the trace goes through vocabulary (is the spec wrong?), through livedata (is the adapter wrong?), possibly through ess.reduce (is the transform wrong?), possibly through a technique package (is the algorithm wrong?). Each boundary is a place the bug could hide. Observability must support "which layer" as a first-class question — today it does not.

## Production implications

Parallel scenarios on the operations side. All of these assume extraction has happened; during the in-repo consolidation phase they do not apply.

**Rolling deploy during an active run.** Today, the new backend and the dashboard come from the same codebase and agree on spec shapes by construction. Post-unification, during a rolling deploy, backend instances at vocabulary v1.3 run alongside a dashboard at v1.2 for the duration of the deploy. Fields present in v1.3 but absent in v1.2 render as ghost inputs. Fields required in v1.3 but unknown to v1.2 cause validation failures on arrival. Mitigation: backward-compat discipline on vocabulary releases (new required fields forbidden without deprecation cycles) and tight deploy windows.

**Rollback during an incident.** Today, rollback is a livedata-only action and is usually safe. Post-unification, rollback of livedata while vocabulary stays at a newer version is safe only if livedata did not depend on new vocabulary features. Rollback of vocabulary is almost never the right move because livedata references specific fields. On-call needs a rehearsed playbook: the default is almost always to roll livedata back, not vocabulary, and the artifact to roll back must include the vocabulary version pin so the environment returns to a known-good state, not just the livedata code.

**Validator change during production.** Today, a validator change deploys with a livedata push. Post-unification, with validator-policy (a) (Python-side enrichment), validator logic can stay in livedata config and deploy at livedata speed. With policy (b) (all consumers Python, validators authoritative), a validator change requires a vocabulary release — which during beam time may be operationally blocked by the monorepo's release cadence. The validator-policy decision is where production operational flexibility lives.

**Scheduled vocabulary release during beam time.** If vocabulary releases on a regular cadence and a release lands during an ESS run, livedata must decide whether to pin the new version immediately, wait, or skip. Mitigation is beam-time-aware release calendars in the vocabulary-owning team. Requires that the vocabulary-owning team knows and respects ESS run schedules.

**Cross-team on-call.** Today, livedata's on-call is one person with merge rights on one repo. Post-unification, an urgent fix may require merge rights on vocabulary and/or a technique package. Either on-call becomes cross-team (organizational commitment), or urgent fixes have paths that bypass normal review (operational risk), or urgent fixes have to wait (acceptance of longer incident windows).

**Version-mismatch detection.** Today there is nothing to detect because there is no mismatch possible. Post-unification, CI must verify that all livedata components share compatible vocabulary versions; runtime must emit the vocabulary version it was built against so that version-mismatch incidents can be correlated with deploy events. Neither exists today and neither is free to build.

**Observability across layers.** A dashboard value is wrong. Which layer? Observability today logs livedata-internal state. Post-unification, meaningful diagnosis requires layer-aware logging: which vocabulary version produced this spec, which ess.reduce transform handled this data, which livedata adapter wired them together. If this observability is not in place before extraction, the first post-extraction incident is also the first time anyone tries to trace through layers in production — a bad combination.

## What could force a reversal

Falsification conditions. The direction is worth pursuing only if it remains reversible on evidence. Reasons to step back — either during consolidation or during extraction — rather than push through:

- **The second consumer never materializes.** If eighteen months after consolidation no technique-package ipywidget UI and no NICOS integration is pulling from the vocabulary, extraction is speculation. Consolidate, stay in-repo, move on.
- **Fix-latency incidents cluster around vocabulary changes.** If the on-call log reveals that vocabulary-side fixes dominate incidents after extraction, the operational cost is real, not theoretical. Either validator policy (a) is non-negotiable, or the direction is wrong for this operational reality.
- **Technique-package maintainers reject spec conformance.** If, when asked to declare their workflows against the shared vocabulary, technique-package maintainers push back citing cadence, authority, or design disagreement, the unification is organizationally infeasible. Fall back to vocabulary-used-only-by-livedata and accept that cross-compute unification will not happen.
- **ess.reduce cannot absorb the transforms.** If the ess.reduce maintainers see detector-view logic as livedata-specific clutter, the cross-compute unification in compute terms is blocked. Livedata keeps the transforms; unification stays at the vocabulary layer only.
- **Version mismatches become routine.** If mismatches appear more than once per quarter despite CI discipline, the pinning strategy is wrong. Tighten or reverse.
- **Dev-loop friction is the top contributor complaint.** If new contributors consistently fail to land changes because of cross-repo coordination, the cost is higher than anticipated.
- **Cross-team coordination does not form.** If the vocabulary-owning team is not responsive, validator fixes take days, interface changes take weeks, and livedata maintainers start working around vocabulary rather than through it — the unification is failing quietly. Detect and reverse.

These are conditions, not predictions. The point is that the direction is falsifiable: we can tell whether it is working, and we can stop if it is not.

## Open questions for explicit resolution

Questions this document cannot answer alone. They need the relevant stakeholders in the room before extraction starts, and ideally before consolidation ends.

1. Do the technique-package maintainers (powder, SANS, reflectometry, diffraction, spectroscopy) accept that their workflows become first-class entries in a shared vocabulary, declared according to that vocabulary's conventions? If not in full, what subset?
2. Does the ess.reduce maintainer group accept the scope expansion of absorbing detector-view, monitor-histogramming, and related cross-cutting transforms currently in livedata? With what timeline?
3. Is there a second concrete consumer of the vocabulary today (a technique-package ipywidget UI in development, a NICOS integration with real requirements)? If not, what is the plausible timeline to one? This is the primary gate on extraction.
4. What is the scipp/ess monorepo's current release cadence, and what is the process for triggering an out-of-cycle release for an urgent fix? This determines whether validator policy (a) or (b) is operationally viable.
5. Does NICOS need JSON-Schema exports or Python-class imports? The answer determines whether the validator-policy question is a real decision or a forced choice.
6. Do the technique packages currently treat their workflow interfaces as public APIs with deprecation discipline, or as internal-to-livedata structures free to change? The answer determines whether authority-option (a) is expensive or ruinous. Needs a survey of recent technique-package changelogs, not a guess.
7. Who is responsible, post-extraction, for coordination costs across repos? If the answer is "the livedata maintainer absorbs it", that is an explicit organizational ask; it should not be implicit.
8. Is the goal extraction, or is the goal internal separation with a clear exit path if demand appears? These are different commitments with different cost structures, and they are easy to conflate.
9. What observability exists today to diagnose layer-crossing bugs, and what would need to exist before extraction is operationally safe? If the answer is "none" and "substantial", that work is on the critical path, not a follow-up.
10. Is there a meaningful way to auto-generate parts of the vocabulary from technique-package type annotations, so that vocabulary tracks technique-package interfaces without requiring a separate release per change? Speculative but potentially consequential.

## Recommendation tilt (non-binding)

A coherent stance reachable today, with the decisions above made explicit:

Consolidate in-repo. Carve out the vocabulary subpackage inside livedata with a strictly enforced dependency boundary (pydantic + stdlib only). Replace `sc.DataArray` output templates with a shape-only schema. Hold the boundary with CI. This captures most of the internal benefit — testability, separation, forcing design decisions — with none of the cross-repo coordination cost. It also makes extraction materially easier later by proving the boundary is holdable.

Treat extraction as gated on a second concrete consumer, not speculation. If a technique-package ipywidget UI or NICOS integration appears with real code and real requirements, begin extraction. If eighteen months pass without one, accept that consolidation is the endpoint.

Commit to validator-policy (a) — validators are Python-side enrichment, not wire contract — because beam-time fix latency is concrete and NICOS-over-wire is hypothetical. Revisit only if the hypothetical becomes concrete.

Treat the migration of cross-cutting transforms to ess.reduce as a separable plan with its own audit, requiring ess.reduce-maintainer buy-in. Do not let it block vocabulary work; do not let it be silently deferred by vocabulary work either.

Scope the vocabulary at option 1(c): framework types plus generic parameter models plus per-instrument specializations, but not per-technique specializations yet. Per-technique absorbs into vocabulary only once technique-package maintainers are concrete participants, not proxies.

Make explicit, before any extraction, the responsiveness commitment from the vocabulary-owning team. Hour-scale review for urgent fixes, day-scale for routine changes, week-scale for design decisions. Without this commitment, authority-option (a) is ruinous regardless of what the org chart says.

The failure mode to avoid is not "extracted too early" or "extracted too late." It is "extracted without deciding the validator policy, the responsiveness commitment, and the second-consumer gate." Those three questions, resolved, make the direction survivable whether extraction happens in a year or never. Left unresolved, any version of the direction fails under pressure.
