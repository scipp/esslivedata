# Survey: workflow / parameter / output descriptions and discoverability

> **Status.** Most proposals below are now implemented: declarative param↔output
> coupling (`OutputView.params` + `WorkflowSpec` resolvers) surfaced as a back-reference
> in the plot wizard, a forward-reference ("Affects outputs") in the workflow-config
> modal, and hover tooltips on the Workflows-card output chips; the TOA pulse-period
> explanation + `ESS_PULSE_PERIOD_MS` constant; friendly enum labels (e.g. "Time of
> arrival (TOA)"); the filled content gaps; and the "Window" → "Time Window" rename. The
> shared glossary (proposal 6) is tracked separately as issue #1019.

Context: issue #993 (Bifrost notes) flags scattered description gaps. This survey puts a
new user — limited ESS/instrument knowledge — in front of the dashboard and walks the
workflow → output → plot flow to find where they get lost. It covers both *content*
(missing/weak descriptions) and *structure* (where the UI fails to connect params,
outputs, and plots). Findings are grouped by leverage, highest first.

## How the description model works today

| Concept | Where defined | Title | Description |
|---|---|---|---|
| Workflow | `WorkflowSpec` (`config/workflow_spec.py`) | `title` | `description` |
| Workflow params | pydantic `Field(title=, description=)` in param models | yes | yes (tooltip) |
| Output (view) | `OutputView(name, title, streams, description)` in `WorkflowOutputsBase` subclasses | yes | optional |
| Aux input | `AuxInput(title, description)` | yes | yes |
| Plotter | `PlotterSpec.description` (`plotter_registry.py`) | yes | yes |
| Plot-display params | pydantic `Field` in `dashboard/plot_params.py` | yes | yes |

There is **no declarative link between a parameter and the output(s) it affects.** The
relationship exists only inside workflow code (e.g. `MonitorDataParams.get_active_edges`
switches on `coordinate_mode`).

## The two parameter surfaces (root cause of most confusion)

A user configures two *different* parameter sets, in two *different* places, with no
signposting that they are different:

1. **Workflow params** — gear on a card in the **Workflows** tab. These change *what is
   computed* (TOA edges, coordinate mode, ranges, ROIs). Tabs: General / Coordinate Mode
   / Time of Arrival Edges / Time of Arrival Range.
2. **Plot-display params** — *Add layer…* wizard, "Step 3: Configure Plot". These change
   *how an already-computed output is displayed* (window, rate, scale, layout,
   orientation).

A newcomer cannot tell which surface owns a given knob. The clearest symptom: in the plot
wizard they pick the output **"Total in range"**, but the *range* that defines it is a
workflow param set on a different tab — the wizard gives no hint it exists or where it
lives. This is the concrete form of #993's "params that pertain only to certain outputs".

## Findings

### High — structural / discoverability

- **No param→output back-reference (the headline ask).** Outputs and the params that
  shape them are never connected in the UI. Examples of hidden coupling:
  - Monitor: `toa_range` only affects the **Total in range** output; `toa_edges` only the
    **Histogram** output; `coordinate_mode` gates both edge/range groups.
  - Detector view: `pixel_weighting` affects the image, not `total_counts`; ROI params
    affect `roi_spectra` only.
  The user has no way to learn this short of reading source. Recommend a declarative
  param↔output map (see Proposals) surfaced in both the workflow-config modal and the
  plot wizard.

- **Outputs are bare pills with no inline description or coupling.** On an expanded
  Workflows card, OUTPUTS render as titles only (`Histogram` / `Total` / `Total in
  range`). No description on hover, no indication of which config produced them. The
  description text *does* exist (`OutputView.description`) and *is* shown in the plot
  wizard's Step 1 — but not here, where the user first meets the outputs.

- **`0-71.42857142857143` TOA default is unexplained and ugly** (#993 "explain 0-71").
  Default is `stop=1000.0 / 14` — one ESS pulse period at 14 Hz (71.43 ms). Neither the
  field nor the group description says so, and the raw float leaks into the input box. A
  new user has no idea why the histogram spans 0–71 ms or what happens outside it.

- **Raw enum values shown as jargon.** Coordinate-mode dropdown displays `toa` (and would
  display `wavelength`). No human label; the only expansion of "TOA" is in the group
  blurb. Enum display values should be friendly ("Time of arrival (TOA)").

### Medium — content gaps (missing/weak descriptions)

- `dummy/specs.py:63` — workflow **"Panel 0"** has `description=''` (empty).
- `bifrost/specs.py` — `DetectorRatemeterParams.region` has a title but **no
  description**; sub-fields (arc, pixel_start/stop) unexplained.
- `bifrost/specs.py` — Q params use unexplained jargon: `q_edges` → "Q bin edges.";
  `QAxisSelection.axis` → "Cut axis."; elastic-map `axis1/axis2` → "First/Second cut
  axis." No definition of Q (momentum transfer), what a "cut" is, or axis options.
- `wavelength_lut_workflow_specs.py` — four `unit` fields described only as **"Unit."**
  (distance/time/Ltotal/cascade) — say which quantity.
- Monitor/detector edge & range params describe *what they are* but not *which output*
  they drive (see High item above).

### Low / already addressed

- Bifrost **"Elastic Q map"** rename (#993) is **already done** (`bifrost/specs.py`).
- `WindowParams` already carries a thorough `_WINDOW_DESCRIPTION`. #993's "Window → Time
  Window" remains only as the **tab/group label** ("Window"), derived from the field
  name — a one-line rename to "Time Window" if still wanted.
- Most workflow descriptions are present and decent; Beam monitor correctly expands
  "time-of-arrival (TOA)".

## What already works (don't regress)

- Plot wizard Step 1 shows workflow *and* output descriptions under the selected buttons.
- Plot wizard Step 3 shows the plotter description as a subtitle.
- Param group descriptions and per-field `(?)` tooltips render in both modals.
- Aux inputs have a role title + description.

## Proposals (rough order of value)

1. **Declarative param↔output coupling.** Add metadata tying each workflow param (or param
   group) to the output view(s) it affects — e.g. an `affects` field on the param via
   `Field(json_schema_extra=...)` or a map on `WorkflowOutputsBase`. Surface it two ways:
   (a) in the workflow-config modal, tag each param group with the outputs it feeds;
   (b) in the plot wizard Step 1, under the chosen output, list "Shaped by: Coordinate
   Mode, Time of Arrival Range" with a pointer to the workflow gear.
2. **Show output descriptions on the Workflows card** (hover/expand the OUTPUTS pills),
   reusing `OutputView.description`. Fill in the empty/weak ones first.
3. **Explain the TOA defaults in-context.** Add to the TOA-edges group description that the
   default range is one ESS pulse period (14 Hz ≈ 71.43 ms) and what falls outside it;
   consider rounding the default and naming the constant in code.
4. **Friendly enum labels** for coordinate mode (and similar), so dropdowns read
   "Time of arrival (TOA)" not `toa`.
5. **Fill the content gaps** in Medium above (dummy Panel 0, Bifrost ratemeter region,
   Bifrost Q jargon, wavelength-LUT "Unit.").
6. **A short glossary** (TOA, Q / momentum transfer, wavelength edges, window vs.
   reduction cadence) linked from the modals, so jargon is defined once.
7. Optional: rename the plot-config "Window" tab to "Time Window" (#993).

## Reproduction

UI walkthroughs were captured against a Kafka-free fake backend seeded from the dummy
fixture (`python scripts/drive_dashboard.py --launch`), driving the plot-creation wizard
and the workflow-config modal. The param↔output coupling is exercised by unit tests in
`tests/config/workflow_spec_test.py` (`TestParamOutputCoupling`).
