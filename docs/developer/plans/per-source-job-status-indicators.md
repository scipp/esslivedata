# Per-Source Job Status Indicators in WorkflowStatusWidget

## Motivation

A workflow runs one `Job` per source (e.g., DREAM has 5 detector banks, LOKI has 9).
The current header shows a single **worst-case badge** — if one source errors, the whole
workflow shows `ERROR`. This hides which sources are healthy and which aren't. For
operators diagnosing issues during a run, they need to know *which* detector bank has
a problem without expanding the card and hunting through logs.

## Current State

The header layout is:

```
[▶] Detector Projection    [ACTIVE]  10:32:15 (4m 12s)     [⟳] [■]
     expand   title          badge     timing                reset stop
```

`_get_status_and_timing()` iterates all `JobStatus` entries for the workflow and
reduces them to a single worst-case state, earliest start time, and first error
message. Per-source data (`job_status.job_id.source_name`, `.state`,
`.error_message`) is already available but discarded during aggregation.

## Proposal: Status Dots Row

Add a row of small colored circles ("status dots") to the header, one per source,
positioned between the badge and the timing display:

```
[▶] Detector Projection    [ACTIVE]  🟢🟢🔴🟢🟢  10:32:15 (4m 12s)   [⟳] [■]
                             badge    dots         timing
```

Each dot represents one source's `JobState`, colored using the existing
`STATUS_COLORS` palette. On hover, a tooltip shows the source name and state
(and error summary if applicable).

### Why dots instead of alternatives?

| Alternative              | Problem                                                      |
|--------------------------|--------------------------------------------------------------|
| Per-source text badges   | Too wide — 9 LOKI sources would overflow the header          |
| Expandable details panel | Requires user action; defeats at-a-glance monitoring         |
| Table in body            | Already have the staging area there; adds clutter            |
| Replace global badge     | Loses the quick summary; harder to scan across workflows     |

Dots are compact (fits 9 sources in ~100px), scannable, and degrade gracefully for
single-source workflows (one dot, redundant with badge — could be hidden).

### Visual Design

- **Size**: 8px diameter circles with 4px gaps
- **Colors**: Reuse `WorkflowWidgetStyles.STATUS_COLORS` mapping
- **Order**: Follows `workflow_spec.source_names` — the instrument-author-defined order.
  Matches the staging area's config group ordering. Meaningful to operators who know
  their detector layout (e.g., DREAM: mantle, endcap_backward, endcap_forward, HR, SANS).
- **Tooltip**: HTML `title` attribute — `"mantle_detector: active"` or
  `"sans_detector: error — ZeroDivisionError"`. Simple, zero-dependency. The native
  tooltip is slightly slow to appear; can upgrade to a CSS `::after` pseudo-element
  tooltip later if that becomes an annoyance.
- **Click**: None initially. Dots are 8px — too small for reliable click targets. If we
  later want click-to-navigate (e.g., jump to source's plot), dots would need to grow
  to 12–14px, changing the space budget. Better to validate the indicator first.
- **Single-source workflows**: Hide the dots row (the global badge is sufficient)
- **PENDING state** (no backend status yet): Show gray dots for all expected sources

### Source Count by Instrument

| Instrument | Typical sources/workflow | Notes                     |
|------------|--------------------------|---------------------------|
| DREAM      | 5                        | 5 detector banks          |
| LOKI       | 9                        | 9 detector banks          |
| NMX        | 3                        | 3 detector panels         |
| TBL        | 1–2                      | separate workflows        |
| Bifrost    | 1                        | unified detector          |
| Estia      | 1                        | single detector           |
| Odin       | 1                        | single detector           |

LOKI with 9 dots is the stress case. At 8px + 4px gap = 12px per dot, that's 108px
total — fits comfortably in the header.

## Implementation Sketch

### Data: extend `_get_status_and_timing()`

Return per-source status alongside the existing aggregated values. Minimal change:

```python
@dataclass(frozen=True)
class SourceStatus:
    source_name: str
    state: JobState
    error_summary: str | None

def _get_status_and_timing(self) -> tuple[str, str, str, str | None, list[SourceStatus]]:
    # ... existing loop, but also collect:
    per_source: list[SourceStatus] = []
    for job_status in ...:
        per_source.append(SourceStatus(
            source_name=job_status.job_id.source_name,
            state=job_status.state,
            error_summary=extract_error_summary(job_status.error_message)
                if job_status.error_message else None,
        ))
    # Order by workflow_spec.source_names (instrument-defined order)
    spec_order = {name: i for i, name in enumerate(self._workflow_spec.source_names)}
    per_source.sort(key=lambda s: spec_order.get(s.source_name, len(spec_order)))
    return status_text, status_color, timing_text, error_html, per_source
```

### Rendering: HTML dots with CSS tooltips

```python
def _make_status_dots_html(self, sources: list[SourceStatus]) -> str:
    if len(sources) <= 1:
        return ''
    dots = []
    for s in sources:
        color = WorkflowWidgetStyles.STATUS_COLORS.get(
            s.state.value, WorkflowWidgetStyles.STATUS_COLORS['active']
        )
        tooltip = f"{s.source_name}: {s.state.value}"
        if s.error_summary:
            tooltip += f" — {s.error_summary}"
        dots.append(
            f'<span title="{tooltip}" style="'
            f'display: inline-block; width: 8px; height: 8px; '
            f'border-radius: 50%; background: {color}; '
            f'margin: 0 2px; cursor: default;'
            f'"></span>'
        )
    return (
        '<span style="display: inline-flex; align-items: center; gap: 0px;">'
        + ''.join(dots)
        + '</span>'
    )
```

### Header layout change

Insert a new `pn.pane.HTML` between the badge and timing spacer:

```python
self._status_dots = pn.pane.HTML(
    self._make_status_dots_html(per_source),
    height=WorkflowWidgetStyles.HEADER_HEIGHT,
    styles={'display': 'flex', 'align-items': 'center'},
)

header = pn.Row(
    self._expand_btn,
    title_html,
    pn.Spacer(width=12),
    self._status_badge,
    pn.Spacer(width=8),
    self._status_dots,       # NEW
    pn.Spacer(width=12),
    self._timing_html,
    ...
)
```

### Refresh: update dots in-place

In `refresh()`, update the dots pane alongside badge and timing:

```python
def refresh(self):
    ...
    status, status_color, timing_text, _, per_source = self._get_status_and_timing()
    # ... existing badge/timing updates ...
    if self._status_dots is not None:
        new_dots = self._make_status_dots_html(per_source)
        if self._status_dots.object != new_dots:
            self._status_dots.object = new_dots
```

No new polling mechanism needed — piggybacks on the existing 500ms refresh cycle.

### PENDING state handling

When no backend status has arrived yet, we still know the expected sources from the
active config:

```python
if not has_fresh_backend_status:
    active_config = self._orchestrator.get_active_config(self._workflow_id)
    if active_config:
        per_source = [
            SourceStatus(name, JobState.scheduled, None)
            for name in sorted(active_config.keys())
        ]
```

## Scope and Risks

**In scope**:
- Status dots in header (rendering + refresh)
- Tooltip with source name + state + error summary

**Out of scope** (future work):
- Click-to-filter: clicking a dot to highlight that source's output plots
- Per-source timing display
- Per-source error detail expansion

**Risks**:
- *Tooltip UX*: HTML `title` tooltips are slow to appear and unstyled. If we want
  richer tooltips (instant, styled), we'd need a CSS-only tooltip or Bokeh HoverTool.
  Starting with `title` keeps it simple; can upgrade later.
- *Color accessibility*: Red/green distinction is problematic for color-blind users.
  The global badge has text ("ERROR"/"ACTIVE") as a fallback. Dots alone rely on
  color — could add shape variation (e.g., `×` for error) as a follow-up.

## Estimated Effort

Small change — touches one file (`workflow_status_widget.py`), no backend changes,
no new dependencies, no new polling/callback machinery.
