# Plan: Hide window mode for cumulative outputs (#800)

## Problem

When a user selects a cumulative output (e.g., "Image (cumulative)") and sets window
mode to `window`, the system crashes with a confusing `KeyError: 'time'` at runtime.
Cumulative outputs have no `time` coordinate — they represent data accumulated since
the start of the run, so windowing and rate normalization are meaningless.

The crash occurs in `WindowAggregatingExtractor.extract()` (`extractors.py:173`) or
`TemporalBuffer._initialize_buffers()` (`temporal_buffers.py:417`), both of which
unconditionally access `data.coords['time']`.

Additionally, the toolbar displays "(latest)" for cumulative outputs (e.g.,
"Workflow A: cumulative (latest)"), which is confusing — cumulative data is always
the latest accumulated value; window mode is not a meaningful concept.

## Design considerations

### Where does the "is windowing applicable?" knowledge belong?

Three places need the same information:

1. **Config modal** — hide `window`/`rate` fields when not applicable
2. **Toolbar label** — omit window mode display for cumulative outputs
3. **Extractor creation** — avoid creating `WindowAggregatingExtractor`

The source of truth is the **output template**: cumulative outputs have no `time`
coordinate. The question is where to derive and propagate this.

### Why not model-intrinsic `hidden_fields`?

We considered having param models declare their own visibility rules (e.g.,
`WindowParams` declaring it requires a `time` coord). This works for the config modal
but doesn't help the toolbar — `_format_window_info` has access to the `PlotConfig`
and its params, not to `ModelWidget`.

The deeper issue: the toolbar and the config modal are unrelated UI components that
both need the same data-derived fact. The param model is the wrong place to store
data properties.

### Chosen approach: `supports_windowing` flag on `PlotConfig`

The output template determines whether windowing applies. We derive a boolean from it
and store it on `PlotConfig`, where it's accessible to all consumers:

- The config modal reads it to compute `hidden_fields` for `ModelWidget`
- `_format_window_info` reads it to suppress window display
- `create_extractors_from_params` can use it as a safety check

This keeps the knowledge in one place (the output template check) and makes the
derived fact available everywhere it's needed.

### Future extension: dynamic field visibility

`ModelWidget.hidden_fields` is designed as `frozenset[str]` — a static set computed
once at construction. This is sufficient for the cumulative output case because the
decision is made before the widget is created.

A future use case is dynamic visibility based on sibling field values (e.g., showing
TOF edges vs wavelength edges depending on a mode enum). This would require:

1. The `hidden_fields` parameter accepting a callable
   `Callable[[dict[str, Any]], frozenset[str]]` in addition to `frozenset`
2. The model defining a `hidden_fields` classmethod that encodes field
   interdependencies
3. `ModelWidget` re-evaluating on field changes and showing/hiding cards

The current design does not preclude this. The static `frozenset` is a subset of the
callable interface — externally-provided and model-intrinsic rules would compose
(union of both). The key architectural decision — that `ModelWidget` accepts
`hidden_fields` and skips those fields in rendering while using defaults in
`parameter_values` — works for both static and dynamic cases.

## Design

### `supports_windowing` on `PlotConfig`

`PlotConfig` gains a `supports_windowing: bool` field (default `True`). It's set at
config creation time based on the output template:

```python
supports_windowing = output_has_time_coord(workflow_spec, output_name)
```

Where `output_has_time_coord` is a small pure function:

```python
def output_has_time_coord(workflow_spec: WorkflowSpec, output_name: str) -> bool:
    template = workflow_spec.get_output_template(output_name)
    if template is None:
        return True  # Assume windowing is supported if no template
    return 'time' in template.coords
```

### `hidden_fields` on `ModelWidget`

`ModelWidget` gains `hidden_fields: frozenset[str] = frozenset()`. Fields in the set
are skipped in `_get_parameter_widget_data()`. When `parameter_values` is called,
hidden fields are absent from `_parameter_widgets`, so Pydantic fills in their
defaults from the model class. The model type and serialization are unchanged.

### Threading through the adapter

`ConfigurationAdapter` gains a `hidden_fields` property (default: `frozenset()`).
`PlotConfigurationAdapter` accepts it in its constructor.
`ConfigurationWidget._create_model_widget()` passes it to `ModelWidget`.

### Config modal: determining hidden fields

`SpecBasedConfigurationStep._create_config_panel()` checks `output_has_time_coord()`.
If `False`, passes `hidden_fields=frozenset({'window'})` to
`PlotConfigurationAdapter`. Rate normalization is still meaningful for
cumulative outputs (divides by time span), so only the window field is hidden.

The same flag is stored on the resulting `PlotConfig` for the toolbar.

### Toolbar: suppressing window display

`_format_window_info` gains access to `supports_windowing` (via `PlotConfig`) and
returns `''` when it is `False`.

### Edit mode: switching from "current" to "cumulative" output

When editing, the user can navigate back and change the output. This is safe:

1. `SpecBasedConfigurationStep.on_enter()` detects the output change and calls
   `_create_config_panel()`, which creates a fresh `PlotConfigurationAdapter`.
2. The new adapter gets `hidden_fields` computed from the new output's template.
3. `ModelWidget` skips hidden fields — no `window`/`rate` cards appear.
4. `parameter_values` returns the full model with defaults for hidden fields.
5. Stale `window` settings from `initial_config.params` are harmless — `ModelWidget`
   only reads `initial_values` for fields it renders.

### Grid template configs

Existing YAML grid templates that reference cumulative outputs with explicit
`window` params (if any) will still work: the stored params are valid Pydantic
values, they just won't be editable in the UI. The `supports_windowing=False`
flag suppresses the toolbar label. `create_extractors_from_params` with the default
`WindowMode.latest` produces a `LatestValueExtractor`, which is correct for
cumulative data.

## Changes

### `PlotConfig` (`dashboard/plot_orchestrator.py`)

- Add `supports_windowing: bool = True` field.

### `ModelWidget` (`dashboard/widgets/model_widget.py`)

- Add `hidden_fields: frozenset[str] = frozenset()` parameter to `__init__`.
- In `_get_parameter_widget_data()`, skip fields in `hidden_fields`.

### `ConfigurationAdapter` (`dashboard/configuration_adapter.py`)

- Add `hidden_fields` property with default `frozenset()`.

### `PlotConfigurationAdapter` (`dashboard/plot_configuration_adapter.py`)

- Accept `hidden_fields: frozenset[str] = frozenset()` in constructor.
- Implement `hidden_fields` property.

### `ConfigurationWidget` (`dashboard/widgets/configuration_widget.py`)

- In `_create_model_widget()`, pass `self._config.hidden_fields` to `ModelWidget`.

### `SpecBasedConfigurationStep._create_config_panel()` (`dashboard/widgets/plot_config_modal.py`)

- Call `output_has_time_coord()` and derive `hidden_fields`.
- Pass to `PlotConfigurationAdapter`.
- Store `supports_windowing` on the resulting `PlotConfig`.

### `_format_window_info` (`dashboard/widgets/plot_widgets.py`)

- Accept or check `supports_windowing` and return `''` when `False`.

### New helper function

- `output_has_time_coord(workflow_spec, output_name) -> bool` — pure function,
  in `plotting_controller.py` or a shared utility module.

## Testing

- Unit test `output_has_time_coord`: template with time coord → `True`;
  template without → `False`; `None` template → `True`.
- Unit test `ModelWidget` with `hidden_fields`: verify hidden fields don't produce
  widgets; verify `parameter_values` returns full model with defaults for hidden
  fields.
- Unit test `_format_window_info` with `supports_windowing=False` → empty string.
- Unit test edit-mode scenario: switching output from current to cumulative
  produces config with `supports_windowing=False` and default window params.

## Out of scope

- Defensive error handling in `WindowAggregatingExtractor` or `TemporalBuffer` for
  missing time coords. The UI prevention should be sufficient; adding runtime guards
  could mask configuration bugs.
- Dynamic field visibility based on sibling values (TOF/wavelength edges). The
  `hidden_fields` mechanism is designed to support this later.
