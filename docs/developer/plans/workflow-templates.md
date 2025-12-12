# Workflow Templates: Dynamic Workflow Spec Creation

## Problem Statement

Correlation histogram workflows don't fit the current `JobOrchestrator` model:

1. **Identity mismatch**: For standard workflows (e.g., detector reduction), `(workflow_id, source_name)` fully identifies a running analysis. For correlation histograms, the correlation axis (e.g., "temperature") is also part of the identity - "counts vs temperature" and "counts vs pressure" are independent analyses.

2. **Lifecycle mismatch**: `JobOrchestrator` assumes one active JobSet per workflow_id, with synchronized start/stop. This makes sense for detector panels running the same reduction, but not for independent correlation analyses.

3. **Plot selection mismatch**: When selecting a workflow output to plot, `workflow_id + output_name` doesn't fully specify a correlation result - you also need to know what it's correlated against.

4. **Architecture smell**: `on_configure_workflow` in `reduction.py` uses `WorkflowController` only for `create_workflow_adapter()`, while `WorkflowStatusWidget` uses `JobOrchestrator` directly. This inconsistency exists because correlation histograms are special-cased.

## Key Insight

The correlation axis should be part of the **workflow type definition**, not a runtime parameter. Instead of one "Correlation Histogram" workflow configured at runtime, we should have distinct workflow specs like "Temperature Correlation Histogram" and "Pressure Correlation Histogram".

## Design: WorkflowTemplate

A `WorkflowTemplate` is a factory that creates `WorkflowSpec` instances from user-provided configuration.

```python
class WorkflowTemplate(Protocol):
    """Factory for creating WorkflowSpec instances dynamically."""

    @property
    def name(self) -> str:
        """Template identifier, e.g., 'correlation_histogram_1d'"""

    @property
    def title(self) -> str:
        """Human-readable title, e.g., '1D Correlation Histogram'"""

    def get_configuration_model(self) -> type[pydantic.BaseModel]:
        """Pydantic model for template configuration (e.g., correlation axis selection)."""

    def create_workflow_spec(self, config: pydantic.BaseModel) -> WorkflowSpec:
        """Create a WorkflowSpec from the template configuration."""

    def make_instance_id(self, config: pydantic.BaseModel) -> WorkflowId:
        """Generate unique workflow ID for this instance."""

    def make_instance_title(self, config: pydantic.BaseModel) -> str:
        """Generate human-readable title, e.g., 'Temperature Correlation Histogram'."""
```

### CorrelationHistogramTemplate

```python
class CorrelationHistogram1dTemplate(WorkflowTemplate):
    name = "correlation_histogram_1d"
    title = "1D Correlation Histogram"

    def get_configuration_model(self) -> type[BaseModel]:
        # Returns model with field for selecting correlation axis timeseries
        ...

    def create_workflow_spec(self, config) -> WorkflowSpec:
        # Creates spec with the correlation axis baked into the identity
        # The axis becomes part of workflow_id, not aux_sources
        ...

    def make_instance_title(self, config) -> str:
        # e.g., "Temperature Correlation Histogram"
        return f"{config.axis_name} Correlation Histogram"
```

## Implementation Plan

### Phase 1: Core Infrastructure (branch off `main`)

This phase introduces the template concept and makes correlation histograms work through the standard JobOrchestrator flow, without new UI. `WorkflowController` remains untouched - it continues to serve the legacy `ReductionWidget`.

#### 1.1 WorkflowTemplate Protocol

Location: `src/ess/livedata/config/workflow_template.py`

- Define `WorkflowTemplate` protocol
- Define `TemplateInstance` model for persistence:
  ```python
  class TemplateInstance(BaseModel):
      template_name: str
      config: dict  # Serialized template configuration
  ```

#### 1.2 CorrelationHistogramTemplate Implementation

Location: `src/ess/livedata/dashboard/correlation_histogram.py` (refactor existing)

- Implement `CorrelationHistogram1dTemplate` and `CorrelationHistogram2dTemplate`

**Key change in configuration flow:**

Old flow (aux-source selection at job configuration time):
1. User opens "Correlation Histogram" configuration modal
2. Modal shows aux-source selection (populated from DataService timeseries)
3. User selects correlation axis (e.g., "temperature")
4. User configures bin edges and selects data sources
5. Job starts with axis as a runtime parameter

New flow (aux-source selection at template instantiation time):
1. User creates new workflow from template
2. Template queries DataService for available timeseries
3. User selects correlation axis → template creates "Temperature Correlation Histogram" spec
4. User later opens this workflow's configuration modal
5. Modal shows only bin edges and data source selection (axis already fixed)
6. Job starts - the axis is part of the workflow identity, not a parameter

**Template configuration model:**
```python
class CorrelationHistogram1dTemplateConfig(BaseModel):
    axis_source: ResultKey  # The timeseries to correlate against
    # For 2D: axis_x_source, axis_y_source
```

The template's `get_configuration_model()` dynamically creates this with axis options populated from `DataService.get_timeseries()` (same source as current mechanism).

**Created WorkflowSpec:**
- `workflow_id` includes axis identity (e.g., `correlation/temperature_histogram_1d/v1`)
- `source_names`: Available data sources that can be correlated (other timeseries)
- `params`: Bin edge configuration (unchanged from current)
- `aux_sources`: None (axis is now baked into the workflow identity)
- Workflow execution still runs in frontend via `CorrelationHistogramController`

#### 1.3 Dynamic Registry in JobOrchestrator

Extend `JobOrchestrator` to support dynamically added workflows:

- Add `WorkflowRegistryManager` helper class (internal to JobOrchestrator):
  ```python
  class WorkflowRegistryManager:
      def __init__(self, static_registry, config_store, templates):
          ...

      def register_from_template(self, template_name: str, config: dict) -> WorkflowId:
          """Create and register a workflow spec from a template."""

      def unregister(self, workflow_id: WorkflowId) -> bool:
          """Remove a template-created workflow (no-op for static workflows)."""

      def get_registry(self) -> Mapping[WorkflowId, WorkflowSpec]:
          """Combined view of static + dynamic workflows."""

      def is_template_instance(self, workflow_id: WorkflowId) -> bool:
          """Check if workflow was created from a template."""
  ```

- Persistence: Store template instances in config store under `_template_instances` key
- On init: Recreate dynamic specs from persisted template instances

#### 1.4 Hard-Coded Template Instances for Testing

For development/testing without UI, hard-code template instances in `dummy` instrument setup or in `DashboardServices`. This validates the model before building the dynamic UI.

### Phase 2: Dynamic UI (later, needs `workflow-control-widget` branch)

This phase adds UI for creating/deleting template instances at runtime.

#### 2.1 Template Selection UI

In `WorkflowStatusListWidget`:

- Add "Add workflow" button (or "+" icon)
- On click: Show template selection (list available templates)
- On template select: Show template configuration modal

#### 2.2 Template Configuration Modal

- Use template's `get_configuration_model()` to generate form
- Consider reusing `ConfigurationAdapter` pattern (investigate fit during implementation)
- On submit: Call `JobOrchestrator.register_from_template()` → widget rebuilds

#### 2.3 Delete Button for Template Instances

In `WorkflowStatusWidget`:

- Show delete button only for template instances (`is_template_instance()`)
- On click: Confirm → `JobOrchestrator.unregister()` → widget rebuilds
- Handle cleanup: Stop running jobs, remove from plot configs if referenced

## Migration Path

1. **Phase 1 complete**: Correlation histograms work through JobOrchestrator with hard-coded instances for testing. `WorkflowController` and `ReductionWidget` remain unchanged (legacy path).

2. **Phase 2 complete**: Users can create/delete correlation workflows dynamically via `WorkflowStatusListWidget`. The new UI path is fully functional.

3. **Future**: Once `WorkflowStatusListWidget` is mature, `ReductionWidget` and its `WorkflowController` dependency can be removed.

## Open Questions

1. **Template discovery**: How do templates register themselves? Global registry? Passed to JobOrchestrator at init?

2. **Correlation execution**: Currently correlation histograms run in-process via `CorrelationHistogramController`. Does this need to change, or does `WorkflowConfigurationAdapter.start_action()` just delegate to it?

3. **Source name mapping**: Correlation "sources" are ResultKeys (workflow outputs), not raw Kafka streams. How does this map to `WorkflowSpec.source_names`?

## Files Affected

### Phase 1
- `src/ess/livedata/config/workflow_template.py` (new)
- `src/ess/livedata/dashboard/job_orchestrator.py` (extend with WorkflowRegistryManager)
- `src/ess/livedata/dashboard/correlation_histogram.py` (add template implementations)
- `src/ess/livedata/dashboard/dashboard_services.py` (hard-coded test instances)

### Phase 2
- `src/ess/livedata/dashboard/widgets/workflow_status_widget.py` (add workflow button, delete button)
- New modal widget for template configuration (or reuse ConfigurationModal)
