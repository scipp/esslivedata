# ConfigurationState Refactor Plan

## Goal

Invert the data model so that `ConfigurationState` represents a single source's configuration (what `JobConfigState` is today), and `JobOrchestrator` manages the collection of these states.

## Current Design (problematic)

```
ConfigurationState
  └── jobs: dict[str, JobConfigState]  # Multi-source container
        └── JobConfigState(params, aux_source_names)  # Single source

ConfigurationAdapter
  └── receives ConfigurationState (multi-source)
  └── uses set_selected_sources() to pick reference
  └── _get_reference_job_config() navigates to single source
```

**Problem**: The adapter works with one source's config, but receives a bundle of all sources. The scoping mechanism exists solely to bridge this mismatch.

## New Design

```
ConfigurationState  # Was JobConfigState - single source config
  └── params: dict
  └── aux_source_names: dict

ConfigurationAdapter
  └── receives ConfigurationState directly (single source)
  └── receives initial_source_names as constructor param
  └── no scoping needed

JobOrchestrator
  └── staged_jobs: dict[SourceName, JobConfig]  # Manages collection
  └── create_workflow_adapter() picks reference and passes directly
```

## Changes Required

### 1. configuration_adapter.py

- **Rename** `JobConfigState` → `ConfigurationState`
- **Delete** old `ConfigurationState` class
- **Update** `ConfigurationAdapter.__init__`:
  ```python
  def __init__(
      self,
      config_state: ConfigurationState | None = None,
      initial_source_names: list[str] | None = None,
  ) -> None:
      self._config_state = config_state
      self._initial_source_names = initial_source_names
  ```
- **Remove** `set_selected_sources()` method
- **Remove** `_get_reference_job_config()` method
- **Simplify** `initial_source_names` property:
  ```python
  @property
  def initial_source_names(self) -> list[str]:
      if self._initial_source_names is not None:
          available = set(self.source_names)
          filtered = [s for s in self._initial_source_names if s in available]
          return filtered if filtered else self.source_names
      return self.source_names
  ```
- **Simplify** `initial_parameter_values` property:
  ```python
  @property
  def initial_parameter_values(self) -> dict[str, Any]:
      if self._config_state is None:
          return {}
      # Validation logic stays the same, but operates on self._config_state directly
      ...
      return self._config_state.params
  ```
- **Simplify** `initial_aux_source_names` property similarly

### 2. workflow_configuration_adapter.py

- Update import: `ConfigurationState` (was `JobConfigState` effectively)
- Update `__init__` signature to pass through `initial_source_names`:
  ```python
  def __init__(
      self,
      spec: WorkflowSpec,
      config_state: ConfigurationState | None,
      initial_source_names: list[str] | None,
      start_callback: ...,
  ) -> None:
      super().__init__(
          config_state=config_state,
          initial_source_names=initial_source_names,
      )
  ```

### 3. plot_configuration_adapter.py

- Update import
- Update `__init__` to accept `initial_source_names`:
  ```python
  def __init__(
      self,
      plot_spec: PlotterSpec,
      source_names: list[str],
      success_callback,
      config_state: ConfigurationState | None = None,
      initial_source_names: list[str] | None = None,
  ):
      super().__init__(
          config_state=config_state,
          initial_source_names=initial_source_names,
      )
  ```

### 4. job_orchestrator.py

- Update import to use new `ConfigurationState`
- **Update** `create_workflow_adapter`:
  ```python
  def create_workflow_adapter(
      self, workflow_id: WorkflowId, selected_sources: list[str] | None = None
  ) -> WorkflowConfigurationAdapter:
      spec = self._workflow_registry[workflow_id]

      # Pick reference config from first selected source (or first staged)
      config_state = self._get_reference_config(workflow_id, selected_sources)

      return WorkflowConfigurationAdapter(
          spec,
          config_state,
          initial_source_names=selected_sources,
          start_callback=...,
      )
  ```
- **Replace** `_get_config_state` with `_get_reference_config`:
  ```python
  def _get_reference_config(
      self, workflow_id: WorkflowId, selected_sources: list[str] | None = None
  ) -> ConfigurationState | None:
      staged_jobs = self.get_staged_config(workflow_id)
      if not staged_jobs:
          return None

      # Pick reference: first selected source, or first staged
      if selected_sources:
          for source in selected_sources:
              if source in staged_jobs:
                  job = staged_jobs[source]
                  return ConfigurationState(
                      params=job.params,
                      aux_source_names=job.aux_source_names,
                  )

      # Fallback to first staged
      first_job = next(iter(staged_jobs.values()))
      return ConfigurationState(
          params=first_job.params,
          aux_source_names=first_job.aux_source_names,
      )
  ```
- **Persistence format** stays as dict-of-dicts in the config store (JSON), but `ConfigurationState` is only used for single-source interaction with adapters

### 5. workflow_controller.py

- Update import
- Update `get_workflow_config` to return `ConfigurationState | None` (single source, for adapter use)
- Update `create_workflow_adapter` similarly

### 6. widgets/workflow_status_widget.py

- Update `_on_gear_click`:
  ```python
  def _on_gear_click(self, source_names: list[str]) -> None:
      adapter = self._orchestrator.create_workflow_adapter(
          self._workflow_id,
          selected_sources=source_names,  # Pass directly to factory
      )
      # Remove: adapter.set_selected_sources(source_names)
  ```

### 7. widgets/plot_config_modal.py

- **Fix broken code** at lines 682-689 that uses old schema
- Update to use new `ConfigurationState` properly

### 8. Tests

- **configuration_adapter_test.py**:
  - Remove `TestSetSelectedSources` class (scoping tests no longer applicable)
  - Update `TestConfigurationStateSchema` for new single-source schema
  - Add tests for `initial_source_names` constructor parameter

- **job_orchestrator_test.py**:
  - Update tests that use `set_selected_sources`
  - Update to pass `selected_sources` to `create_workflow_adapter`

- **workflow_controller_test.py**:
  - Update similarly

- **config_persistence_test.py**:
  - Update assertions for new schema

## Migration Notes

- **Config files will break** - the stored format changes from `ConfigurationState` with `jobs` dict to just storing the dict directly in the orchestrator's persistence layer
- The orchestrator's `_load_configs_from_store` and `_persist_state_to_store` already work with raw dicts, so persistence format can stay the same internally
- Only the `ConfigurationState` Pydantic model changes semantics

## Benefits

1. Data structure matches usage - no bridging layer
2. Simpler `ConfigurationAdapter` - no scoping mechanism
3. Clearer ownership - orchestrator owns the collection
4. Fewer indirections when debugging
