# Issue #433: Next Steps

## Completed Work

The two-phase registration pattern has been successfully implemented with the following components:

1. ✅ Core infrastructure (`SpecHandle`, `register_spec()`, `attach_factory()`)
2. ✅ Detector handler split (`detector_view_specs.py` lightweight, `detector_data_handler.py` heavy)
3. ✅ Dummy instrument converted to submodule structure
4. ✅ Backend services updated to explicitly import factories
5. ✅ Comprehensive test coverage
6. ✅ Stream mapping separated from specs (moved to `streams.py` module for cleaner separation of concerns)

## Remaining Work

### 1. Split Monitor and Timeseries Handlers (HIGH PRIORITY)

**Problem**: `monitor_data_handler.py` and `timeseries_handler.py` currently pull in heavy dependencies through `accumulators.py`:
- `monitor_data_handler` → `accumulators` → `ess.reduce.live.roi`
- This prevents truly lightweight spec loading

**Solution**:
- Create `monitor_view_specs.py` with lightweight spec registration helpers
- Refactor `monitor_data_handler.py` to separate spec registration from factory implementation
- Create `timeseries_view_specs.py` for timeseries workflows
- Update `accumulators.py` to split lightweight and heavy components

**Impact**: This will enable full lightweight spec loading for all instruments, including dummy.

### 2. Convert Remaining Instruments to Submodule Structure

Each instrument needs conversion following the dummy pattern.

**Dummy instrument structure** (reference implementation):
- `__init__.py` - imports only specs module
- `specs.py` - lightweight spec registration (no heavy imports, no stream_mapping)
- `factories.py` - heavy factory implementations with ess.reduce imports
- `streams.py` - stream mapping configuration (Kafka infrastructure, not needed by frontend)

The key improvement is that `stream_mapping` has been moved to a separate `streams.py` module, keeping the `specs.py` module truly lightweight and frontend-friendly. The `get_stream_mapping()` function in `config/streams.py` automatically handles both the old pattern (single file with stream_mapping) and new pattern (separate streams.py submodule).

#### LOKI
- [ ] Create `instruments/loki/` submodule
- [ ] Create `loki/__init__.py` (imports only specs)
- [ ] Create `loki/specs.py` with:
  - `SansWorkflowParams` Pydantic model
  - `LokiAuxSources` model
  - Spec registrations using `register_detector_view_specs()`
  - Spec registrations for SANS workflows (return handles)
  - NO stream_mapping (moved to streams.py)
- [ ] Create `loki/streams.py` with:
  - `stream_mapping` dictionary
  - Detector configuration helpers
  - Kafka-related infrastructure (not needed by frontend)
- [ ] Create `loki/factories.py` with:
  - Heavy imports (`ess.loki.live`, `ess.reduce`)
  - Detector projection factory attachments
  - SANS workflow factory implementations
  - Attach using handles from specs
- [ ] Remove old `loki.py`
- [ ] Test both specs-only and full loading

#### DREAM
- [ ] Create `instruments/dream/` submodule
- [ ] Split into `specs.py` (lightweight), `streams.py` (Kafka config), and `factories.py` (heavy)
- [ ] Handle DREAM-specific workflows and detector views
- [ ] Remove old `dream.py`

#### BIFROST
- [ ] Create `instruments/bifrost/` submodule
- [ ] Split into `specs.py`, `streams.py`, and `factories.py`
- [ ] Handle spectroscopy workflows
- [ ] Remove old `bifrost.py`

#### ODIN
- [ ] Create `instruments/odin/` submodule
- [ ] Split into `specs.py`, `streams.py`, and `factories.py`
- [ ] Handle imaging workflows
- [ ] Remove old `odin.py`

#### NMX
- [ ] Create `instruments/nmx/` submodule
- [ ] Split into `specs.py`, `streams.py`, and `factories.py`
- [ ] Handle crystallography workflows
- [ ] Remove old `nmx.py`

#### TBL (Test Beamline)
- [ ] Create `instruments/tbl/` submodule
- [ ] Split into `specs.py`, `streams.py`, and `factories.py`
- [ ] Remove old `tbl.py`

### 3. Update DetectorProjection to Use New Pattern

**Current**: `DetectorProjection.__init__()` auto-registers workflows (line 125 in old pattern)

**New pattern**:
```python
# In instrument/specs.py
from ess.livedata.handlers.detector_view_specs import register_detector_view_specs

handles = register_detector_view_specs(
    instrument=instrument,
    projections=['xy_plane'],
    source_names=detector_names
)

# In instrument/factories.py
from ess.livedata.handlers.detector_data_handler import DetectorProjection

projection = DetectorProjection(
    instrument=instrument,
    projection='xy_plane',
    resolution=resolution,
    # ... other config
)

# Explicitly attach factories using handles
projection.attach_to_handles(
    view_handle=handles['xy_plane']['view'],
    roi_handle=handles['xy_plane']['roi']
)
```

### 4. Frontend Verification

Once monitor/timeseries handlers are split:

- [ ] Verify dashboard can import specs without heavy dependencies
- [ ] Test workflow configuration UI generation from specs
- [ ] Verify no `ess.reduce`, `ess.loki`, etc. imports in frontend
- [ ] Add integration test for frontend spec loading

### 5. Cleanup Old Pattern

After all instruments are converted:

- [ ] Remove `Instrument.register_workflow()`
- [ ] Remove auto-registration from `DetectorProcessorFactory.__init__`
- [ ] Update all docstrings to reference new pattern
- [ ] Remove old instrument `.py` files
- [ ] Remove support for old pattern from `get_stream_mapping`.

### 6. Documentation Updates

- [ ] Update developer docs with two-phase registration pattern
- [ ] Document migration guide for converting instruments
- [ ] Update architecture diagrams
- [ ] Add examples of spec registration and factory attachment
- [ ] Document the lightweight/heavy split pattern

## Testing Strategy

For each instrument conversion:
1. Write test that loads specs without heavy imports
2. Write test that loads specs + factories
3. Verify existing instrument tests still pass
4. Test backend service can create workflows
5. Test frontend can generate UI from specs

## Success Criteria

- ✅ Frontend can load all instrument specs without `ess.reduce` or instrument-specific packages
- ✅ Backend services load both specs and factories correctly
- ✅ All existing tests pass
- ✅ New integration tests verify lightweight/heavy split
- ✅ Documentation clearly explains the pattern
