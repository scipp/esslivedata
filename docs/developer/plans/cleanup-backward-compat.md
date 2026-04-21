# Backward-compat & defensive-code cleanup

Living checklist of cleanup candidates identified 2026-04-20. Each item is
self-contained — verify line numbers before acting (file may have drifted).

When addressing an item: confirm the suspicion, decide between *remove*,
*tighten*, or *keep with justification*, then check it off (or strike through
with a one-line note on why it stays).

Conventions:
- `[ ]` open · `[x]` done · `[~]` decided to keep (add reason)
- One PR per item unless trivially related.

---

## A. Backward-compat shims (likely safe to remove)

- [x] **`dashboard/plot_orchestrator.py:1119-1209`** — three parallel cell
  config formats (legacy `'config'`, new `'layers'`, plus a fallback that
  *creates invalid configs*). Dropped all legacy branches; migrated the 11
  grid template YAMLs (dream/bifrost/dummy, 46 cells) to the new
  `layers: [{data_sources: {primary: {...}}, plot_name, params}]` shape.
- [x] **`dashboard/data_subscriber.py:24-26`** — single-role flat-dict output
  preserved "for backward compatibility". Collapsed to a single role-grouped
  output shape; `Plotter.compute`, `SlicerPlotter.compute`, and
  `BaseROIRequestPlotter.compute` now extract `data[PRIMARY]` themselves.
- [ ] **`kafka/x5f2_compat.py:514`** —
  `message_type = message.get("message_type", "job")`. Producer is in-repo;
  enforce the field instead of defaulting.
- [ ] **`handlers/monitor_workflow_specs.py:144`**,
  **`handlers/detector_view_specs.py:145`** — field names frozen "for
  compatibility with existing workflow templates and serialized configs".
  Decide: introduce schema versioning, or break and migrate stored configs.

## B. Defensive fallbacks for conditions that don't happen

- [ ] **`kafka/sink.py:180`** — `if hasattr(self, '_producer')` in `close()`.
  `_producer` is always set in `__init__`. Replace with explicit `is not None`
  check or remove.
- [ ] **`core/job_manager_adapter.py:67`** — `source_name: str | None`, then
  immediately raises if `None`. Make the parameter required.
- [~] **`dashboard/reduction.py:60`** —
  `if hasattr(self, 'state') and hasattr(self.state, 'toolbar')` on a
  HoloViews `LayoutPlot` (not a Panel template). `state` is
  `handles["plot"]`, which after `initialize_plot` may be a Bokeh `Tabs`
  or wrapping `Column` with no `toolbar` attribute. Keep the guard.
- [ ] **`dashboard/widgets/plot_grid_tabs.py:89, 995`** —
  `if hasattr(params, 'plot_aspect')` then defaults to `'stretch_both'` either
  way. Either branch returns the same value; remove the dead check or make
  `plot_aspect` required.
- [ ] **`dashboard/plots.py:708`**,
  **`dashboard/plotting_controller.py:217`** — `getattr(params, 'rate', None)`
  / `getattr(params, 'window', None)` defensive against polymorphic param
  types. Decide: tighten typing (Protocol / union), or accept the duck-typing.
- [ ] **`kafka/x5f2_compat.py:281`** — `software_version: str | None = None`
  always replaced by `status.version`. Drop the parameter or make required.
- [ ] **`core/service.py:173`** — `if hasattr(self._processor, 'finalize')`.
  Either define a Protocol with optional `finalize`, or require it.
- [ ] **`core/orchestrating_processor.py:94`** —
  `if hasattr(accumulator, 'release_buffers')`. Same as above.
- [ ] **`kafka/source.py:372`** —
  `if not hasattr(self._consumer, 'assignment')`. Document the version
  compatibility, or rely on the type and let it crash if violated.

## C. Silent exception swallowing (hides real bugs)

- [ ] **`dashboard/plot_orchestrator.py:1078`** — params validation failure
  logged then replaced with empty defaults. Surface the error in the UI.
- [ ] **`dashboard/configuration_adapter.py:175-201`** — incompatible stored
  params silently replaced with `{}`. User config lost without warning.
  Surface in UI or refuse to load.
- [ ] **`dashboard/config_store.py:200-228`** — both load and save errors
  swallowed; "starting with empty config store" loses entire config silently.
  At minimum: refuse to overwrite a file we couldn't read.
- [ ] **`config/instrument.py:460-466`** — `except (ValueError, KeyError):
  pass` on `_load_detector_from_nexus`. Detectors silently skipped. Log and
  count, or narrow the exception.
- [ ] **`kafka/scipp_da00_compat.py:54-62`** — silently drops EFU coords with
  mismatched dims (issue #679). Add a one-shot warning per stream.
- [ ] **`dashboard/roi_request_plots.py:717, 873`** — geometry parse failures
  return `{}`, dropping user input. Surface the error.
- [ ] **`handlers/config_handler.py:107, 128`** — bare `except Exception` on
  every config message; bad configs vanish without acknowledgement. Either
  return an error ack or narrow exceptions.
- [ ] **`dashboard/plots.py:448`** — broad `except Exception` rendered as
  `hv.Text(...)` with the exception message. Log full traceback alongside.
- [ ] **`dashboard/widgets/configuration_widget.py:428`** — bare
  `except Exception: pass` on modal cleanup. Log at debug at least.
- [ ] **`dashboard/job_orchestrator.py:925, 948, 1012`** — broad excepts in
  workflow start/stop callbacks may leave subscription state inconsistent.
  Audit whether failure paths corrupt state.

## D. Acknowledged hacks / "for now" with no plan

- [ ] **`dashboard/reduction.py:153`** —
  `workflow_registry=...workflow_controller._workflow_registry` with
  "Temporary hack" comment. Expose via the controller or pass explicitly.
- [ ] **`core/message_batcher.py:74`** — `batch_length_s` accepted but
  ignored. Remove the parameter or implement it.
- [ ] **`config/keys.py:28`** — `WORKFLOW_CONFIG service_name=None`
  broadcasts to all services "for now". Decide on the routing model.
- [ ] **`config/instruments/dream/factories.py:162`** — fake proton charge in
  production "until monitor normalization is fixed". Track and remove.
- [ ] **`dashboard/widgets/plot_grid_tabs.py:906-920`** —
  `linked_axes=False` workaround for HoloViews/Bokeh/Panel. Recheck against
  current upstream versions.
- [ ] **`dashboard/frame_aspect.py:1-27`** — JS workaround for upstream
  responsive-aspect bug. Recheck against current upstream versions.
- [ ] **`kafka/message_adapter.py:153, 332`** — fallback timestamp from Kafka
  envelope when `ev44.reference_time` is empty, "useful in particular for
  testing". Move to test adapters or document the production case.

## E. Optional-with-default parameters that are never optional

Group: each masks a design question — should the dependency be required, or
is the fallback the actual feature?

- [x] **`dashboard/orchestrator.py:120`** —
  `_active_job_registry is None: return True  # Permissive mode`.
  Made required; tests use `PermissiveJobRegistry` fake.
- [x] **`dashboard/orchestrator.py:126`** —
  `_job_orchestrator is None: return`. Made required; tests pass
  `FakeJobOrchestrator`.
- [ ] **`dashboard/stream_manager.py:62`** — `extractors is None` →
  build defaults.
- [x] **`dashboard/data_service.py:71`** — `buffer_manager is None` →
  `TemporalBufferManager()`. No caller ever passed a value; parameter
  dropped, manager constructed internally.
- [x] **`dashboard/session_updater.py:170`** — `_notification_queue is None`
  → `[]`. Made required; tests pass a real `NotificationQueue`.
- [~] **`dashboard/plotting_controller.py:89`** — `template is None` →
  return all plotters with a "trust me" boolean. Keep: mis-categorized —
  `template` is a runtime return value from
  `workflow_spec.get_output_template()`, not an optional-arg default.
