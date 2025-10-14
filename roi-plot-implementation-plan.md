# Implementation Plan: ROI Plot with Interactive Editing

## Implementation Status Summary

**Last updated**: 2025-10-14 (after commit 9ece12b4)

### Completed ✅
1. **ROIDetectorPlotter foundation** (commit 090f6379): Created plotter class with BoxEdit overlay, Layout support, and graceful degradation
2. **Backend ROI infrastructure** (commit da120d72): Added LIVEDATA_ROI topic, StreamKind, routing, and detector_data subscription
3. **Plotter registration** (commits 090f6379, a80e5c6f): Registered roi_detector plotter with data requirements
4. **Test coverage**: 93 passing tests (5 plotter tests + 88 infrastructure tests)

### In Progress 🔄
1. **BoxEdit integration**: Stream created but not yet connected to publishing mechanism
2. **ROI shape display**: Request overlay done, readback overlay not yet implemented

### Not Started ❌
1. **Publishing ROI updates**: Dashboard publishing infrastructure for aux data
2. **Readback shape overlays**: Static shape display from backend `roi_rectangle` stream
3. **BoxEdit change detection**: Subscribe to BoxEdit data parameter and serialize to RectangleROI

### Next Steps
The primary blocker is **section 6: Publishing ROI Updates**. Once dashboard can publish ROI changes to the LIVEDATA_ROI topic, the backend will automatically process them (infrastructure is ready). After that, implement readback overlays (section 3) to close the feedback loop.

---

## Understanding Summary

**Backend (already implemented):**
- `DetectorView` workflow outputs 5 results per job: `cumulative`, `current`, `roi_cumulative`, `roi_current`, `roi_rectangle`
- Each has a predictable `ResultKey`: `{workflow_id, job_id: {job_number, source_name}, output_name}`
- ROI config flows: Dashboard → `WORKFLOW_CONFIG` topic → Backend (via `aux_source_names` in WorkflowConfig)
- ROI data flows: Backend → `DATA` topic → Dashboard (as regular job outputs)
- Auxiliary input flows: **New AUX_DATA topic** → Backend job (via `aux_data` parameter in `JobData`)

**Frontend publishing path:**
- ROI updates are **auxiliary input data** (not config), so they need a **new Kafka topic**
- Dashboard publishes ROI → **New `LIVEDATA_AUX_DATA` topic** (da00 schema with ROI models) → Backend job
- Stream name in aux data: `{job_number}/roi_rectangle` (matches `DetectorROIAuxSources.render()`). Since the shape tool in the plot produces the data for this topic there must be a mechanism to either (a) tie the plotter to the concrete DetectorROIAuxSources or (b) use the JobNumber from the ResultKey. The latter might be simpler.
- Backend job receives aux data via `Job.add(JobData(aux_data={stream_name: roi_dataarray}))`. The mechanism for this is fully in place, we just need to subscribe to the additional topic (detector_data.py is the entry point for this).
- **Message schema**: da00 (DataArray serialization using `scipp_to_da00`).

## Implementation Plan

### **1. Create ROIDetectorPlotter** ✅ DONE (commit 090f6379)
- ✅ New plotter class in `dashboard/plots.py` (ROIDetectorPlotter)
- ✅ Returns `hv.Layout` with detector image and ROI spectrum
- ✅ BoxEdit overlay for interactive rectangle editing (red, 0.3 alpha)
- ✅ Gracefully handles missing ROI spectrum data
- ✅ Registered as 'roi_detector' plotter with validator
- ✅ Comprehensive test coverage (5 tests)
- 🔄 **Partially complete**: Consumes detector data and roi_spectrum, but not yet subscribing to all ROI shape streams
  - Currently: Only displays BoxEdit overlay (user "request" ROI)
  - **TODO**: Subscribe to `roi_rectangle`, `roi_polygon`, `roi_ellipse` readback streams
  - **TODO**: Display static shape overlays for backend "readback" ROI
  - **TODO**: Implement dual overlay system (request vs readback)

### **2. BoxEdit Integration Strategy** 🔄 IN PROGRESS
- ✅ BoxEdit stream attached to the 2D detector image element (commit 090f6379)
- ✅ BoxEdit stream accessible via `roi_plotter.box_stream` attribute
- ✅ Configured for single rectangle editing (see `git diff bdcd3b2c roi-stream` for POC reference)
- ❌ **TODO**: Subscribe to BoxEdit `data` parameter changes to detect user edits
- ❌ **TODO**: On change: Serialize rectangle to `RectangleROI` model → publish to aux data topic
- ❌ **TODO**: Target stream: `{job_number}/roi_rectangle` (extract job_number from `ResultKey.job_id.job_number`)
- ❌ **TODO**: Integrate with publishing infrastructure (see section 6 below)

### **3. ROI Shape Display (Readback + Request)** ❌ NOT STARTED
- **Two separate visual overlays**:
  1. **Request ROI**: BoxEdit overlay (user-editable, shown as dashed lines) ✅ DONE (commit 090f6379)
  2. **Readback ROI**: Static shape overlay from `roi_rectangle` backend output (solid lines) ❌ TODO
- ❌ **TODO**: Subscribe to `roi_rectangle` output stream from backend
- ❌ **TODO**: When received: Parse `RectangleROI` from DataArray → create static shape overlay (not BoxEdit)
- ❌ **TODO**: Implement separate visual styling for request vs readback (dashed vs solid)
- **Why two overlays**: BoxEdit reflects user's immediate edits; readback shows backend's accepted ROI. After a brief period they converge to the same location/shape.
- BoxEdit should **not** update its position when readback arrives (user remains in control of the "request")

### **4. Handling Missing ROI Data** ✅ DONE (commit 090f6379)
- ✅ On first plot creation, gracefully handles missing `roi_spectrum` data
- ✅ Keep it minimal and simple:
  - ✅ Show detector image immediately
  - ✅ Enable BoxEdit tool for user to draw ROI
  - ✅ Show detector-only view if ROI spectrum missing (returns Overlay instead of Layout)
  - ✅ Update automatically when backend starts publishing ROI data (handled by HoloViews reactivity)

### **5. Plotter Registration** ✅ DONE (commits 090f6379, a80e5c6f)
- ✅ Registered in `dashboard/plotting.py` with `plotter_registry`
- ✅ Registered as 'roi_detector' plotter type
- ✅ Data requirements:
  - ✅ min_dims=2, max_dims=2 (exactly 2D)
  - ✅ multiple_datasets=False (single dataset only)
  - ✅ Custom validator placeholder (commit a80e5c6f simplified validator)
- ⚠️ **Note**: Validator currently returns True (does not check namespace). May need enhancement later to validate `detector_data` namespace.

### **6. Publishing ROI Updates** ❌ NOT STARTED
- ❌ **TODO**: Need new infrastructure for auxiliary data publishing (separate from config)
- ❌ **TODO**: Implementation approach: Let the implementer decide the best location (may not be `KafkaTransport` which was designed for config messages)
- ❌ **TODO**: Functionality needed: `publish_aux_data(stream_name, dataarray)` method
- ❌ **TODO**: Serialize `RectangleROI` → `DataArray` → DA00 → publish to `LIVEDATA_ROI` topic
- ❌ **TODO**: Stream name: `{job_number}/roi_rectangle` (extract job_number from `ResultKey.job_id.job_number`)
- ⚠️ **Note**: Backend infrastructure ready (see section on Q1 below) - ROI topic exists and detector_data service subscribes to it

## Open Questions

**Q1: New Kafka topic for auxiliary input data** ✅ **RESOLVED & IMPLEMENTED** (commit da120d72)
- ✅ **Decision**: Need new `LIVEDATA_ROI` topic separate from config and data topics
- ✅ ROI updates are auxiliary **input** to jobs (not output, not config)
- ✅ Use DA00 schema for serialization (same as data topic)
- ✅ **Implemented** (commit da120d72):
  - ✅ Added `LIVEDATA_ROI` to `StreamKind` enum ([message.py:25](src/ess/livedata/core/message.py#L25))
  - ✅ Created ROI topic mapping in config ([streams.py:39-40](src/ess/livedata/config/streams.py#L39-L40))
  - ✅ Added `livedata_roi_topic` property to `StreamMapping` ([stream_mapping.py:48,61-63](src/ess/livedata/kafka/stream_mapping.py#L48))
  - ✅ Implemented `RoutingAdapterBuilder.with_livedata_roi_route()` ([routes.py:93-100](src/ess/livedata/kafka/routes.py#L93-L100))
  - ✅ Backend: detector_data service subscribes to ROI topic ([detector_data.py:24](src/ess/livedata/services/detector_data.py#L24))
  - ✅ Backend: Route aux data messages via existing stream name matching (already in JobManager)
  - ❌ Dashboard: New publishing mechanism for aux data (NOT YET IMPLEMENTED - see section 6)

**Q2: Backend subscription to aux data** ✅ **RESOLVED & IMPLEMENTED** (commit da120d72)
- ✅ **How routing works**: `JobManager.push_data()` receives all messages in `WorkflowData.data`
- ✅ `JobManager._push_data_to_job()` filters by checking if `stream.name` matches:
  - `job.source_names` → goes to `JobData.primary_data`
  - `job.aux_source_names` → goes to `JobData.aux_data`
- ✅ **Implemented**: detector_data service subscribes to ROI topic via `with_livedata_roi_route()` ([detector_data.py:24](src/ess/livedata/services/detector_data.py#L24))
- ✅ Messages now flow: LIVEDATA_ROI topic → RoutingAdapter → JobManager → correct job's `aux_data`

**Q3: Graceful degradation** ✅ **RESOLVED & IMPLEMENTED** (commit 090f6379)
- ✅ **Decision**: Keep it minimal and simple for now
- ✅ **Implemented**: Shows detector-only view when ROI spectrum data doesn't exist yet
- ✅ Returns `Overlay` (detector + BoxEdit) instead of `Layout` when spectrum missing
- ✅ Expect to iterate on this behavior later based on user feedback

**Q4: Hard-coding rectangle for now** ✅ **RESOLVED & PARTIALLY IMPLEMENTED** (commit 090f6379)
- ✅ **Decision**: For MVP, hard-code `hv.streams.BoxEdit` (rectangle only)
- ✅ **Implemented**: BoxEdit stream created and attached to detector image
- ❌ **TODO**: Subscribe to all 3 readback streams (`roi_rectangle`, `roi_polygon`, `roi_ellipse`) for future-proofing
- ✅ Only display rectangle tool in MVP (readback overlay not yet implemented)
- ⏭️ **Future work**: Read aux_sources config to determine which tool to show
