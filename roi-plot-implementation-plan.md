# Implementation Plan: ROI Plot with Interactive Editing

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

### **1. Create ROIDetectorPlotter**
- New plotter class in `dashboard/plots.py`
- Consumes **2 data outputs and 3 ROI shape outputs from same job**:
  - **Data**: `current` (or `cumulative`), `roi_current` (or `roi_cumulative`)
  - **Readback shapes**: `roi_rectangle`, `roi_polygon`, `roi_ellipse`
- Subscribe to all 3 shape streams (future-proof), but only display/enable rectangle tool for MVP
- Data requirements:
  - Single dataset only (`multiple_datasets: False`)
  - Must be from `detector_data` namespace
  - Must have exactly 2D detector image data
- Returns `hv.Layout` with:
  - Left: 2D detector image with (1) BoxEdit overlay for "request" ROI (2) static shape overlays for "readback" ROI from `roi_*` streams
  - Right: 1D TOA spectrum from `roi_current`/`roi_cumulative`

### **2. BoxEdit Integration Strategy**
- BoxEdit stream attached to the 2D detector image element (user-editable "request" ROI)
- Subscribe to BoxEdit `data` parameter changes (for now we only implement the rectangle case, i.e., BoxEdit!)
- See `git diff bdcd3b2c roi-stream` for a POC to get started!
- On change: Serialize rectangle to `RectangleROI` model → publish to aux data topic
- Target stream: `{job_number}/roi_rectangle` (extract job_number from `ResultKey.job_id.job_number`)

### **3. ROI Shape Display (Readback + Request)**
- **Two separate visual overlays**:
  1. **Request ROI**: BoxEdit overlay (user-editable, shown as dashed lines)
  2. **Readback ROI**: Static shape overlay from `roi_rectangle` backend output (solid lines)
- Subscribe to `roi_rectangle` output stream from backend
- When received: Parse `RectangleROI` from DataArray → create static shape overlay (not BoxEdit)
- **Why two overlays**: BoxEdit reflects user's immediate edits; readback shows backend's accepted ROI. After a brief period they converge to the same location/shape.
- BoxEdit should **not** update its position when readback arrives (user remains in control of the "request")

### **4. Handling Missing ROI Data**
- On first plot creation, `roi_cumulative`/`roi_current`/`roi_rectangle` may not exist in DataService
- Keep it minimal and simple:
  - Show detector image immediately
  - Show empty/minimal spectrum plot (expect to iterate on this later)
  - Enable BoxEdit tool for user to draw ROI
  - Update automatically when backend starts publishing ROI data

### **5. Plotter Registration**
- Register in `dashboard/plotting.py` with `plotter_registry`
- Data requirements validator:
  - Check `workflow_id.namespace == 'detector_data'`
  - Check data is exactly 2D
  - Single dataset only

### **6. Publishing ROI Updates**
- Need new infrastructure for auxiliary data publishing (separate from config)
- Implementation approach: Let the implementer decide the best location (may not be `KafkaTransport` which was designed for config messages)
- Functionality needed: `publish_aux_data(stream_name, dataarray)` method
- Serialize `RectangleROI` → `DataArray` → DA00 → publish to `LIVEDATA_AUX_DATA` topic
- Stream name: `{job_number}/roi_rectangle` (extract job_number from `ResultKey.job_id.job_number`)

## Open Questions

**Q1: New Kafka topic for auxiliary input data (RESOLVED)**
- ✅ **Decision**: Need new `LIVEDATA_AUX_DATA` topic separate from config and data topics
- ✅ ROI updates are auxiliary **input** to jobs (not output, not config)
- ✅ Use DA00 schema for serialization (same as data topic)
- ❓ **Need to implement**:
  - Add `LIVEDATA_AUX_DATA` to `StreamKind` enum
  - Create aux data topic in config (default.yaml)
  - Backend: Subscribe orchestrating processor to aux data topic
  - Backend: Route aux data messages to correct job via stream name matching
  - Dashboard: New publishing mechanism for aux data

**Q2: Backend subscription to aux data (RESOLVED)**
- ✅ **How routing works**: `JobManager.push_data()` receives all messages in `WorkflowData.data`
- ✅ `JobManager._push_data_to_job()` filters by checking if `stream.name` matches:
  - `job.source_names` → goes to `JobData.primary_data`
  - `job.aux_source_names` → goes to `JobData.aux_data`
- ✅ **What we need**: Service must subscribe to aux data topic so messages flow into `WorkflowData`
- ✅ **How to implement**: Add aux data topic subscription in `reduction.py` entry point (service configuration)

**Q3: Graceful degradation (RESOLVED)**
- ✅ **Decision**: Keep it minimal and simple for now
- ✅ Show empty/minimal spectrum plot when ROI data doesn't exist yet
- ✅ Expect to iterate on this behavior later based on user feedback

**Q4: Hard-coding rectangle for now (RESOLVED)**
- ✅ **Decision**: For MVP, hard-code `hv.streams.BoxEdit` (rectangle only)
- ✅ Subscribe to all 3 readback streams (`roi_rectangle`, `roi_polygon`, `roi_ellipse`) for future-proofing
- ✅ Only display rectangle tool and readback overlay in MVP
- ⏭️ **Future work**: Read aux_sources config to determine which tool to show
