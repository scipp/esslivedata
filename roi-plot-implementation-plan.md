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
- Consumes **2 data outputs and 3 roi shape outputs from same job**: `current` (or `cumulative`), `roi_current` (or `roi_cumulative`), `roi_rectangle`, `roi_polygon`, `roi_ellipse`.
- Data requirements:
  - Single dataset only (`multiple_datasets: False`)
  - Must be from `detector_data` namespace
  - Must have 2D detector image data
- Returns `hv.Layout` with:
  - Left: 2D detector image with (1) BoxEdit overlay (2) shapes from `roi_*` streams.
  - Right: 1D TOA spectrum

### **2. BoxEdit Integration Strategy**
- BoxEdit stream attached to the 2D detector image element
- Subscribe to BoxEdit `data` parameter changes (for now we only implement the rectangle case, i.e., BoxEdit!)
- On change: Serialize rectangle to `RectangleROI` model → publish to config service
- Target stream: `{job_number}/roi_rectangle` (extract job_number from ResultKey)

### **3. ROI Shape Display**
- Subscribe to `roi_rectangle` output stream from backend
- When received: Parse `RectangleROI` from DataArray → update BoxEdit overlay (or equivalent, boxes drawn from stream? We want the "readback" ROI, i.e., the ROI that is actually corresponding to the current data *not user editable*, the "request" ROI is user-editable).
- Show "readback" as solid lines, "request" as dashed.
- This shows the backend's "accepted" ROI (handles lag between user edit and backend update)

### **4. Handling Missing ROI Data**
- On first plot creation, `roi_cumulative`/`roi_current`/`roi_rectangle` may not exist in DataService
- Plot should:
  - Show detector image immediately
  - Show empty spectrum plot (or placeholder text)
  - Enable BoxEdit tool for user to draw ROI
  - Update automatically when backend starts publishing ROI data

### **5. Plotter Registration**
- Register in `dashboard/plotting.py` with `plotter_registry`
- Data requirements validator:
  - Check `workflow_id.namespace == 'detector_data'`
  - Check data is 2D
  - Single dataset only

### **6. Publishing ROI Updates**
- Need new infrastructure for auxiliary data publishing (separate from config)
- Create method: `publish_aux_data(stream_name, dataarray)` in dashboard transport layer
- Serialize `RectangleROI` → `DataArray` → DA00 → publish to `LIVEDATA_AUX_DATA` topic
- Stream name: `{job_number}/roi_rectangle` (extract job_number from ResultKey)

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
- ❓ **Need to check**: How to add aux data topic to `DataServiceBuilder` / service configuration

**Q3: Graceful degradation**
- When ROI outputs don't exist yet, should we:
  - Show empty/placeholder 1D plot?
  - Show error message?
  - Hide 1D plot until data arrives?

**Q4: Hard-coding rectangle for now**
- For MVP, we hard-code `hv.streams.BoxEdit` and ignore polygon/ellipse options?
- Future work: Read aux_sources config to determine which tool to show?
