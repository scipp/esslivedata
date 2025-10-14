# Implementation Plan: ROI Plot with Interactive Editing

## Understanding Summary

**Backend (already implemented):**
- `DetectorView` workflow outputs 5 results per job: `cumulative`, `current`, `roi_cumulative`, `roi_current`, `roi_rectangle`
- Each has a predictable `ResultKey`: `{workflow_id, job_id: {job_number, source_name}, output_name}`
- ROI config flows: Dashboard → `WORKFLOW_CONFIG` topic → Backend (via `aux_source_names` in WorkflowConfig)
- ROI data flows: Backend → `DATA` topic → Dashboard (as regular job outputs)

**Frontend publishing path:**
- ROI updates should go through `ConfigService` → `KafkaTransport` → `WORKFLOW_CONFIG` topic
- Need to publish to a stream matching the job's aux source name: `{job_number}/roi_rectangle`

## Implementation Plan

### **1. Create ROIDetectorPlotter**
- New plotter class in `dashboard/plots.py`
- Consumes **3 outputs from same job**: `current` (or `cumulative`), `roi_current` (or `roi_cumulative`), `roi_rectangle`
- Data requirements:
  - Single dataset only (`multiple_datasets: False`)
  - Must be from `detector_data` namespace
  - Must have 2D detector image data
- Returns `hv.Layout` with:
  - Left: 2D detector image with BoxEdit overlay
  - Right: 1D TOA spectrum

### **2. BoxEdit Integration Strategy**
- BoxEdit stream attached to the 2D detector image element
- Subscribe to BoxEdit `data` parameter changes
- On change: Serialize rectangle to `RectangleROI` model → publish to config service
- Target stream: `{job_number}/roi_rectangle` (extract job_number from ResultKey)

### **3. ROI Shape Display**
- Subscribe to `roi_rectangle` output stream from backend
- When received: Parse `RectangleROI` from DataArray → update BoxEdit overlay
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
- PlottingController needs access to `ConfigService` (may already have?)
- Create method: `publish_roi_update(job_number, roi_model)`
- Serialize `RectangleROI` → `DataArray` → publish to stream `{job_number}/roi_rectangle`
- Use existing Kafka infrastructure (same as workflow configs)

## Open Questions

**Q1: Publishing mechanism details**
- Should we publish ROI updates through `ConfigService.update_config()` or a different mechanism?
- What's the exact ConfigKey format for aux source streams `{job_number}/roi_rectangle`?
- Or should we use a different Kafka topic entirely (data topic vs config topic)?

**Q2: ConfigService access in PlottingController**
- Does `PlottingController` currently have access to `ConfigService`?
- If not, should we inject it as a dependency?

**Q3: Graceful degradation**
- When ROI outputs don't exist yet, should we:
  - Show empty/placeholder 1D plot?
  - Show error message?
  - Hide 1D plot until data arrives?

**Q4: Multiple detector images**
- You mentioned the Layout/Overlay restriction is fine - so ROI plotter only accepts **single source selection**, correct?
- Should we enforce this in the data requirements or let the configuration widget handle it?

**Q5: Hard-coding rectangle for now**
- Confirm: For MVP, we hard-code `hv.streams.BoxEdit` and ignore polygon/ellipse options?
- Future work: Read aux_sources config to determine which tool to show?

## Next Steps

Once you clarify the open questions above, I can proceed with:
1. Implementing `ROIDetectorPlotter` class
2. Setting up BoxEdit integration and Kafka publishing
3. Handling subscription to ROI shape updates from backend
4. Testing with the existing detector view workflows
