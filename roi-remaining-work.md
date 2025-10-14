# ROI Feature - Remaining Work

## Status Overview

The ROI detector plotter is implemented and integrated. Users can create rectangles using BoxEdit. The backend infrastructure is ready to receive ROI updates.

## Remaining Tasks

### 1. Limit BoxEdit to Single Rectangle ✅ DONE

BoxEdit already has `num_objects=3` configured in [plotting_controller.py:377](src/ess/livedata/dashboard/plotting_controller.py#L377). Change to `num_objects=1`.

---

### 2. Publishing ROI Updates ✅ DONE

**Implementation summary**:
- Created `ROIPublisher` class in [roi_publisher.py](src/ess/livedata/dashboard/roi_publisher.py)
  - Owns a `KafkaSink` with DA00 serializer
  - Provides `publish_roi()` and `publish_rois()` methods
  - Includes `FakeROIPublisher` for testing
- Created `boxes_to_rois()` helper function
  - Converts BoxEdit data format to `RectangleROI` instances
  - Handles inverted coordinates and degenerate boxes
- Updated `PlottingController`:
  - Added optional `roi_publisher` parameter to `__init__`
  - Added `_setup_roi_watcher()` method to watch BoxEdit stream
  - On box change: Parse → convert to `RectangleROI` → publish
  - **Change tracking**: Only publishes if ROI actually changed (avoids redundant Kafka messages)
- Stream naming: `{job_number}/roi_rectangle_{index}`
- Limited BoxEdit to 1 rectangle (`num_objects=1`)
- Added comprehensive test coverage (11 new tests in `roi_publisher_test.py`, 3 integration tests)

**Why not ConfigService?**:
- ConfigService is for **bidirectional** config sync (dashboard ↔ backend services)
- ROI publishing is **unidirectional** (dashboard → backend only)
- No need for schema registry, throttling, or readback deduplication
- Simpler pattern = less coupling, easier testing

---

### 3. ROI Shape Display (Readback) ❌

Display the ROI that the backend is actually using (feedback loop):

- Subscribe to `roi_rectangle` backend output stream via `JobService`/`StreamManager`
- Parse `RectangleROI` from DataArray → create static shape overlay (solid lines, not editable)
- Implement dual overlay system: dashed lines for user "request" (BoxEdit), solid lines for backend "readback"
- Keep BoxEdit independent (user retains control even after readback arrives)

**Note**: May defer this if backend always uses requested ROI (no validation/snapping). Useful for debugging but not critical for MVP.

---

## What's Already Done ✅

- ROIDetectorPlotter foundation with BoxEdit overlay
- Backend ROI infrastructure (topic, routing, subscriptions)
- Plotter registration with PlottingController
- Comprehensive test coverage (93 tests passing)
- Graceful degradation when ROI spectrum data is missing
- `RectangleROI.to_data_array()` / `from_data_array()` serialization
- DA00 adapter for LIVEDATA_ROI stream kind
- KafkaSink infrastructure with DA00 serialization

---

## Original Plan Reference

See [roi-plot-implementation-plan.md](roi-plot-implementation-plan.md) for the original detailed implementation plan.
