# Implementation Plan: Stacked ROI Spectra Outputs

## Overview

Replace the current approach of publishing individual ROI spectrum messages (`roi_current_0`, `roi_current_1`, etc.) with a single stacked 2D DataArray output. This simplifies the output model, reduces message count, and enables proper declaration in `DetectorViewOutputs`.

## Implementation Phases

1. **Phase 1** ✅ COMPLETE: Add new 2D stacked outputs (backend) - keep old individual outputs
2. **Phase 2**: Implement new frontend plotter consuming 2D outputs
3. **Phase 3**: Remove old individual outputs and cleanup

---

## Phase 1: Backend - Add Stacked Outputs ✅ COMPLETE

### Changes Made

#### `DetectorViewOutputs` (`detector_view_specs.py`)

New output fields added:

- `roi_spectra_current`: 2D stacked array (`roi` × `time_of_arrival`) with `roi_index` and `time` coords
- `roi_spectra_cumulative`: 2D stacked array with `roi_index` coord (no `time` coord)
- `roi_rectangle`: ROI geometry readback (now declared in output model)
- `roi_polygon`: ROI geometry readback (now declared in output model)

#### `DetectorView.finalize()` (`detector_view.py`)

- Produces stacked outputs alongside existing individual outputs
- ROIs stacked in sorted order by index (for consistent color mapping)
- Empty case produces valid 2D array with shape `[0, n_toa_bins]`

#### Key Implementation Details

- **Dimension naming**: Uses `time_of_arrival` (matching existing `ROIHistogram` code)
- **`roi_index` coordinate**: Uses `unit=None` (just an index, not a dimensionless quantity)
- **`time` coordinate**: Only on `roi_spectra_current`, not on cumulative
- **ROI readback titles**: Must be `'rectangles'` and `'polygons'` to match DataArray names (see Known Issues below)

### Tests Added

New test class `TestDetectorViewStackedROISpectra` in `detector_view_test.py` covering:
- Correct 2D shape and dimensions
- `roi_index` coordinate values and unit
- Sorted stacking order
- Time coordinate presence
- Cumulative accumulation
- Empty case handling
- Multiple geometry types
- Data consistency with individual outputs

---

## Phase 2: Frontend - New Stacked Spectrum Plotter

Update `ROIDetectorPlotFactory` to consume the new stacked outputs instead of individual ROI keys.

### Key Changes

1. Subscribe to single `roi_spectra_current`/`roi_spectra_cumulative` key instead of 8 individual keys
2. Iterate over `roi` dimension to extract individual spectra
3. Color by array position (matches detector overlay color assignment)
4. Use `roi_index` coordinate for legend labels

### Color Consistency

Colors are assigned by **position along the `roi` dimension**, not by `roi_index` value:

| Array Position | `roi_index` Value | Color |
|----------------|-------------------|-------|
| 0 | 0 | `colors[0]` |
| 1 | 3 | `colors[1]` |
| 2 | 5 | `colors[2]` |

This naturally matches the detector overlay because both iterate in sorted index order.

---

## Phase 3: Cleanup - Remove Old Individual Outputs

Once Phase 2 is verified working:

1. **Backend**: Remove individual ROI output generation from `finalize()`
2. **`roi_names.py`**: Remove `current_key()`, `cumulative_key()`, `all_histogram_keys()` methods
3. **Frontend**: Remove `_generate_spectrum_keys()` and individual key references
4. **Tests**: Update to remove individual output tests

### Breaking Changes (Phase 3 only)

- Output keys `roi_current_0`, `roi_cumulative_0`, etc. will no longer be published
- External consumers must switch to `roi_spectra_current`/`roi_spectra_cumulative`

---

## Known Issues

### ROI Readback Title/Name Coupling

The `roi_rectangle` and `roi_polygon` fields require their `title` to match the DataArray `.name` attribute produced by `to_concatenated_data_array()` (i.e., `'rectangles'` and `'polygons'`).

**Why**: `job_manager.py` overwrites `DataArray.name` with the Pydantic field `title`. The ROI parser relies on the name to determine the ROI type.

**Impact**: This coupling between field title and serialization format is fragile. If titles are changed, ROI parsing breaks with "Unknown ROI type" errors.

**Future work**: Decouple the display title from the serialization name, possibly by:
- Not overwriting DataArray.name in job_manager for certain output types
- Using a separate attribute for ROI type identification
- Adding explicit ROI type metadata

---

## Migration Notes

| Phase | Breaking Changes | Backward Compatible |
|-------|------------------|---------------------|
| Phase 1 ✅ | None | Yes - old outputs still published |
| Phase 2 | None | Yes - frontend switches to new outputs |
| Phase 3 | Yes | No - old outputs removed |

Recommended deployment: Phase 1 + Phase 2 together, then Phase 3 as separate release.
