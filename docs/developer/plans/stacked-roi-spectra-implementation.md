# Implementation Plan: Stacked ROI Spectra Outputs

## Overview

Replace the current approach of publishing individual ROI spectrum messages (`roi_current_0`, `roi_current_1`, etc.) with a single stacked 2D DataArray output. This simplifies the output model, reduces message count, and enables proper declaration in `DetectorViewOutputs`.

## Implementation Phases

1. **Phase 1** ✅ COMPLETE: Add new 2D stacked outputs (backend) - keep old individual outputs
2. **Phase 2** ✅ COMPLETE: Implement new frontend plotter consuming 2D outputs
3. **Phase 3** ✅ COMPLETE: Remove old individual outputs and cleanup

---

## Phase 1: Backend - Add Stacked Outputs ✅ COMPLETE

### Changes Made

#### `DetectorViewOutputs` (`detector_view_specs.py`)

New output fields added:

- `roi_spectra_current`: 2D stacked array (`roi` × `time_of_arrival`) with `roi` and `time` coords
- `roi_spectra_cumulative`: 2D stacked array with `roi` coord (no `time` coord)
- `roi_rectangle`: ROI geometry readback (now declared in output model)
- `roi_polygon`: ROI geometry readback (now declared in output model)

#### `DetectorView.finalize()` (`detector_view.py`)

- Produces stacked outputs alongside existing individual outputs
- ROIs stacked in sorted order by index (for predictable iteration and legend ordering)
- Empty case produces valid 2D array with shape `[0, n_toa_bins]`

#### Key Implementation Details

- **Dimension naming**: Uses `time_of_arrival` (matching existing `ROIHistogram` code)
- **`roi` coordinate**: Uses `unit=None` (just an index, not a dimensionless quantity)
- **`time` coordinate**: Only on `roi_spectra_current`, not on cumulative
- **ROI readback titles**: Must be `'rectangles'` and `'polygons'` to match DataArray names (see Known Issues below)

### Tests Added

New test class `TestDetectorViewStackedROISpectra` in `detector_view_test.py` covering:
- Correct 2D shape and dimensions
- `roi` coordinate values and unit
- Sorted stacking order
- Time coordinate presence
- Cumulative accumulation
- Empty case handling
- Multiple geometry types
- Data consistency with individual outputs

---

## Phase 2: Frontend - New Stacked Spectrum Plotter ✅ COMPLETE

### Changes Made

#### `Overlay1DPlotter` (`plots.py`)

A new generic plotter that slices 2D data along the first dimension and overlays as 1D curves:

- Takes 2D data with dims `[slice_dim, plot_dim]`
- Iterates over the first dimension to extract individual spectra
- Uses coordinate values from the first dimension for legend labels (e.g., `roi=0`, `roi=3`)
- Colors assigned by coordinate value for stable identity

#### Plotter Registration (`plotting.py`)

Registered as `overlay_1d` with:
- `min_dims=2, max_dims=2`
- `multiple_datasets=False`
- Uses `PlotParams1d` (1D curve output, not 2D image)

#### Color Consistency Fix (`roi_detector_plot_factory.py`)

Updated `_compute_index_to_color()` to use index-based coloring:

```python
def _compute_index_to_color(self) -> dict[int, str]:
    return {
        idx: self._colors[idx % len(self._colors)]
        for idx in self._active_roi_indices
    }
```

This provides **stable color identity** - ROI 3 always has the same color regardless of which other ROIs are active, matching the `Overlay1DPlotter` coloring.

#### Shared Helper (`plots.py`)

Added `Plotter._convert_histogram_to_curve_data()` static method:
- Converts bin-edge coordinates to midpoints for curve plotting
- Histograms with many narrow bins show black outlines; curves display cleanly
- Used by both `LinePlotter` and `Overlay1DPlotter`

### Tests Added

New test class `TestOverlay1DPlotter` in `plots_test.py` covering:
- Overlay creation from 2D data
- Correct number of curves per slice
- Single slice returns single element (not overlay)
- Empty first dimension handling
- Labels using coordinate values
- Colors assigned by coordinate value
- Fallback to indices without coordinates
- Rejection of non-2D data
- Bokeh rendering
- Registry compatibility checks
- Bin-edge coordinates produce Curves (not Histograms)

---

## Phase 3: Cleanup - Remove Old Individual Outputs ✅ COMPLETE

### Changes Made

1. **Backend**: Removed individual ROI output generation from `DetectorView.finalize()`
   - No longer produces `roi_current_N` and `roi_cumulative_N` keys
   - Only stacked outputs (`roi_spectra_current`, `roi_spectra_cumulative`) are published

2. **`roi_names.py`**: Removed deprecated methods:
   - `current_key()`
   - `cumulative_key()`
   - `all_current_keys()`
   - `all_cumulative_keys()`
   - `all_histogram_keys()`
   - `parse_roi_index()`
   - `_roi_index_pattern` regex

3. **Frontend**: Spectrum overlay was already removed from `ROIDetectorPlotFactory` in Phase 2
   (now handled by `Overlay1DPlotter`)

4. **Tests**: Updated all tests to use stacked outputs:
   - `roi_names_test.py`: Removed tests for deprecated methods
   - `detector_view_test.py`: Updated helper functions and 40+ test assertions to use stacked outputs

### Breaking Changes

- Output keys `roi_current_0`, `roi_cumulative_0`, etc. are no longer published
- External consumers must use `roi_spectra_current`/`roi_spectra_cumulative` instead

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
| Phase 2 ✅ | None | Yes - new plotter available, old outputs still work |
| Phase 3 ✅ | Yes | No - old outputs removed |

All phases complete. The stacked ROI spectra implementation is now the only output format.
