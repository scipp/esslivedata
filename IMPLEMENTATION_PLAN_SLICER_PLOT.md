# Implementation Plan: 3D Slicer Plotter

## Overview

Add a new `SlicerPlotter` to the ESSlivedata dashboard that allows interactive exploration of 3D data by slicing along one dimension with a slider control.

## Architecture Analysis

### Current State Assessment ✅

The existing architecture is well-prepared for this feature:

1. **Plotter Registry System** ([plotting.py](src/ess/livedata/dashboard/plotting.py)):
   - Extensible registry with `DataRequirements` validation
   - Factory pattern with Pydantic parameter models
   - Already supports 1D (`LinePlotter`) and 2D (`ImagePlotter`) data

2. **DynamicMap Integration** ([plotting_controller.py:323](src/ess/livedata/dashboard/plotting_controller.py#L323)):
   - Uses `hv.DynamicMap(plotter, streams=[pipe])` pattern
   - **Supports multiple streams** - we can add slider stream alongside data pipe
   - Cache management already configured (`cache_size=1`)

3. **Autoscaling** ([plots.py:122-128](src/ess/livedata/dashboard/plots.py#L122-L128)):
   - Per-`ResultKey` autoscalers track min/max bounds
   - Updates cumulatively as new data arrives
   - **Should maintain consistent bounds across all slices** once all slices have been viewed

### Key Design Decisions

#### 1. Stream Architecture: Multiple Streams in DynamicMap

The DynamicMap will have **two streams**:

```python
# In PlottingController.create_plot()
hv.DynamicMap(plotter, streams=[data_pipe, plotter.slice_stream])
```

- **Data Pipe** (existing): Pushes new 3D data from Kafka/JobService
- **Slice Stream** (new): User-controlled slider that updates slice index

**Why this works**: HoloViews DynamicMap reruns the plotter callable whenever ANY stream triggers an update.

#### 2. Slice Index Display: Show Current Position

**Solution**: Display slice information in the plot title using coord values when available:

```python
# If coord exists for slice dimension:
title = f"{data.name} - {slice_dim}={coord_value:.3f} {coord.unit} (index {slice_idx}/{max_idx})"

# If no coord (just integer indices):
title = f"{data.name} - {slice_dim}[{slice_idx}/{max_idx}]"
```

This provides clear feedback about which slice is currently displayed.

#### 3. Autoscaling Behavior: Consistent Bounds Across Slices

**Desired behavior**: Color scale and spatial axes should remain consistent when moving the slider.

**How `Autoscaler` handles this**:
- Each `Autoscaler` instance maintains cumulative min/max bounds
- `update_bounds()` returns `True` if bounds changed, `False` otherwise
- Once all slices viewed, bounds stabilize → no more rescaling
- This provides **consistent visual scale** for comparing slices

**Alternative option** (for future): Add a "global autoscale" mode that computes bounds from the entire 3D volume upfront.

#### 4. Slice Dimension Selection

**Initial implementation**: Single parameter in `PlotParams3d` specifying which dimension to slice:

```python
class PlotParams3d(PlotParamsBase):
    slice_dimension: str = Field(
        description="Dimension to slice along (other two dims shown in image)"
    )
    plot_scale: PlotScaleParams2d = Field(...)
```

**Future enhancement**: Auto-detect "sliceable" dimensions (dims not needed for 2D image display).

#### 5. Slider Range: Dynamic Based on Data

**Challenge**: Different data sources or workflow updates may have different sizes along the slice dimension.

**Solution**:
- Store reference to the data shape in the plotter
- Update slider bounds when data shape changes
- Use `hv.streams.Stream.define()` with mutable parameters

```python
class SlicerPlotter(Plotter):
    def __init__(self, ...):
        # Define custom stream with slice_index parameter
        SliceIndex = hv.streams.Stream.define('SliceIndex', slice_index=0)
        self.slice_stream = SliceIndex()
        self._max_slice_idx = None  # Will be set from data
```

## Implementation Plan

### Phase 1: Core Slicer Implementation

#### Task 1.1: Create `PlotParams3d` Model

**File**: [src/ess/livedata/dashboard/plot_params.py](src/ess/livedata/dashboard/plot_params.py)

```python
class PlotParams3d(PlotParamsBase):
    """Parameters for 3D slicer plots."""

    slice_dimension: str = pydantic.Field(
        description="Dimension to slice along. The other two dimensions will be displayed as a 2D image.",
        title="Slice Dimension",
    )

    initial_slice_index: int = pydantic.Field(
        default=0,
        ge=0,
        description="Initial slice index to display.",
        title="Initial Slice Index",
    )

    plot_scale: PlotScaleParams2d = pydantic.Field(
        default_factory=PlotScaleParams2d,
        description="Scaling options for the plot axes and color.",
    )
```

**Validation**: The `slice_dimension` must be validated against actual data dimensions at plot creation time.

#### Task 1.2: Implement `SlicerPlotter` Class

**File**: [src/ess/livedata/dashboard/plots.py](src/ess/livedata/dashboard/plots.py)

```python
class SlicerPlotter(Plotter):
    """Plotter for 3D data with interactive slicing."""

    def __init__(
        self,
        slice_dim: str,
        initial_slice_index: int,
        scale_opts: PlotScaleParams2d,
        **kwargs,
    ):
        """
        Initialize the slicer plotter.

        Parameters
        ----------
        slice_dim:
            The dimension to slice along.
        initial_slice_index:
            Initial slice index to display.
        scale_opts:
            Scaling options for axes and color.
        **kwargs:
            Additional keyword arguments passed to the base class.
        """
        super().__init__(**kwargs)
        self._slice_dim = slice_dim
        self._scale_opts = scale_opts
        self._max_slice_idx = {}  # Per-ResultKey max slice index

        # Create custom stream for slice selection
        # This will automatically create a slider widget
        SliceIndex = hv.streams.Stream.define(
            'SliceIndex',
            slice_index=initial_slice_index
        )
        self.slice_stream = SliceIndex()

        # Base options for the image plot (similar to ImagePlotter)
        self._base_opts = {
            'colorbar': True,
            'cmap': 'viridis',
            'logx': scale_opts.x_scale == PlotScale.log,
            'logy': scale_opts.y_scale == PlotScale.log,
            'logz': scale_opts.color_scale == PlotScale.log,
        }

    @classmethod
    def from_params(cls, params: PlotParams3d):
        """Create SlicerPlotter from PlotParams3d."""
        return cls(
            slice_dim=params.slice_dimension,
            initial_slice_index=params.initial_slice_index,
            scale_opts=params.plot_scale,
            value_margin_factor=0.1,
            layout_params=params.layout,
            aspect_params=params.plot_aspect,
        )

    def _validate_and_update_bounds(
        self, data: dict[ResultKey, sc.DataArray]
    ) -> None:
        """Validate slice dimension and update slider bounds if needed."""
        for data_key, da in data.items():
            # Validate that slice_dim exists in the data
            if self._slice_dim not in da.dims:
                raise ValueError(
                    f"Slice dimension '{self._slice_dim}' not found in data. "
                    f"Available dimensions: {da.dims}"
                )

            # Check if max slice index changed
            new_max = da.sizes[self._slice_dim] - 1
            if data_key not in self._max_slice_idx or self._max_slice_idx[data_key] != new_max:
                self._max_slice_idx[data_key] = new_max
                # TODO: Update slider bounds - need to figure out how to do this
                # The slider is created by HoloViews/Panel from the stream definition
                # May need to recreate the stream or use a different approach

    def _get_slice_index(self) -> int:
        """Get current slice index, clipped to valid range."""
        current_idx = self.slice_stream.slice_index
        if not self._max_slice_idx:
            return current_idx

        # Clip to the minimum of all max indices across datasets
        max_idx = min(self._max_slice_idx.values())
        return min(current_idx, max_idx)

    def _format_slice_label(
        self, data: sc.DataArray, slice_idx: int
    ) -> str:
        """Format a label showing the current slice position."""
        max_idx = data.sizes[self._slice_dim] - 1

        # Try to get coordinate value at this slice
        if self._slice_dim in data.coords:
            coord = data.coords[self._slice_dim]
            # Handle both edge and point coordinates
            if data.coords.is_edges(self._slice_dim):
                # For edges, show the bin center
                value = sc.midpoints(coord, dim=self._slice_dim)[slice_idx]
            else:
                value = coord[slice_idx]

            # Format with unit if available
            if value.unit:
                label = f"{self._slice_dim}={value.value:.3g} {value.unit}"
            else:
                label = f"{self._slice_dim}={value.value:.3g}"
            label += f" (slice {slice_idx}/{max_idx})"
        else:
            # No coordinate, just show index
            label = f"{self._slice_dim}[{slice_idx}/{max_idx}]"

        return label

    def plot(self, data: sc.DataArray, data_key: ResultKey) -> hv.Image:
        """
        Create a 2D image from a slice of 3D data.

        Parameters
        ----------
        data:
            3D DataArray to slice.
        data_key:
            Key identifying this data.

        Returns
        -------
        :
            A HoloViews Image element showing the selected slice.
        """
        slice_idx = self._get_slice_index()

        # Slice the 3D data to get 2D
        sliced_data = data[self._slice_dim, slice_idx]

        # Apply log masking if needed (same as ImagePlotter)
        if self._scale_opts.color_scale == PlotScale.log:
            plot_data = sliced_data.to(dtype='float64')
            plot_data = plot_data.assign(
                sc.where(
                    plot_data.data <= sc.scalar(0.0, unit=plot_data.unit),
                    sc.scalar(np.nan, unit=plot_data.unit, dtype=plot_data.dtype),
                    plot_data.data,
                )
            )
        else:
            plot_data = sliced_data.to(dtype='float64')

        # Update autoscaler and get framewise flag
        framewise = self._update_autoscaler_and_get_framewise(plot_data, data_key)

        # Create the image
        image = to_holoviews(plot_data)

        # Add slice information to title
        slice_label = self._format_slice_label(data, slice_idx)
        title = f"{data.name or 'Data'} - {slice_label}"

        return image.opts(
            framewise=framewise,
            title=title,
            **self._base_opts
        )

    def __call__(
        self, data: dict[ResultKey, sc.DataArray]
    ) -> hv.Overlay | hv.Layout | hv.Element:
        """
        Create plots from 3D data, slicing at the current index.

        Parameters
        ----------
        data:
            Dictionary of 3D DataArrays to plot.

        Returns
        -------
        :
            HoloViews element(s) showing the sliced data.
        """
        # Validate dimensions and update slider bounds
        self._validate_and_update_bounds(data)

        # Use parent implementation which calls self.plot() for each dataset
        return super().__call__(data)
```

**Key design points**:
- Stream is created in `__init__` as an instance attribute
- Slice index is read from stream in `plot()` method
- Slice label with coord value is added to plot title
- Autoscaler handles bounds across slices automatically
- Handles both edge and point coordinates

#### Task 1.3: Register Slicer in Plotter Registry

**File**: [src/ess/livedata/dashboard/plotting.py](src/ess/livedata/dashboard/plotting.py)

Add after the existing plotter registrations:

```python
from .plots import ImagePlotter, LinePlotter, SlicerPlotter  # Add SlicerPlotter

# ... existing registrations ...

plotter_registry.register_plotter(
    name='slicer',
    title='3D Slicer',
    description='Interactively slice through 3D data along one dimension.',
    data_requirements=DataRequirements(min_dims=3, max_dims=3),
    factory=SlicerPlotter.from_params,
)
```

#### Task 1.4: Update `PlottingController.create_plot()` to Handle Additional Streams

**File**: [src/ess/livedata/dashboard/plotting_controller.py](src/ess/livedata/dashboard/plotting_controller.py)

**Current code** (line 323):
```python
return hv.DynamicMap(plotter, streams=[pipe], cache_size=1).opts(
    shared_axes=False
)
```

**Updated code**:
```python
# Collect all streams: data pipe + any plotter-specific streams
streams = [pipe]

# Check if plotter has additional streams (e.g., slice_stream for SlicerPlotter)
if hasattr(plotter, 'slice_stream'):
    streams.append(plotter.slice_stream)

return hv.DynamicMap(plotter, streams=streams, cache_size=1).opts(
    shared_axes=False
)
```

**Rationale**: This allows plotters to optionally provide additional streams without breaking existing plotters.

### Phase 2: Testing and Refinement

#### Task 2.1: Create Unit Tests

**File**: [tests/dashboard/plots_test.py](tests/dashboard/plots_test.py)

Test cases:
1. **Initialization**: Verify `SlicerPlotter` initializes with correct parameters
2. **Slicing**: Test that `plot()` correctly slices 3D data at different indices
3. **Bounds validation**: Test error handling for invalid slice dimensions
4. **Label formatting**: Test slice label generation with/without coords
5. **Edge coordinates**: Test handling of edge vs point coordinates
6. **Multiple datasets**: Test plotting multiple 3D datasets together

Example test:
```python
def test_slicer_plotter_slices_3d_data():
    """Test that SlicerPlotter correctly slices 3D data."""
    # Create 3D test data
    data = sc.data.table_xyz(100, 50, 30)  # 100 x 50 x 30

    # Create plotter
    params = PlotParams3d(
        slice_dimension='z',
        initial_slice_index=5,
        plot_scale=PlotScaleParams2d(),
    )
    plotter = SlicerPlotter.from_params(params)

    # Create test data dict
    result_key = ResultKey(...)
    data_dict = {result_key: data}

    # Call plotter
    plot = plotter(data_dict)

    # Verify it created a 2D plot from the 3D data
    assert isinstance(plot, hv.Image)
    # Add more assertions...
```

#### Task 2.2: Create Integration Test with Fake Data

**File**: New file `tests/dashboard/slicer_integration_test.py`

Test the full workflow:
1. Create fake 3D workflow output
2. Register with JobService
3. Create slicer plot through PlottingController
4. Verify DynamicMap has both streams
5. Simulate slider changes by updating the slice_stream
6. Verify plot updates correctly

#### Task 2.3: Manual Testing with Real Instrument Data

Test with actual instrument configurations:
- DREAM: 3D diffraction data (Q-space volumes)
- Bifrost: Time-energy-Q data
- LOKI: 3D SANS data

Verify:
- Slider appears and is functional
- Slice labels are informative
- Autoscaling provides consistent bounds
- Performance is acceptable (no lag when moving slider)

### Phase 3: Enhancements (Future Work)

#### Enhancement 3.1: Slider Bounds Update

**Issue**: If data shape changes between updates, slider max should update.

**Solution**: Investigate HoloViews stream parameter updating:
```python
# Potential approach:
self.slice_stream.event(slice_index=self.slice_stream.slice_index)
# Or recreate stream with new bounds
```

#### Enhancement 3.2: Multiple Slice Dimensions

Allow slicing along multiple dimensions with multiple sliders:
```python
class PlotParams3dMultiSlice(PlotParamsBase):
    slice_dimensions: list[str] = Field(
        description="Dimensions to slice (one slider per dimension)"
    )
    display_dimensions: list[str] = Field(
        description="Two dimensions to display as image",
        min_items=2,
        max_items=2,
    )
```

#### Enhancement 3.3: Global Autoscaling Option

Add option to compute bounds from entire 3D volume upfront:
```python
class PlotParams3d(PlotParamsBase):
    # ... existing fields ...
    autoscale_mode: Literal['per_slice', 'global'] = Field(
        default='per_slice',
        description="Whether to autoscale per slice or use global bounds"
    )
```

#### Enhancement 3.4: Slice Animation

Add play button to animate through slices automatically (similar to plopp's `enable_player`).

## Implementation Checklist

### Phase 1: Core Implementation
- [ ] Task 1.1: Add `PlotParams3d` to `plot_params.py`
- [ ] Task 1.2: Implement `SlicerPlotter` in `plots.py`
- [ ] Task 1.3: Register slicer in `plotting.py`
- [ ] Task 1.4: Update `PlottingController.create_plot()` to handle additional streams
- [ ] Update imports in `__init__.py` if needed

### Phase 2: Testing
- [ ] Task 2.1: Write unit tests for `SlicerPlotter`
- [ ] Task 2.2: Write integration test with fake data
- [ ] Task 2.3: Manual testing with instrument data
- [ ] Verify autoscaling behavior across slices
- [ ] Verify performance (responsiveness of slider)

### Phase 3: Documentation
- [ ] Add docstrings to all new classes and methods
- [ ] Add example usage to developer docs
- [ ] Update user guide with 3D slicer feature

## Technical Notes

### HoloViews Stream Details

**Stream definition**:
```python
SliceIndex = hv.streams.Stream.define('SliceIndex', slice_index=0)
```

This creates a new stream class with a `slice_index` parameter. When used in a DynamicMap, HoloViews/Panel automatically creates a slider widget for this parameter.

**Stream parameter bounds**: Need to investigate how to set min/max for the slider. May need to use Panel widgets directly:
```python
import panel as pn
slider = pn.widgets.IntSlider(start=0, end=max_idx, value=0)
# Connect slider to stream somehow
```

### Autoscaler Behavior

The existing `Autoscaler` ([plots.py:122-128](src/ess/livedata/dashboard/plots.py#L122-L128)):
- Maintains separate instances per `ResultKey`
- `update_bounds(data)` expands bounds to include new data
- Returns `True` if bounds changed (triggers rescaling), `False` otherwise
- After viewing all slices once, bounds should stabilize

**Expected behavior**:
1. First time viewing slice 5 → bounds expand → plot rescales
2. Second time viewing slice 5 → bounds unchanged → no rescale
3. After viewing all slices → consistent bounds across all slices

This provides the desired behavior without modification!

### Data Shape Validation

The `DataRequirements` system already validates dimensions at selection time:
```python
DataRequirements(min_dims=3, max_dims=3)
```

This ensures only 3D data can use the slicer plotter.

Additional runtime validation in `SlicerPlotter._validate_and_update_bounds()` ensures the specified `slice_dim` actually exists in the data.

## Open Questions

1. **Slider bounds updating**: How to update slider max value when data shape changes?
   - May need to use Panel widgets directly instead of HoloViews streams
   - Or accept that slider bounds are set from first data and may be incorrect if shape changes

2. **Multi-dataset slicing**: When plotting multiple datasets, should they all use the same slice index?
   - Current design: Yes, single slider controls all datasets
   - Alternative: Separate sliders per dataset (more complex UI)

3. **Dimension selection UI**: Should the UI help users select which dimension to slice?
   - Current design: User specifies dimension name in params
   - Alternative: Dropdown populated from available dimensions in data

4. **Default slice dimension**: How to choose sensible default?
   - Option 1: First dimension (alphabetically)
   - Option 2: Dimension with most slices
   - Option 3: No default, make it required

## Success Criteria

✅ 3D data can be visualized with interactive slicing
✅ Slider widget appears and controls which slice is displayed
✅ Current slice position is clearly indicated (index + coord value)
✅ Autoscaling provides consistent bounds across slices after initial exploration
✅ Works with streaming data from Kafka (plot updates when new 3D data arrives)
✅ Compatible with existing dashboard architecture
✅ Unit and integration tests pass
✅ Performance is acceptable (no noticeable lag)

## Timeline Estimate

- **Phase 1** (Core Implementation): 1-2 days
- **Phase 2** (Testing & Refinement): 1 day
- **Phase 3** (Documentation): 0.5 days
- **Total**: 2.5-3.5 days

## References

- [HoloViews DynamicMap](https://holoviews.org/reference/containers/bokeh/DynamicMap.html)
- [HoloViews Streams](https://holoviews.org/user_guide/Responding_to_Events.html)
- [Plopp Slicer](https://scipp.github.io/plopp/about/generated/plopp.slicer.html) (inspiration, different framework)
- Existing plotter implementations: [plots.py](src/ess/livedata/dashboard/plots.py)
