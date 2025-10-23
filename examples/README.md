# PlotGrid Demo

This directory contains example applications demonstrating ESSlivedata dashboard widgets.

## PlotGrid Demo

The `plot_grid_demo.py` demonstrates the PlotGrid widget, which allows users to:

- Create a customizable grid layout (configurable rows and columns)
- Select cells or rectangular regions by clicking
- Insert HoloViews DynamicMap plots into the grid
- Remove plots using the close button
- Prevent overlapping plot selections

### Running the Demo

```bash
panel serve examples/plot_grid_demo.py --show
```

This will start a local Panel server and open the demo in your default browser.

### Features Demonstrated

1. **Two-click region selection**: Click a cell to start selection, click another cell to complete it
2. **Multiple plot types**: Choose from curves, scatter plots, heatmaps, and bar charts
3. **Dynamic plot insertion**: Each plot is a HoloViews DynamicMap with interactive widgets
4. **Plot removal**: Click the × button on any plot to remove it
5. **Overlap prevention**: Cannot select cells that overlap with existing plots

### Implementation Notes

The PlotGrid widget is standalone and does not depend on the rest of the ESSlivedata infrastructure (controllers, services, etc.). It only requires:

- A callback function that returns a `hv.DynamicMap`
- Grid dimensions (nrows, ncols)

This makes it easy to integrate into any Panel application.
