#!/usr/bin/env python
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""
Demo application for the PlotGrid widget.

This script demonstrates the PlotGrid widget with randomly generated
HoloViews plots. Users can select cells or regions in the grid and
insert different types of plots.

Run with:
    panel serve examples/plot_grid_demo.py --show
"""

from __future__ import annotations

import holoviews as hv
import numpy as np
import panel as pn

from ess.livedata.dashboard.widgets.plot_grid import PlotGrid

# Enable Panel and HoloViews extensions
pn.extension('tabulator')
hv.extension('bokeh')


class PlotGridDemo:
    """Demo application for PlotGrid widget."""

    def __init__(self) -> None:
        self._plot_counter = 0
        self._plot_types = ['curve', 'scatter', 'heatmap', 'bars']
        self._current_plot_type = 'curve'

        # Create plot type selector
        self._plot_type_selector = pn.widgets.RadioButtonGroup(
            name='Plot Type',
            options=self._plot_types,
            value=self._current_plot_type,
            button_type='primary',
        )
        self._plot_type_selector.param.watch(self._on_plot_type_change, 'value')

        # Create grid size controls
        self._nrows_input = pn.widgets.IntInput(
            name='Number of Rows', value=3, start=1, end=10, step=1
        )
        self._ncols_input = pn.widgets.IntInput(
            name='Number of Columns', value=3, start=1, end=10, step=1
        )
        self._recreate_button = pn.widgets.Button(
            name='Recreate Grid', button_type='warning'
        )
        self._recreate_button.on_click(self._on_recreate_grid)

        # Create the plot grid
        self._grid = PlotGrid(
            nrows=3,
            ncols=3,
            plot_request_callback=self._create_random_plot,
        )

        # Instructions
        self._instructions = pn.pane.Markdown(
            """
            ## PlotGrid Demo

            **Instructions:**
            1. Select the type of plot you want to create using the radio buttons above
            2. Click a cell in the grid to start selection
            3. Click another cell (or the same cell) to complete the selection
            4. A plot will be inserted into the selected region
            5. Click the close button on any plot to remove it

            **Features:**
            - Select single cells or rectangular regions
            - Multiple plots can coexist in the grid
            - Cannot select cells that overlap existing plots
            - Plots are randomly generated for demonstration purposes
            """
        )

    def _on_plot_type_change(self, event: pn.widgets.Widget) -> None:
        """Update the current plot type."""
        self._current_plot_type = event.new

    def _on_recreate_grid(self, event: pn.widgets.Button) -> None:
        """Recreate the grid with new dimensions."""
        self._grid = PlotGrid(
            nrows=self._nrows_input.value,
            ncols=self._ncols_input.value,
            plot_request_callback=self._create_random_plot,
        )
        # Update the layout (this would need to be handled by the parent layout)
        # For now, we just show a notification
        pn.state.notifications.info(
            'Grid recreated! (refresh the page to see changes)', duration=3000
        )

    def _create_random_plot(self) -> hv.DynamicMap:
        """Create a random plot based on the selected plot type."""
        self._plot_counter += 1
        plot_name = f'{self._current_plot_type.capitalize()} {self._plot_counter}'

        if self._current_plot_type == 'curve':
            return self._create_curve_plot(plot_name)
        elif self._current_plot_type == 'scatter':
            return self._create_scatter_plot(plot_name)
        elif self._current_plot_type == 'heatmap':
            return self._create_heatmap_plot(plot_name)
        elif self._current_plot_type == 'bars':
            return self._create_bars_plot(plot_name)
        else:
            return self._create_curve_plot(plot_name)

    def _create_curve_plot(self, title: str) -> hv.DynamicMap:
        """Create a curve plot with random data."""

        def create_curve(frequency):
            x = np.linspace(0, 10, 200)
            y = np.sin(frequency * x) + 0.1 * np.random.randn(200)
            return hv.Curve((x, y), kdims=['x'], vdims=['y']).opts(
                title=title, width=400, height=300, tools=['hover']
            )

        return hv.DynamicMap(create_curve, kdims=['frequency']).redim.range(
            frequency=(0.5, 5.0)
        )

    def _create_scatter_plot(self, title: str) -> hv.DynamicMap:
        """Create a scatter plot with random data."""

        def create_scatter(n_points):
            x = np.random.randn(int(n_points))
            y = np.random.randn(int(n_points))
            return hv.Scatter((x, y), kdims=['x'], vdims=['y']).opts(
                title=title, width=400, height=300, size=5, tools=['hover']
            )

        return hv.DynamicMap(create_scatter, kdims=['n_points']).redim.range(
            n_points=(10, 200)
        )

    def _create_heatmap_plot(self, title: str) -> hv.DynamicMap:
        """Create a heatmap plot with random data."""

        def create_heatmap(scale):
            data = scale * np.random.randn(20, 20)
            return hv.Image(data).opts(
                title=title,
                width=400,
                height=300,
                colorbar=True,
                cmap='viridis',
                tools=['hover'],
            )

        return hv.DynamicMap(create_heatmap, kdims=['scale']).redim.range(
            scale=(0.1, 2.0)
        )

    def _create_bars_plot(self, title: str) -> hv.DynamicMap:
        """Create a bar plot with random data."""

        def create_bars(n_bars):
            categories = [f'Cat {i}' for i in range(int(n_bars))]
            values = np.random.randint(1, 100, int(n_bars))
            bars = hv.Bars((categories, values), kdims=['category'], vdims=['value'])
            return bars.opts(
                title=title, width=400, height=300, xrotation=45, tools=['hover']
            )

        return hv.DynamicMap(create_bars, kdims=['n_bars']).redim.range(n_bars=(3, 10))

    def panel(self) -> pn.viewable.Viewable:
        """Get the Panel layout for this demo."""
        return pn.template.FastListTemplate(
            title='PlotGrid Demo',
            sidebar=[
                self._instructions,
                pn.Card(
                    self._plot_type_selector,
                    title='Plot Configuration',
                    collapsed=False,
                ),
                pn.Card(
                    pn.Column(
                        self._nrows_input,
                        self._ncols_input,
                        self._recreate_button,
                    ),
                    title='Grid Configuration',
                    collapsed=True,
                ),
            ],
            main=[
                self._grid.panel,
            ],
        )


# Create and serve the demo
demo = PlotGridDemo()
demo.panel().servable()
