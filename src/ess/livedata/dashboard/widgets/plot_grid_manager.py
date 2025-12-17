# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""
PlotGridManager - Widget for managing plot grid configurations.

Provides UI for adding and removing plot grids through PlotOrchestrator.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from io import StringIO

import panel as pn
import yaml

from ...config.grid_template import GridSpec
from ...config.workflow_spec import WorkflowId, WorkflowSpec
from ..plot_orchestrator import (
    GridId,
    PlotCell,
    PlotGridConfig,
    PlotOrchestrator,
    SubscriptionId,
)
from .plot_widgets import (
    create_close_button,
    create_download_button,
    get_workflow_display_info,
)

# Sentinel value for "no template selected" in the dropdown
_NO_TEMPLATE = "-- No template --"

# Colors for template preview cells (cycle through these)
_CELL_COLORS = [
    '#e3f2fd',  # light blue
    '#f3e5f5',  # light purple
    '#e8f5e9',  # light green
    '#fff3e0',  # light orange
    '#fce4ec',  # light pink
    '#e0f7fa',  # light cyan
]


class GridRow:
    """
    Widget row for a single grid in the grid list.

    Displays grid info and action buttons (download, remove).

    Parameters
    ----------
    grid_id
        ID of the grid this row represents.
    grid_config
        Configuration of the grid (title, dimensions).
    on_remove
        Callback to invoke when the remove button is clicked.
    get_yaml_content
        Callback that returns the YAML content for download.
    """

    def __init__(
        self,
        grid_id: GridId,
        grid_config: PlotGridConfig,
        *,
        on_remove: Callable[[], None],
        get_yaml_content: Callable[[], StringIO],
    ) -> None:
        self._grid_id = grid_id
        self._grid_config = grid_config

        # Grid info label
        label = pn.pane.Str(
            f'{grid_config.title} ({grid_config.nrows}x{grid_config.ncols})',
            styles={'flex-grow': '1'},
        )

        # Download button - generates YAML when clicked
        download_button = create_download_button(
            filename=f'{_sanitize_filename(grid_config.title)}.yaml',
            callback=get_yaml_content,
        )

        # Remove button
        remove_button = create_close_button(on_remove)

        self._widget = pn.Row(
            label,
            download_button,
            remove_button,
            sizing_mode='stretch_width',
        )

    @property
    def panel(self) -> pn.Row:
        """Get the Panel viewable object for this row."""
        return self._widget


def _sanitize_filename(title: str) -> str:
    """
    Sanitize a title for use as a filename.

    Replaces spaces with underscores and removes/replaces problematic characters.
    """
    # Replace spaces with underscores
    result = title.replace(' ', '_')
    # Remove characters that are problematic in filenames
    invalid_chars = '<>:"/\\|?*'
    for char in invalid_chars:
        result = result.replace(char, '')
    # Lowercase for consistency
    return result.lower()


class PlotGridManager:
    """
    Widget for managing plot grid configurations.

    Provides a form to add new grids and a list of existing grids with
    remove buttons. Automatically updates when grids are added or removed
    through the orchestrator.

    Parameters
    ----------
    orchestrator
        The orchestrator managing plot grid configurations. Templates are
        retrieved from the orchestrator via get_available_templates().
    workflow_registry
        Registry of available workflows and their specifications. Used to
        look up workflow titles for display in the grid preview.
    """

    def __init__(
        self,
        orchestrator: PlotOrchestrator,
        workflow_registry: Mapping[WorkflowId, WorkflowSpec],
    ) -> None:
        self._orchestrator = orchestrator
        self._workflow_registry = workflow_registry
        templates = orchestrator.get_available_templates()
        self._templates = {t.name: t for t in templates}
        self._selected_template = None
        # Flag to suppress redundant preview updates during batch operations
        self._suppress_preview_update = False

        # Template selector (only shown if templates available)
        template_options = [_NO_TEMPLATE, *self._templates.keys()]
        self._template_selector = pn.widgets.Select(
            name='Template',
            options=template_options,
            value=_NO_TEMPLATE,
            visible=bool(templates),
        )
        self._template_selector.param.watch(self._on_template_selected, 'value')

        # Input fields for new grid
        self._title_input = pn.widgets.TextInput(
            name='Grid Title', value='New Grid', placeholder='Enter grid title'
        )
        self._nrows_input = pn.widgets.IntInput(name='Rows', value=3, start=2, end=6)
        self._ncols_input = pn.widgets.IntInput(name='Columns', value=3, start=2, end=6)

        # Watch for rows/cols changes to update preview
        self._nrows_input.param.watch(self._on_grid_size_changed, 'value')
        self._ncols_input.param.watch(self._on_grid_size_changed, 'value')

        # Add grid button
        self._add_button = pn.widgets.Button(name='Add Grid', button_type='primary')
        self._add_button.on_click(self._on_add_grid)

        # Grid preview container (always shown)
        # Fixed dimensions prevent layout jumps when content is replaced
        self._grid_preview = pn.Column(
            sizing_mode='fixed',
            width=424,  # preview_width (400) + padding (24)
            height=264,  # preview_height (240) + padding (24)
            margin=(0, 0, 0, 20),
        )

        # Grid list container
        # IMPORTANT: Use stretch_both (not stretch_width) to ensure consistent
        # sizing behavior with PlotGrid tabs. Panel's Tabs widget handles dynamic
        # tab addition better when all tabs have the same sizing mode.
        self._grid_list = pn.Column(sizing_mode='stretch_both')

        # Subscribe to orchestrator updates
        self._subscription_id: SubscriptionId | None = (
            self._orchestrator.subscribe_to_lifecycle(
                on_grid_created=self._on_grid_created,
                on_grid_removed=self._on_grid_removed,
            )
        )

        # Initialize grid list and preview
        self._update_grid_list()
        self._update_preview()

        # Form column (left side) - fixed width
        form_column = pn.Column(
            self._template_selector,
            self._title_input,
            self._nrows_input,
            self._ncols_input,
            self._add_button,
            width=200,
            sizing_mode='fixed',
        )

        # Row containing form and preview side by side
        # Spacer stretches to push preview to the right
        form_row = pn.Row(
            form_column,
            pn.Spacer(sizing_mode='stretch_width'),
            self._grid_preview,
            sizing_mode='stretch_width',
            align='start',
        )

        # Main widget layout
        # IMPORTANT: Use stretch_both to match PlotGrid tab sizing. This ensures
        # Panel's Tabs widget properly handles height when tabs are added/removed.
        self._widget = pn.Column(
            pn.pane.Markdown('## Add New Grid'),
            form_row,
            pn.layout.Divider(),
            pn.pane.Markdown('## Existing Grids'),
            self._grid_list,
            sizing_mode='stretch_both',
        )

    def _reset_to_defaults(self) -> None:
        """Reset form inputs to default state (no template selected)."""
        self._selected_template = None
        self._title_input.value = 'New Grid'
        self._nrows_input.value = 3
        self._ncols_input.value = 3
        self._nrows_input.start = 2
        self._ncols_input.start = 2

    def _update_preview(self) -> None:
        """Update the grid preview based on current state."""
        preview = self._create_grid_preview(
            nrows=self._nrows_input.value,
            ncols=self._ncols_input.value,
            template=self._selected_template,
        )
        # Atomic replacement avoids layout jumps from separate clear+append
        self._grid_preview.objects = [preview]

    def _create_grid_preview(
        self,
        nrows: int,
        ncols: int,
        template: GridSpec | None,
    ) -> pn.Column:
        """
        Create a visual preview of the grid layout.

        Shows a mini grid with colored boxes representing each cell,
        labeled with the workflow name if a template is selected.

        Parameters
        ----------
        nrows
            Number of rows in the grid.
        ncols
            Number of columns in the grid.
        template
            Optional template to show cells from.

        Returns
        -------
        :
            Panel Column containing the preview.
        """
        cells: Sequence[PlotCell] = template.cells if template else ()

        # Fixed preview size - cells scale to fit
        preview_width = 400
        preview_height = 240

        # Create grid spec with fixed size
        grid = pn.GridSpec(
            width=preview_width,
            height=preview_height,
            sizing_mode='fixed',
        )

        # Calculate which cells are covered by template cells
        covered_cells: set[tuple[int, int]] = set()
        for cell in cells:
            geometry = cell.geometry
            row_start = geometry.row
            row_end = row_start + geometry.row_span
            col_start = geometry.col
            col_end = col_start + geometry.col_span
            # Only count cells that fit in current grid size
            if row_end <= nrows and col_end <= ncols:
                for r in range(row_start, row_end):
                    for c in range(col_start, col_end):
                        covered_cells.add((r, c))

        # Add empty cells as background (light gray border) only for uncovered cells
        for row in range(nrows):
            for col in range(ncols):
                if (row, col) not in covered_cells:
                    grid[row, col] = pn.pane.HTML(
                        '',
                        styles={
                            'background-color': '#f5f5f5',
                            'border': '1px dashed #ccc',
                            'box-sizing': 'border-box',
                        },
                        sizing_mode='stretch_both',
                        margin=1,
                    )

        # Add template cells
        for i, cell in enumerate(cells):
            # Check if cell fits in current grid size
            geometry = cell.geometry
            row_start = geometry.row
            row_end = row_start + geometry.row_span
            col_start = geometry.col
            col_end = col_start + geometry.col_span

            if row_end > nrows or col_end > ncols:
                continue  # Cell doesn't fit, skip it

            # Look up workflow title from the first layer's config
            first_layer = cell.layers[0]
            config = first_layer.config
            workflow_title, _ = get_workflow_display_info(
                self._workflow_registry, config.workflow_id, config.output_name
            )

            # Truncate long titles for the compact preview
            if len(workflow_title) > 20:
                workflow_title = workflow_title[:17] + '...'

            output_name = config.output_name or ''
            layer_count = len(cell.layers)

            color = _CELL_COLORS[i % len(_CELL_COLORS)]

            layer_info = f' (+{layer_count - 1})' if layer_count > 1 else ''
            label_html = (
                f'<div style="font-size: 10px; font-weight: 500;">'
                f'{workflow_title}{layer_info}</div>'
                f'<div style="font-size: 9px; color: #666;">{output_name}</div>'
            )
            grid[row_start:row_end, col_start:col_end] = pn.pane.HTML(
                label_html,
                styles={
                    'background-color': color,
                    'border': '2px solid #1976d2',
                    'border-radius': '4px',
                    'display': 'flex',
                    'flex-direction': 'column',
                    'align-items': 'center',
                    'justify-content': 'center',
                    'text-align': 'center',
                    'box-sizing': 'border-box',
                },
                sizing_mode='stretch_both',
                margin=1,
            )

        return pn.Column(
            grid,
            width=preview_width + 24,
            styles={
                'background-color': '#fafafa',
                'border': '1px solid #e0e0e0',
                'border-radius': '4px',
                'padding': '10px',
            },
        )

    def _on_grid_size_changed(self, event) -> None:
        """Handle rows/cols input change."""
        # Skip if we're in a batch operation that will update preview at the end
        if not self._suppress_preview_update:
            self._update_preview()

    def _on_template_selected(self, event) -> None:
        """Handle template selection change."""
        template_name = event.new
        # Batch widget updates and suppress redundant preview updates
        self._suppress_preview_update = True
        try:
            with pn.io.hold():
                if template_name == _NO_TEMPLATE:
                    self._reset_to_defaults()
                else:
                    template = self._templates[template_name]
                    self._selected_template = template
                    # Populate widgets with template values
                    self._title_input.value = template.title
                    self._nrows_input.value = template.nrows
                    self._ncols_input.value = template.ncols
                    # Set minimum to prevent shrinking below what cells require
                    self._nrows_input.start = template.min_rows
                    self._ncols_input.start = template.min_cols

                self._update_preview()
        finally:
            self._suppress_preview_update = False

    def _on_add_grid(self, event) -> None:
        """Handle add grid button click."""
        grid_id = self._orchestrator.add_grid(
            title=self._title_input.value,
            nrows=self._nrows_input.value,
            ncols=self._ncols_input.value,
        )

        # Add template cells if a template is selected
        if self._selected_template:
            for cell in self._selected_template.cells:
                cell_id = self._orchestrator.add_cell(grid_id, cell.geometry)
                for layer in cell.layers:
                    self._orchestrator.add_layer(cell_id, layer.config)

        # Reset inputs and template selection
        # Batch widget updates and suppress redundant preview updates
        self._suppress_preview_update = True
        try:
            with pn.io.hold():
                self._template_selector.value = _NO_TEMPLATE
                self._reset_to_defaults()
                self._update_preview()
        finally:
            self._suppress_preview_update = False

    def _on_grid_created(self, grid_id: GridId, grid_config: PlotGridConfig) -> None:
        """Handle grid creation from orchestrator."""
        self._update_grid_list()

    def _on_grid_removed(self, grid_id: GridId) -> None:
        """Handle grid removal from orchestrator."""
        self._update_grid_list()

    def _update_grid_list(self) -> None:
        """Update the grid list display."""
        self._grid_list.clear()
        for grid_id, grid_config in self._orchestrator.get_all_grids().items():
            row = GridRow(
                grid_id=grid_id,
                grid_config=grid_config,
                on_remove=self._make_remove_handler(grid_id),
                get_yaml_content=self._make_yaml_callback(grid_id),
            )
            self._grid_list.append(row.panel)

    def _make_remove_handler(self, grid_id: GridId) -> Callable[[], None]:
        """Create a closure that removes the given grid."""

        def handler() -> None:
            self._orchestrator.remove_grid(grid_id)

        return handler

    def _make_yaml_callback(self, grid_id: GridId) -> Callable[[], StringIO]:
        """Create a closure that generates YAML content for the given grid."""

        def callback() -> StringIO:
            grid_data = self._orchestrator.serialize_grid(grid_id)
            sio = StringIO()
            yaml.dump(grid_data, sio, default_flow_style=False, sort_keys=False)
            sio.seek(0)
            return sio

        return callback

    def shutdown(self) -> None:
        """Unsubscribe from lifecycle events."""
        if self._subscription_id is not None:
            self._orchestrator.unsubscribe_from_lifecycle(self._subscription_id)
            self._subscription_id = None

    @property
    def panel(self) -> pn.Column:
        """Get the Panel viewable object for this widget."""
        return self._widget
