# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""
Cell-properties modal: rename a cell and edit its layer composition.

Opened from the cell titlebar's edit (pencil) button. Rename is the only field
saved on "Save"; add/remove act immediately, so the modal doubles as the cell's
layer-composition editor:

- a cell-level "Add layer…" button hands off to the plot config modal (a new,
  independent layer);
- each layer row has a ``+`` offering overlays derived from that layer (added in
  place, keeping the modal open) and an ``x`` to remove it.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping

import panel as pn

from ess.livedata.config.workflow_spec import WorkflowId, WorkflowSpec

from ..data_roles import PRIMARY
from ..plot_orchestrator import (
    CellId,
    DataSourceConfig,
    LayerId,
    PlotConfig,
    PlotOrchestrator,
)
from ..plotting_controller import PlottingController
from .buttons import ButtonStyles, create_tool_button
from .plot_widgets import (
    create_overlay_add_button,
    get_plot_cell_display_info,
    overlay_suggestions_for_layer,
)


class CellPropertiesModal:
    """
    Modal for renaming a cell and editing its layers.

    Builds a ``pn.Modal`` (exposed as :attr:`modal`) for the owner to place in
    its shared modal container and :meth:`show`. The two outward interactions
    are passed as callbacks: ``on_add_layer`` hands the "Add layer…" flow back
    to the owner (which opens the plot config modal in the shared container),
    and ``on_close`` tears down the container on Save/Cancel/remove-last.

    Parameters
    ----------
    orchestrator:
        Orchestrator owning the cell; mutated directly by add/remove/rename.
    workflow_registry:
        Registry mapping workflow IDs to specs, for layer display titles.
    plotting_controller:
        Controller providing overlay suggestions and plotter specs.
    cell_id:
        ID of the cell to edit.
    current_title:
        The currently displayed title (user-defined or derived).
    has_user_title:
        Whether ``current_title`` is user-defined; if not, the input starts
        empty so the placeholder hints at the derived fallback.
    on_add_layer:
        Invoked with the cell ID to open the add-layer config modal. The
        properties modal closes itself first (the owner reuses the container).
    on_close:
        Invoked to tear down the shared container after the modal closes.
    """

    def __init__(
        self,
        *,
        orchestrator: PlotOrchestrator,
        workflow_registry: Mapping[WorkflowId, WorkflowSpec],
        plotting_controller: PlottingController,
        cell_id: CellId,
        current_title: str,
        has_user_title: bool,
        on_add_layer: Callable[[CellId], None],
        on_close: Callable[[], None],
    ) -> None:
        self._orchestrator = orchestrator
        self._workflow_registry = workflow_registry
        self._plotting_controller = plotting_controller
        self._cell_id = cell_id
        self._on_add_layer = on_add_layer
        self._on_close = on_close
        self._cell = orchestrator.get_cell(cell_id)
        self._done = False

        self._text = pn.widgets.TextInput(
            value=current_title if has_user_title else '',
            label='Cell title',
            placeholder='Leave empty to use the derived title',
            sizing_mode='stretch_width',
        )
        self._layers_col = pn.Column(sizing_mode='stretch_width')
        self.modal = self._build()
        self._render_layers()

    def show(self) -> None:
        """Open the modal."""
        self.modal.open = True

    def _build(self) -> pn.Modal:
        self._add_layer_button = pn.widgets.Button(
            name='Add layer…', button_type='default'
        )
        self._add_layer_button.on_click(lambda _: self._on_add())
        save_button = pn.widgets.Button(name='Save', button_type='primary')
        save_button.on_click(lambda _: self._finish(save=True))
        cancel_button = pn.widgets.Button(name='Cancel')
        cancel_button.on_click(lambda _: self._finish(save=False))

        modal = pn.Modal(
            pn.Column(
                pn.pane.Markdown('### Cell properties', margin=(0, 0, 5, 0)),
                pn.pane.Markdown('#### Header', margin=(0, 0, 0, 0)),
                self._text,
                pn.pane.Markdown('#### Layers', margin=(10, 0, 0, 0)),
                pn.Row(self._add_layer_button, pn.Spacer(sizing_mode='stretch_width')),
                self._layers_col,
                pn.Row(
                    pn.Spacer(sizing_mode='stretch_width'),
                    cancel_button,
                    save_button,
                    margin=(10, 0, 0, 0),
                ),
                sizing_mode='stretch_width',
            ),
            margin=20,
            width=480,
        )
        # Closing via the X button or ESC cancels.
        modal.param.watch(
            lambda event: None if event.new else self._finish(save=False), 'open'
        )
        return modal

    def _persist_title(self) -> None:
        new_title = (self._text.value_input or '').strip() or None
        self._orchestrator.set_cell_title(self._cell_id, new_title)

    def _close(self) -> None:
        self._done = True
        self.modal.open = False
        self._on_close()

    def _finish(self, *, save: bool) -> None:
        if self._done:
            return
        if save:
            self._persist_title()
        self._close()

    def _on_add(self) -> None:
        if self._done:
            return
        self._persist_title()
        # Hand off to the owner's add-layer flow, which reuses the container.
        self._done = True
        self.modal.open = False
        self._on_add_layer(self._cell_id)

    def _on_remove(self, layer_id: LayerId) -> None:
        # ``remove_layer`` mutates the shared ``cell.layers`` in place, and
        # removes the whole cell once its last layer is gone.
        self._orchestrator.remove_layer(layer_id)
        if not self._cell.layers:
            self._close()
        else:
            # Per-layer overlay suggestions depend on the remaining layers.
            self._render_layers()

    def _on_add_overlay(
        self, view_name: str, plotter_name: str, base_config: PlotConfig
    ) -> None:
        if self._done:
            return
        # Add in place and keep the modal open so several overlays can be added
        # in one session; the new layer appears as its own row.
        self._add_overlay_layer(base_config, view_name, plotter_name)
        self._render_layers()

    def _add_overlay_layer(
        self, base_config: PlotConfig, view_name: str, plotter_name: str
    ) -> None:
        spec = self._plotting_controller.get_spec(plotter_name)
        params = spec.params() if spec.params else None
        overlay_config = PlotConfig(
            data_sources={
                PRIMARY: DataSourceConfig(
                    workflow_id=base_config.workflow_id,
                    source_names=list(base_config.source_names),
                    view_name=view_name,
                )
            },
            plot_name=plotter_name,
            params=params,
        )
        self._orchestrator.add_layer(self._cell_id, overlay_config)

    def _render_layers(self) -> None:
        existing = frozenset(layer.config.plot_name for layer in self._cell.layers)
        # A non-overlayable layer (table or layout-mode plot) cannot share a cell,
        # so no further layer can join it; disable the cell-level add button. The
        # same reason hides per-layer overlay buttons below.
        has_non_overlayable = any(
            not self._plotting_controller.is_overlayable(
                layer.config.plot_name, layer.config.params
            )
            for layer in self._cell.layers
        )
        self._add_layer_button.disabled = has_non_overlayable
        self._add_layer_button.description = (
            'A table or layout-mode plot cannot share a cell with other layers; '
            'place a new layer in its own cell.'
            if has_non_overlayable
            else None
        )
        rows: list = []
        for layer in self._cell.layers:
            title, _ = get_plot_cell_display_info(
                layer.config,
                self._workflow_registry,
                get_source_title=self._orchestrator.get_source_title,
            )
            controls: list = []
            overlays = (
                overlay_suggestions_for_layer(
                    layer.config,
                    existing,
                    self._workflow_registry,
                    self._plotting_controller,
                )
                if self._plotting_controller.is_overlayable(
                    layer.config.plot_name, layer.config.params
                )
                else []
            )
            if overlays:
                controls.append(
                    create_overlay_add_button(
                        overlays=overlays,
                        on_overlay_selected=(
                            lambda v, p, b=layer.config: self._on_add_overlay(v, p, b)
                        ),
                    )
                )
            controls.append(
                create_tool_button(
                    icon_name='x',
                    button_color=ButtonStyles.DANGER_RED,
                    hover_color=ButtonStyles.DANGER_HOVER,
                    on_click_callback=lambda lid=layer.layer_id: self._on_remove(lid),
                )
            )
            title_pane = pn.pane.HTML(
                title,
                sizing_mode='stretch_width',
                align='center',
                styles={'overflow': 'hidden'},
            )
            rows.append(pn.Row(title_pane, *controls, sizing_mode='stretch_width'))
        self._layers_col[:] = rows
