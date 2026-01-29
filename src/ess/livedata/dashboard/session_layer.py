# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""
SessionLayer - Per-session state for a single plot layer.

Each browser session creates SessionLayer instances to hold session-bound
HoloViews components (Pipe, DynamicMap, Presenter). This replaces the
separate SessionPlotManager class with a simpler, self-contained unit.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import holoviews as hv

from .plot_data_service import LayerId, LayerStateMachine

if TYPE_CHECKING:
    from .plots import Plotter, Presenter


@dataclass
class SessionLayer:
    """
    Per-session state for a single plot layer.

    Holds session-bound HoloViews components and tracks the last seen version
    from PlotDataService for change detection.

    Parameters
    ----------
    layer_id:
        The layer's unique identifier.
    presenter:
        Per-session presenter created from the plotter.
    pipe:
        Session-local HoloViews Pipe for data updates.
    dmap:
        The DynamicMap or Element created by the presenter.
    last_seen_version:
        Version from PlotDataService when this was created/updated.
    """

    layer_id: LayerId
    presenter: Presenter
    pipe: hv.streams.Pipe
    dmap: hv.DynamicMap | hv.Element
    last_seen_version: int

    def update_pipe(self) -> bool:
        """
        Push pending update to pipe if available.

        Returns
        -------
        :
            True if an update was sent, False if no pending update.
        """
        if not self.presenter.has_pending_update():
            return False
        self.pipe.send(self.presenter.consume_update())
        return True

    def is_valid_for(self, plotter: Plotter | None) -> bool:
        """
        Check if this session layer is still valid for the given plotter.

        Returns False if the plotter has been replaced (e.g., workflow restart),
        indicating the session components should be recreated.

        Parameters
        ----------
        plotter:
            The current plotter from PlotDataService, or None if layer removed.

        Returns
        -------
        :
            True if still valid, False if plotter changed or was removed.
        """
        return plotter is not None and self.presenter.is_owned_by(plotter)

    @classmethod
    def create(cls, layer_id: LayerId, state: LayerStateMachine) -> SessionLayer | None:
        """
        Create a SessionLayer if data is available.

        Parameters
        ----------
        layer_id:
            The layer's unique identifier.
        state:
            Layer state from PlotDataService.

        Returns
        -------
        :
            A new SessionLayer, or None if no displayable plot yet.
        """
        if not state.has_displayable_plot():
            return None

        plotter: Any = state.plotter  # Avoid circular import issues with type
        presenter = plotter.create_presenter()
        pipe = hv.streams.Pipe(data=plotter.get_cached_state())
        dmap = presenter.present(pipe)

        return cls(
            layer_id=layer_id,
            presenter=presenter,
            pipe=pipe,
            dmap=dmap,
            last_seen_version=state.version,
        )
