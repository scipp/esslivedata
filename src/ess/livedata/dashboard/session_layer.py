# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""
SessionLayer - Per-session state for a single plot layer.

Each browser session creates SessionLayer instances to track layer versions
and hold session-bound HoloViews components (Pipe, DynamicMap, Presenter)
when data is available.
"""

from __future__ import annotations

from dataclasses import dataclass

import holoviews as hv

from .plot_data_service import LayerId, LayerStateMachine
from .plots import Plotter, Presenter


@dataclass
class SessionComponents:
    """
    Session-bound HoloViews components for rendering a layer.

    These components are created together when a layer has displayable data
    and are always used as a unit.

    Parameters
    ----------
    presenter:
        Per-session presenter created from the plotter.
    pipe:
        Session-local HoloViews Pipe for data updates.
    dmap:
        The DynamicMap or Element created by the presenter.
    """

    presenter: Presenter
    pipe: hv.streams.Pipe
    dmap: hv.DynamicMap | hv.Element

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
        Check if these components are still valid for the given plotter.

        Returns False if the plotter has been replaced (e.g., workflow restart),
        indicating the components should be recreated.
        """
        return plotter is not None and self.presenter.is_owned_by(plotter)

    @classmethod
    def create(cls, state: LayerStateMachine) -> SessionComponents | None:
        """
        Create session components if data is available.

        Parameters
        ----------
        state:
            Layer state from PlotDataService.

        Returns
        -------
        :
            New components, or None if no displayable plot yet.
        """
        if not state.has_displayable_plot():
            return None

        plotter = state.plotter
        if plotter is None:
            raise ValueError("Plotter must not be None when plot is displayable")
        presenter = plotter.create_presenter()
        pipe = hv.streams.Pipe(data=plotter.get_cached_state())
        dmap = presenter.present(pipe)

        return cls(presenter=presenter, pipe=pipe, dmap=dmap)


@dataclass
class SessionLayer:
    """
    Per-session state for a single plot layer.

    Tracks the last seen version for change detection. Optionally holds
    session-bound rendering components when data is available.

    Parameters
    ----------
    layer_id:
        The layer's unique identifier.
    last_seen_version:
        Version from PlotDataService when this was last seen.
    components:
        Session-bound rendering components, or None if no displayable data yet.
    """

    layer_id: LayerId
    last_seen_version: int
    components: SessionComponents | None = None

    @property
    def dmap(self) -> hv.DynamicMap | hv.Element | None:
        """The DynamicMap or Element for rendering, or None if not available."""
        return self.components.dmap if self.components else None

    def update_pipe(self) -> bool:
        """
        Push pending update to pipe if components are available.

        Returns
        -------
        :
            True if an update was sent, False otherwise.
        """
        if self.components is None:
            return False
        return self.components.update_pipe()

    def ensure_components(self, state: LayerStateMachine) -> bool:
        """
        Ensure components exist if data is now available.

        Creates components if they don't exist and data is displayable.
        Invalidates components if plotter has changed.

        Parameters
        ----------
        state:
            Current layer state from PlotDataService.

        Returns
        -------
        :
            True if components exist after this call, False otherwise.
        """
        # Check if existing components are still valid
        if self.components is not None:
            if not self.components.is_valid_for(state.plotter):
                self.components = None
            else:
                return True

        # Try to create components
        self.components = SessionComponents.create(state)
        return self.components is not None
