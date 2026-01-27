# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""
SessionPlotManager - Manages per-session plot components.

This module provides session-specific plot management, separating:
- Shared state (PlotDataService stores plotter references)
- Session state (per-session Pipes, Presenters, DynamicMaps)

Each browser session creates its own SessionPlotManager instance via the
SessionUpdater. The manager creates session-bound HoloViews components
in the correct session context.

The manager always returns something displayable for any layer:
- hv.Text placeholder for initializing/waiting/error/stopped states
- hv.DynamicMap when data is available

This allows PlotGridTabs to be a simple grid display without knowledge
of data readiness or state transitions.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import holoviews as hv

from .plot_data_service import LayerId, PlotDataService

if TYPE_CHECKING:
    from .plots import Presenter

logger = logging.getLogger(__name__)


class SessionPlotManager:
    """
    Manages per-session Pipes, Presenters, and DynamicMaps.

    Each browser session gets its own SessionPlotManager instance. The manager:
    1. Always returns something displayable for any layer (placeholder or real plot)
    2. Creates session-bound Pipes and DynamicMaps when data becomes available
    3. Forwards data updates from presenters to session Pipes

    Parameters
    ----------
    plot_data_service:
        Shared service storing plot layer state.
    """

    def __init__(self, plot_data_service: PlotDataService) -> None:
        self._plot_data_service = plot_data_service
        self._presenters: dict[LayerId, Presenter] = {}
        self._pipes: dict[LayerId, hv.streams.Pipe] = {}
        self._dmaps: dict[LayerId, hv.DynamicMap | hv.Element] = {}
        # Layers that returned placeholder, waiting for data to become available
        self._pending_layers: set[LayerId] = set()

    def get_dmap(self, layer_id: LayerId) -> hv.DynamicMap | hv.Element | None:
        """
        Get existing DynamicMap or Element for a layer, or None if not yet created.

        Use `setup_layer` to create DynamicMaps/Elements for new layers.

        Parameters
        ----------
        layer_id:
            Layer ID to get DynamicMap/Element for.

        Returns
        -------
        :
            The session's DynamicMap or Element for this layer, or None if not set up.
        """
        return self._dmaps.get(layer_id)

    def _is_setup(self, layer_id: LayerId) -> bool:
        """Check if a layer has a real DynamicMap (not just placeholder)."""
        return layer_id in self._dmaps

    def get_or_create_layer(self, layer_id: LayerId) -> hv.DynamicMap | hv.Element:
        """
        Get or create a displayable element for a layer.

        Always returns something displayable:
        - hv.Text placeholder for initializing/waiting/error/stopped states
        - hv.DynamicMap when data is available

        When data becomes available (plotter ready with cached state), creates
        session-bound Pipe and DynamicMap using the Presenter pattern.

        Layers that return placeholders are tracked as "pending" - update_pipes()
        will detect when they become ready and signal for widget rebuilds.

        Parameters
        ----------
        layer_id:
            Layer ID to get or create.

        Returns
        -------
        :
            A displayable HoloViews element (always succeeds).
        """
        # Already set up with real DynamicMap - return it
        if layer_id in self._dmaps:
            return self._dmaps[layer_id]

        # Get state from shared service
        state = self._plot_data_service.get(layer_id)

        # No state yet - layer just created
        if state is None:
            self._pending_layers.add(layer_id)
            return hv.Text(0, 0, "Initializing...")

        # Error state - show error message
        if state.error:
            self._pending_layers.discard(layer_id)
            short_error = (
                state.error[:200] + "..." if len(state.error) > 200 else state.error
            )
            return hv.Text(0, 0, f"Error:\n{short_error}")

        # Stopped state - show stopped message
        if state.stopped:
            self._pending_layers.discard(layer_id)
            return hv.Text(0, 0, "Workflow stopped")

        # Waiting for plotter or data
        if state.plotter is None or not state.plotter.has_cached_state():
            self._pending_layers.add(layer_id)
            return hv.Text(0, 0, "Waiting for data...")

        # Data available - create real DynamicMap
        self._pending_layers.discard(layer_id)
        try:
            presenter = state.plotter.create_presenter()
            pipe = hv.streams.Pipe(data=state.plotter.get_cached_state())
            dmap = presenter.present(pipe)

            self._presenters[layer_id] = presenter
            self._pipes[layer_id] = pipe
            self._dmaps[layer_id] = dmap

            logger.debug("Created session DynamicMap for layer_id=%s", layer_id)
            return dmap

        except Exception:
            logger.exception(
                "Failed to create session DynamicMap for layer_id=%s", layer_id
            )
            return hv.Text(0, 0, "Failed to create plot")

    def invalidate_layer(self, layer_id: LayerId) -> None:
        """
        Remove cached components for a layer.

        Called when a layer is no longer in PlotDataService (orphaned).

        Parameters
        ----------
        layer_id:
            Layer ID to invalidate.
        """
        self._presenters.pop(layer_id, None)
        self._pipes.pop(layer_id, None)
        self._dmaps.pop(layer_id, None)
        self._pending_layers.discard(layer_id)
        logger.debug("Invalidated session cache for layer_id=%s", layer_id)

    def update_pipes(self) -> set[LayerId]:
        """
        Poll presenters for pending updates and forward to session Pipes.

        Checks each presenter's dirty flag and sends new state to the
        corresponding session Pipe. This is how plot updates propagate
        to each browser session.

        Also checks pending layers to see if they became ready (data available).
        Returns layer IDs that transitioned from placeholder to ready state -
        these need widget rebuilds to display the real DynamicMap.

        Also cleans up orphaned layers - when update_layer_config() replaces
        a layer with a new layer_id, the old layer_id is removed from
        PlotDataService and becomes orphaned in the session cache.

        Returns
        -------
        :
            Layer IDs that transitioned from placeholder to ready state.
        """
        transitioned: set[LayerId] = set()

        # Clean up orphaned layers no longer in PlotDataService.
        for layer_id in list(self._dmaps.keys()):
            if self._plot_data_service.get(layer_id) is None:
                self.invalidate_layer(layer_id)

        # Also clean up pending layers that were removed
        for layer_id in list(self._pending_layers):
            if self._plot_data_service.get(layer_id) is None:
                self._pending_layers.discard(layer_id)

        # Check pending layers for transitions (data became available)
        for layer_id in list(self._pending_layers):
            state = self._plot_data_service.get(layer_id)
            if (
                state is not None
                and state.plotter is not None
                and state.plotter.has_cached_state()
                and not state.error
                and not state.stopped
            ):
                transitioned.add(layer_id)
                logger.debug("Layer %s transitioned to ready state", layer_id)

        # Forward data updates to existing pipes
        for layer_id, presenter in list(self._presenters.items()):
            if not presenter.has_pending_update():
                continue

            try:
                session_pipe = self._pipes.get(layer_id)
                if session_pipe is not None:
                    session_pipe.send(presenter.consume_update())
                    logger.debug(
                        "Forwarded state update to session pipe for layer_id=%s",
                        layer_id,
                    )
            except Exception:
                logger.exception(
                    "Failed to forward state to session pipe for layer_id=%s",
                    layer_id,
                )

        return transitioned

    def get_tracked_layer_ids(self) -> set[LayerId]:
        """Get all layer IDs tracked by this session."""
        return set(self._dmaps.keys())

    def cleanup(self) -> None:
        """Clean up session resources."""
        self._presenters.clear()
        self._pipes.clear()
        self._dmaps.clear()
        self._pending_layers.clear()
        logger.debug("SessionPlotManager cleaned up")
