# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""
SessionPlotManager - Manages per-session plot components.

This module extracts session-specific plot management from PlotGridTabs,
providing a clean separation between:
- Shared state (PlotDataService stores plotter references)
- Session state (per-session Pipes, Presenters, DynamicMaps)

Each browser session creates its own SessionPlotManager instance via the
SessionUpdater. The manager creates session-bound HoloViews components
in the correct session context.
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
    1. Tracks which layers have been set up for this session
    2. Creates session-bound Pipes and DynamicMaps on demand
    3. Polls presenters for pending updates and forwards to session Pipes

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

    def has_layer(self, layer_id: LayerId) -> bool:
        """Check if a layer has been set up for this session."""
        return layer_id in self._dmaps

    def setup_layer(self, layer_id: LayerId) -> hv.DynamicMap | hv.Element | None:
        """
        Set up a layer for this session if data is available.

        Creates session-bound Pipe and DynamicMap/Element using the Presenter pattern.
        Returns immediately if the layer is already set up.

        Parameters
        ----------
        layer_id:
            Layer ID to set up.

        Returns
        -------
        :
            The newly created DynamicMap or Element, or None if no data available yet.
        """
        # Already set up
        if layer_id in self._dmaps:
            return self._dmaps[layer_id]

        # Get data from shared service
        state = self._plot_data_service.get(layer_id)
        # Need plotter (for create_presenter and get_cached_state) to set up.
        # plotter.has_cached_state() returns False if waiting for data.
        if state is None or state.plotter is None:
            return None
        if not state.plotter.has_cached_state():
            return None

        try:
            # Use Presenter pattern - plotter creates appropriate presenter
            presenter = state.plotter.create_presenter()

            # Create session-bound Pipe with initial state from plotter's cache
            # (either pre-computed HoloViews elements or raw data for kdims plotters)
            pipe = hv.streams.Pipe(data=state.plotter.get_cached_state())
            dmap = presenter.present(pipe)

            self._presenters[layer_id] = presenter
            self._pipes[layer_id] = pipe
            self._dmaps[layer_id] = dmap

            logger.debug(
                "Created session DynamicMap for layer_id=%s",
                layer_id,
            )

            return dmap

        except Exception:
            logger.exception(
                "Failed to create session DynamicMap for layer_id=%s",
                layer_id,
            )
            return None

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
        logger.debug("Invalidated session cache for layer_id=%s", layer_id)

    def update_pipes(self) -> set[LayerId]:
        """
        Poll presenters for pending updates and forward to session Pipes.

        Checks each presenter's dirty flag and sends new state to the
        corresponding session Pipe. This is how plot updates propagate
        to each browser session.

        Also cleans up orphaned layers - when update_layer_config() replaces
        a layer with a new layer_id, the old layer_id is removed from
        PlotDataService and becomes orphaned in the session cache.

        Returns
        -------
        :
            Set of layer IDs that received updates.
        """
        updated_layers: set[LayerId] = set()

        # Clean up orphaned layers no longer in PlotDataService.
        # When update_layer_config() creates a new layer_id, the old one
        # is removed from PlotDataService. We detect and clean up here.
        for layer_id in list(self._dmaps.keys()):
            if self._plot_data_service.get(layer_id) is None:
                self.invalidate_layer(layer_id)

        for layer_id, presenter in list(self._presenters.items()):
            if not presenter.has_pending_update():
                continue

            try:
                # Consume update from presenter and send to session pipe
                session_pipe = self._pipes.get(layer_id)
                if session_pipe is not None:
                    session_pipe.send(presenter.consume_update())
                    updated_layers.add(layer_id)
                    logger.debug(
                        "Forwarded state update to session pipe for layer_id=%s",
                        layer_id,
                    )
            except Exception:
                logger.exception(
                    "Failed to forward state to session pipe for layer_id=%s",
                    layer_id,
                )

        return updated_layers

    def get_tracked_layer_ids(self) -> set[LayerId]:
        """Get all layer IDs tracked by this session."""
        return set(self._dmaps.keys())

    def cleanup(self) -> None:
        """Clean up session resources."""
        self._presenters.clear()
        self._pipes.clear()
        self._dmaps.clear()
        logger.debug("SessionPlotManager cleaned up")
