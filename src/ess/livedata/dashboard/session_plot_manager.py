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

import holoviews as hv

from .plot_data_service import LayerId, PlotDataService
from .plots import Presenter


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
        Get DynamicMap or Element for a layer, attempting setup if not yet created.

        Automatically calls `setup_layer` if the layer hasn't been set up yet.
        This eliminates the "flash to placeholder" when layer data is already
        available (e.g., static overlays or when workflow is already running).

        Parameters
        ----------
        layer_id:
            Layer ID to get DynamicMap/Element for.

        Returns
        -------
        :
            The session's DynamicMap or Element for this layer, or None if
            data is not yet available.
        """
        if layer_id not in self._dmaps:
            self._setup_layer(layer_id)
        return self._dmaps.get(layer_id)

    def has_layer(self, layer_id: LayerId) -> bool:
        """Check if a layer has been set up for this session."""
        return layer_id in self._dmaps

    def _setup_layer(self, layer_id: LayerId) -> hv.DynamicMap | hv.Element | None:
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
            presenter = state.plotter.create_presenter()
            pipe = hv.streams.Pipe(data=state.plotter.get_cached_state())
            dmap = presenter.present(pipe)
            self._presenters[layer_id] = presenter
            self._pipes[layer_id] = pipe
            self._dmaps[layer_id] = dmap
            return dmap

        except Exception:
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

            # Consume update from presenter and send to session pipe
            session_pipe = self._pipes.get(layer_id)
            if session_pipe is not None:
                session_pipe.send(presenter.consume_update())
                updated_layers.add(layer_id)

        return updated_layers

    def get_tracked_layer_ids(self) -> set[LayerId]:
        """Get all layer IDs tracked by this session."""
        return set(self._dmaps.keys())

    def cleanup(self) -> None:
        """Clean up session resources."""
        self._presenters.clear()
        self._pipes.clear()
        self._dmaps.clear()
