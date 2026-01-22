# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""
SessionPlotManager - Manages per-session plot components.

This module extracts session-specific plot management from PlotGridTabs,
providing a clean separation between:
- Shared state (PlotDataService stores computed plot data)
- Session state (per-session Pipes, Presenters, DynamicMaps)

Each browser session creates its own SessionPlotManager instance via the
SessionUpdater. The manager creates session-bound HoloViews components
in the correct session context.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import holoviews as hv

from .state_stores import LayerId as StateLayerId
from .state_stores import PlotDataService

if TYPE_CHECKING:
    from .plot_orchestrator import LayerId
    from .plots import Presenter

logger = logging.getLogger(__name__)


class SessionPlotManager:
    """
    Manages per-session Pipes, Presenters, and DynamicMaps.

    Each browser session gets its own SessionPlotManager instance. The manager:
    1. Tracks which layers have been set up for this session
    2. Creates session-bound Pipes and DynamicMaps on demand
    3. Forwards data updates from PlotDataService to session Pipes

    Parameters
    ----------
    plot_data_service:
        Shared service storing plot data with version tracking.
    """

    def __init__(self, plot_data_service: PlotDataService) -> None:
        self._plot_data_service = plot_data_service
        self._presenters: dict[LayerId, Presenter] = {}
        self._pipes: dict[LayerId, hv.streams.Pipe] = {}
        self._dmaps: dict[LayerId, hv.DynamicMap] = {}
        self._last_versions: dict[LayerId, int] = {}
        # Track shared pipes for data forwarding (streaming updates)
        self._shared_pipes: dict[LayerId, hv.streams.Pipe] = {}
        self._last_data_ids: dict[LayerId, int] = {}

    def get_dmap(self, layer_id: LayerId) -> hv.DynamicMap | None:
        """
        Get existing DynamicMap for a layer, or None if not yet created.

        Use `setup_layer` to create DynamicMaps for new layers.

        Parameters
        ----------
        layer_id:
            Layer ID to get DynamicMap for.

        Returns
        -------
        :
            The session's DynamicMap for this layer, or None if not set up.
        """
        return self._dmaps.get(layer_id)

    def has_layer(self, layer_id: LayerId) -> bool:
        """Check if a layer has been set up for this session."""
        return layer_id in self._dmaps

    def setup_layer(self, layer_id: LayerId) -> hv.DynamicMap | None:
        """
        Set up a layer for this session if data is available.

        Creates session-bound Pipe and DynamicMap using the Presenter pattern.
        Returns immediately if the layer is already set up.

        Parameters
        ----------
        layer_id:
            Layer ID to set up.

        Returns
        -------
        :
            The newly created DynamicMap, or None if no data available yet.
        """
        # Already set up
        if layer_id in self._dmaps:
            return self._dmaps[layer_id]

        # Get data from shared service
        state_layer_id = StateLayerId(str(layer_id))
        state = self._plot_data_service.get(state_layer_id)
        if state is None or state.plotter is None:
            return None

        try:
            # Use Presenter pattern - plotter creates appropriate presenter
            presenter = state.plotter.create_presenter()
            pipe = hv.streams.Pipe(data=state.initial_data)
            dmap = presenter.present(pipe)

            self._presenters[layer_id] = presenter
            self._pipes[layer_id] = pipe
            self._dmaps[layer_id] = dmap
            self._last_versions[layer_id] = state.version

            # Store shared pipe reference for data forwarding
            if state.shared_pipe is not None:
                self._shared_pipes[layer_id] = state.shared_pipe
                self._last_data_ids[layer_id] = id(state.shared_pipe.data)

            logger.debug(
                "Created session DynamicMap for layer_id=%s (version %d)",
                layer_id,
                state.version,
            )

            return dmap

        except Exception:
            logger.exception(
                "Failed to create session DynamicMap for layer_id=%s",
                layer_id,
            )
            return None

    def update_pipes(self) -> set[LayerId]:
        """
        Forward data updates from shared pipes to session Pipes.

        Checks each shared pipe for data changes (via object identity) and
        sends new data to the corresponding session Pipe. This is how
        streaming data updates propagate to each browser session.

        Returns
        -------
        :
            Set of layer IDs that received updates.
        """
        updated_layers: set[LayerId] = set()

        for layer_id, shared_pipe in list(self._shared_pipes.items()):
            session_pipe = self._pipes.get(layer_id)
            if session_pipe is None:
                continue

            # Check if data has changed using object identity
            current_data = shared_pipe.data
            current_data_id = id(current_data)
            last_data_id = self._last_data_ids.get(layer_id, 0)

            if current_data_id != last_data_id:
                try:
                    session_pipe.send(current_data)
                    self._last_data_ids[layer_id] = current_data_id
                    updated_layers.add(layer_id)
                    logger.debug(
                        "Forwarded data update to session pipe for layer_id=%s",
                        layer_id,
                    )
                except Exception:
                    logger.exception(
                        "Failed to forward data to session pipe for layer_id=%s",
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
        self._last_versions.clear()
        self._shared_pipes.clear()
        self._last_data_ids.clear()
        logger.debug("SessionPlotManager cleaned up")
