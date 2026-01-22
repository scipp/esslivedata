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
        Poll PlotDataService and forward updates to session Pipes.

        Checks each tracked layer for version changes and sends new data
        to the corresponding session Pipe.

        Returns
        -------
        :
            Set of layer IDs that received updates.
        """
        updated_layers: set[LayerId] = set()
        updates = self._plot_data_service.get_updates_since(
            {StateLayerId(str(k)): v for k, v in self._last_versions.items()}
        )

        for state_layer_id, state in updates.items():
            # Convert back to LayerId (strip the StateLayerId wrapper)
            layer_id_str = str(state_layer_id)
            matching_layer_id = None
            for lid in self._pipes.keys():
                if str(lid) == layer_id_str:
                    matching_layer_id = lid
                    break

            if matching_layer_id is None:
                continue

            pipe = self._pipes.get(matching_layer_id)
            if pipe is None:
                continue

            try:
                pipe.send(state.initial_data)
                self._last_versions[matching_layer_id] = state.version
                updated_layers.add(matching_layer_id)
                logger.debug(
                    "Forwarded data update to session pipe for layer_id=%s "
                    "(version %d)",
                    matching_layer_id,
                    state.version,
                )
            except Exception:
                logger.exception(
                    "Failed to forward data to session pipe for layer_id=%s",
                    matching_layer_id,
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
        logger.debug("SessionPlotManager cleaned up")
