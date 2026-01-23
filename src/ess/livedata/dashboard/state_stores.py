# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""
Plot data service for multi-session synchronization.

Provides polling-based access to computed plot state, ensuring each
session's periodic callback accesses state in the correct session context.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from typing import Any, NewType

logger = logging.getLogger(__name__)


@dataclass
class PlotLayerState:
    """State for a single plot layer with version tracking.

    Stores the computed plot state along with references needed for
    per-session component creation:

    - state: The computed HoloViews elements from plotter.compute()
    - plotter: Reference to the Plotter instance for create_presenter()
    """

    state: Any  # The computed HoloViews elements from plotter.compute()
    version: int = 0
    plotter: Any = None  # Reference to Plotter for create_presenter()


LayerId = NewType('LayerId', str)


class PlotDataService:
    """
    Stores computed plot state with version tracking.

    PlotOrchestrator stores results after calling `plotter.compute()`.
    Each session polls for updates in its periodic callback.

    Thread-safe: can be called from background threads and periodic callbacks.
    """

    def __init__(self) -> None:
        self._layers: dict[LayerId, PlotLayerState] = {}
        self._lock = threading.Lock()

    def update(
        self,
        layer_id: LayerId,
        state: Any,
        *,
        plotter: Any = None,
    ) -> None:
        """
        Update state for a layer.

        Parameters
        ----------
        layer_id:
            Layer ID to update.
        state:
            New computed plot state (HoloViews elements from plotter.compute()).
        plotter:
            Optional plotter instance for per-session presenter creation.
            Only needs to be provided on first update.
        """
        with self._lock:
            if layer_id in self._layers:
                current = self._layers[layer_id]
                # Preserve plotter reference if not provided on update
                effective_plotter = plotter if plotter is not None else current.plotter
                self._layers[layer_id] = PlotLayerState(
                    state=state,
                    version=current.version + 1,
                    plotter=effective_plotter,
                )
            else:
                self._layers[layer_id] = PlotLayerState(
                    state=state,
                    version=1,
                    plotter=plotter,
                )
            logger.debug(
                "Updated plot state for %s at version %d",
                layer_id,
                self._layers[layer_id].version,
            )

    def get(self, layer_id: LayerId) -> PlotLayerState | None:
        """
        Get current state for a layer.

        Parameters
        ----------
        layer_id:
            Layer ID to retrieve.

        Returns
        -------
        :
            Current layer state, or None if not set.
        """
        with self._lock:
            return self._layers.get(layer_id)

    def get_version(self, layer_id: LayerId) -> int:
        """
        Get version for a specific layer.

        Parameters
        ----------
        layer_id:
            Layer ID to check.

        Returns
        -------
        :
            Version number, or 0 if layer doesn't exist.
        """
        with self._lock:
            if layer_id in self._layers:
                return self._layers[layer_id].version
            return 0

    def get_updates_since(
        self, versions: dict[LayerId, int]
    ) -> dict[LayerId, PlotLayerState]:
        """
        Get layers that have been updated since the given versions.

        Parameters
        ----------
        versions:
            Dictionary mapping layer IDs to last-seen versions.

        Returns
        -------
        :
            Dictionary of layers with newer versions than provided.
        """
        with self._lock:
            updates: dict[LayerId, PlotLayerState] = {}
            for layer_id, layer_state in self._layers.items():
                last_version = versions.get(layer_id, 0)
                if layer_state.version > last_version:
                    updates[layer_id] = layer_state
            return updates

    def remove(self, layer_id: LayerId) -> None:
        """
        Remove state for a layer.

        Parameters
        ----------
        layer_id:
            Layer ID to remove.
        """
        with self._lock:
            if layer_id in self._layers:
                del self._layers[layer_id]
                logger.debug("Removed plot state for %s", layer_id)

    def clear(self) -> None:
        """Clear all state. Mainly useful for testing."""
        with self._lock:
            self._layers.clear()
