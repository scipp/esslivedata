# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""
Plot data service for multi-session synchronization.

Provides storage for plot layer state (plotter, error, stopped).
Change notification is handled by the Plotter/Presenter dirty flag mechanism.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, NewType
from uuid import UUID

if TYPE_CHECKING:
    from .plots import Plotter

logger = logging.getLogger(__name__)


@dataclass
class PlotLayerState:
    """State for a single plot layer.

    Stores the plotter reference and lifecycle flags. The computed state
    is cached within the plotter itself (via get_cached_state()).
    Change notification is handled by the Plotter/Presenter dirty flag mechanism.
    """

    plotter: Plotter | None = field(default=None)
    error: str | None = field(default=None)
    stopped: bool = field(default=False)


LayerId = NewType('LayerId', UUID)


class PlotDataService:
    """
    Stores plot layer state (plotter, error, stopped).

    PlotOrchestrator stores plotter references after setup.
    Change notification is handled by the Plotter/Presenter dirty flag mechanism -
    when plotter.compute() is called, it marks all presenters dirty.

    Thread-safe: can be called from background threads and periodic callbacks.
    """

    def __init__(self) -> None:
        self._layers: dict[LayerId, PlotLayerState] = {}
        self._lock = threading.Lock()

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

    def set_plotter(self, layer_id: LayerId, plotter: Any) -> None:
        """
        Set the plotter for a layer.

        Creates or replaces the layer entry. This resets any previous
        error/stopped state since setting a plotter means starting fresh.

        Parameters
        ----------
        layer_id:
            Layer ID to update.
        plotter:
            Plotter instance for per-session presenter creation.
        """
        with self._lock:
            # Create or reset layer state - setting a plotter means starting fresh
            self._layers[layer_id] = PlotLayerState(plotter=plotter)
            logger.debug("Set plotter for %s", layer_id)

    def set_error(self, layer_id: LayerId, error_msg: str) -> None:
        """
        Set error state for a layer.

        Marks presenters dirty so sessions see the change.

        Parameters
        ----------
        layer_id:
            Layer ID to update.
        error_msg:
            Error message to display.
        """
        with self._lock:
            state = self._layers.setdefault(layer_id, PlotLayerState())
            state.error = error_msg
            if state.plotter is not None:
                state.plotter.mark_presenters_dirty()
            logger.debug("Set error for %s: %s", layer_id, error_msg)

    def set_stopped(self, layer_id: LayerId) -> None:
        """
        Mark a layer as stopped.

        Sets the stopped flag to indicate the workflow has ended and no more
        data is expected. Marks presenters dirty so sessions see the change.

        Parameters
        ----------
        layer_id:
            Layer ID to update.
        """
        with self._lock:
            state = self._layers.setdefault(layer_id, PlotLayerState())
            state.stopped = True
            if state.plotter is not None:
                state.plotter.mark_presenters_dirty()
            logger.debug("Set stopped for %s", layer_id)

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
