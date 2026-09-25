# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""
SessionLayer - Per-session state for a single plot layer.

Each browser session creates SessionLayer instances to hold session-bound
HoloViews components (Pipe, DynamicMap, PresenterBase) when data is
available; the instance also serves as the session's viewer-interest token.
"""

from __future__ import annotations

from dataclasses import dataclass

import holoviews as hv

from .plot_data_service import LayerId, LayerSnapshot
from .plots import Plotter, PresenterBase, layout_shape


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
        The DynamicMap or Element created by the presenter. It outlives every
        cell widget built over it (until the plotter is replaced), so builds
        must compose over it without modifying it. ``DynamicMap.opts()``
        modifies it: it wraps the map's callback in place and returns the same
        object, so each build would stack one more wrap. Pass ``clone=True``.
    layout_shape:
        :func:`~.plots.layout_shape` of the frame ``dmap`` was created with,
        the only shape of frame it can display.
    """

    presenter: PresenterBase
    pipe: hv.streams.Pipe
    dmap: hv.DynamicMap | hv.Element
    layout_shape: tuple[type, ...] | None

    def update_pipe(self) -> bool:
        """
        Push pending update to pipe if available.

        A frame of a different layout shape is not sent: it would stop ``dmap``
        from updating for good. The cell rebuilds on the shape change instead,
        creating new components (see :meth:`is_valid_for`).

        Returns
        -------
        :
            True if an update was sent, False if no pending update.
        """
        if not self.presenter.has_pending_update():
            return False
        frame = self.presenter.consume_update()
        if layout_shape(frame) != self.layout_shape:
            return False
        self.pipe.send(frame)
        return True

    def is_valid_for(self, plotter: Plotter | None) -> bool:
        """
        Check if these components are still valid for the given plotter.

        Returns False if the plotter has been replaced (e.g., workflow restart)
        or its frame has a different layout shape, indicating the components
        should be recreated.
        """
        return (
            plotter is not None
            and self.presenter.is_owned_by(plotter)
            and plotter.layout_shape() == self.layout_shape
        )

    @classmethod
    def create(cls, state: LayerSnapshot) -> SessionComponents | None:
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
        frame = plotter.get_cached_state()
        pipe = hv.streams.Pipe(data=frame)
        dmap = presenter.present(pipe)

        return cls(
            presenter=presenter,
            pipe=pipe,
            dmap=dmap,
            layout_shape=layout_shape(frame),
        )


@dataclass
class SessionLayer:
    """
    Per-session state for a single plot layer.

    Holds session-bound rendering components when data is available, and
    doubles as this session's viewer-interest token for the layer. Change
    detection lives with the reconcile pass, which compares the immutable
    :class:`~.plot_data_service.LayerSnapshot` a widget was built from against
    the current one (see ``cell_plan``).

    Parameters
    ----------
    layer_id:
        The layer's unique identifier.
    components:
        Session-bound rendering components, or None if no displayable data yet.
    """

    layer_id: LayerId
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

    def ensure_components(self, state: LayerSnapshot) -> bool:
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
