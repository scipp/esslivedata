# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""
SessionLayer - Per-session state for a single plot layer.

Each browser session creates SessionLayer instances to hold session-bound
rendering state (presenter, current frame, figure subscriptions or a
DynamicMap) when data is available; the instance also serves as the
session's viewer-interest token.
"""

from __future__ import annotations

from collections.abc import Callable, Hashable
from dataclasses import dataclass, field

import holoviews as hv

from .frame import Frame
from .plot_data_service import LayerId, LayerSnapshot
from .plots import Plotter, PresenterBase


@dataclass
class SessionComponents:
    """
    Session-bound state for rendering a layer.

    Figures receive the layer's frames by one of two routes, chosen by the cell
    that draws the layer (see ``widgets/cell.py``):

    - Figures built once and updated by writing each frame's columns into their
      data sources register a callback with :meth:`subscribe`.
    - A HoloViews ``DynamicMap`` fed by a ``Pipe`` (:attr:`dmap`), for frames
      whose element types cannot be written directly.

    Parameters
    ----------
    presenter:
        Per-session presenter created from the plotter.
    frame:
        The frame this session's figures of the layer show: the plotter's
        current frame at creation, then the last one pushed. Figures accept
        only frames of its spec; a frame of another spec rebuilds the cell
        instead, creating new components (see :meth:`is_valid_for`).
    """

    presenter: PresenterBase
    frame: Frame
    _subscribers: list[Callable[[Frame], None]] = field(
        default_factory=list, init=False
    )
    _pipe: hv.streams.Pipe | None = field(default=None, init=False)
    _dmap: hv.DynamicMap | hv.Element | None = field(default=None, init=False)

    @property
    def spec(self) -> Hashable:
        """Spec of the frames this layer's figures accept."""
        return self.frame.spec

    @property
    def dmap(self) -> hv.DynamicMap | hv.Element:
        """The layer's DynamicMap or Element, created on first access.

        It outlives every cell widget built over it (until the plotter is
        replaced), so builds must compose over it without modifying it.
        ``DynamicMap.opts()`` modifies it: it wraps the map's callback in place
        and returns the same object, so each build would stack one more wrap.
        Pass ``clone=True``.
        """
        if self._dmap is None:
            self._pipe = hv.streams.Pipe(data=self.frame.element)
            self._dmap = self.presenter.present(self._pipe)
        return self._dmap

    def subscribe(self, callback: Callable[[Frame], None]) -> None:
        """Call ``callback`` with every frame pushed from now on."""
        self._subscribers.append(callback)

    def unsubscribe(self, callback: Callable[[Frame], None]) -> None:
        """Stop calling ``callback``; a no-op if it is not subscribed."""
        if callback in self._subscribers:
            self._subscribers.remove(callback)

    def push_update(self) -> bool:
        """
        Push the pending frame to the layer's figures, if there is one.

        A frame of a different spec is not pushed: no figure built from
        :attr:`frame` can show it, and for a DynamicMap of a layout it would
        stop the map from updating for good. The cell rebuilds on the spec
        change instead.

        Returns
        -------
        :
            True if a frame was pushed, False otherwise.
        """
        if not self.presenter.has_pending_update():
            return False
        frame = self.presenter.consume_update()
        if frame.spec != self.frame.spec:
            return False
        self.frame = frame
        for callback in list(self._subscribers):
            callback(frame)
        if self._pipe is not None:
            self._pipe.send(frame.element)
        return True

    def is_valid_for(self, plotter: Plotter | None) -> bool:
        """
        Check if these components are still valid for the given plotter.

        Returns False if the plotter has been replaced (e.g., workflow restart)
        or its frame has a different spec, indicating the components should be
        recreated.
        """
        return (
            plotter is not None
            and self.presenter.is_owned_by(plotter)
            and plotter.frame_spec() == self.spec
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
        return cls(presenter=plotter.create_presenter(), frame=plotter.get_frame())


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

    def push_update(self) -> bool:
        """
        Push the pending frame to the layer's figures, if components exist.

        Returns
        -------
        :
            True if a frame was pushed, False otherwise.
        """
        if self.components is None:
            return False
        return self.components.push_update()

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
