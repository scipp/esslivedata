# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Interactive histogram slice range plotter for spectrum plots.

This plotter creates a Bokeh RangeTool overlay on spectrum plots that allows
users to select a spectral range for filtering the detector image.
Edits are debounced and published to Kafka for backend processing.

Architecture
------------
Follows the same two-stage compute/present pattern as ROI request plotters:

1. compute(): Extracts ResultKey and spectral unit from the readback data.

2. create_presenter(): Creates a presenter with a Bokeh hook that attaches
   a RangeTool to the plot. The hook captures per-session range state and
   a debounce timer in a closure.

3. Presenter.present(): Returns a transparent Curve element whose sole purpose
   is to carry the Bokeh hook that installs the RangeTool on the figure.

The presenter handles only HoloViews/Bokeh mechanics. Domain logic (range
comparison, publishing) stays in the plotter via the edit callback.
"""

from __future__ import annotations

from collections.abc import Callable
from threading import Timer
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import holoviews as hv
import numpy as np
import pydantic
import structlog
from bokeh.models import Range1d
from bokeh.models.tools import RangeTool

from .plots import Plotter, PresenterBase
from .static_plots import Color, LineDash

if TYPE_CHECKING:
    from ..config.workflow_spec import ResultKey
    from .range_publisher import RangePublisher

logger = structlog.get_logger(__name__)

DEBOUNCE_DELAY_S = 0.3
"""Delay before publishing a range change after the user stops dragging."""


@runtime_checkable
class RangePublisherAware(Protocol):
    """Protocol for plotters that can publish range updates."""

    def set_range_publisher(self, publisher: RangePublisher | None) -> None:
        """Set the range publisher for this plotter."""
        ...


class RangeRequestStyle(pydantic.BaseModel):
    """Style options for range selection band."""

    color: Color = pydantic.Field(
        default=Color("#4488cc"),
        title="Color",
    )
    line_width: float = pydantic.Field(
        default=1.5,
        ge=0.0,
        le=10.0,
        title="Line Width",
        description="Line width in pixels",
    )
    line_dash: LineDash = pydantic.Field(
        default=LineDash.dashed,
        title="Line Style",
    )
    fill_alpha: float = pydantic.Field(
        default=0.15,
        ge=0.0,
        le=1.0,
        title="Fill Opacity",
        description="Opacity of the range band fill.",
    )


class RangeRequestParams(pydantic.BaseModel):
    """Parameters for interactive range request plotter."""

    style: RangeRequestStyle = pydantic.Field(
        default_factory=RangeRequestStyle,
        title="Appearance",
        description="Visual styling options.",
    )


class RangeRequestPresenter(PresenterBase):
    """Presenter for histogram range selection using Bokeh RangeTool.

    Attaches a RangeTool to the spectrum plot via a Bokeh hook. The tool
    provides a draggable/resizable band overlay. Range changes are debounced
    so that only the final position after the user stops dragging is published.

    Parameters
    ----------
    plotter:
        The plotter that created this presenter.
    style:
        Style parameters for the range band.
    on_edit:
        Callback invoked with ``(low, high)`` when the range settles after
        dragging, or ``None`` when the range is cleared.
    """

    def __init__(
        self,
        *,
        plotter: Plotter,
        style: RangeRequestStyle,
        on_edit: Callable[[tuple[float, float] | None], None],
    ) -> None:
        super().__init__(plotter)
        self._style = style
        self._on_edit = on_edit

    def present(self, pipe: hv.streams.Pipe) -> hv.Curve:
        """Return a transparent Curve that carries the RangeTool hook.

        The passed pipe is ignored — range request plotters do not render
        data, they only provide the interactive tool overlay.
        """
        del pipe
        style = self._style
        on_edit = self._on_edit

        # Per-session state captured in closure
        selection_range = Range1d()
        debounce_timer: list[Timer | None] = [None]
        committed_range: list[tuple[float, float] | None] = [None]

        def _settle():
            low = min(selection_range.start, selection_range.end)
            high = max(selection_range.start, selection_range.end)
            new_range = (low, high)
            if new_range != committed_range[0]:
                committed_range[0] = new_range
                try:
                    on_edit(new_range)
                except Exception:
                    logger.exception("Failed to process range edit")

        def _on_range_change(attr, old, new):
            if debounce_timer[0] is not None:
                debounce_timer[0].cancel()
            debounce_timer[0] = Timer(DEBOUNCE_DELAY_S, _settle)
            debounce_timer[0].daemon = True
            debounce_timer[0].start()

        def hook(plot, element):
            fig = plot.state
            tool = RangeTool(
                x_range=selection_range,
                x_interaction=True,
                y_interaction=False,
            )
            tool.overlay.fill_color = str(style.color)
            tool.overlay.fill_alpha = style.fill_alpha
            tool.overlay.line_color = str(style.color)
            tool.overlay.line_alpha = min(style.fill_alpha * 4, 1.0)
            tool.overlay.line_dash = style.line_dash.value
            tool.overlay.line_width = style.line_width

            tool.start_gesture = "tap"
            tool.overlay.inverted = True
            tool.overlay.use_handles = True
            tool.overlay.handles.all.hover_fill_color = "grey"
            tool.overlay.handles.all.hover_fill_alpha = 0.25
            tool.overlay.handles.all.fill_alpha = 0.1
            tool.overlay.handles.all.line_alpha = 0.25

            fig.add_tools(tool)
            selection_range.on_change("start", _on_range_change)
            selection_range.on_change("end", _on_range_change)

        # Return a transparent Curve that only serves as the hook carrier.
        # Using NaN data so it doesn't affect axis ranges.
        return hv.Curve([(np.nan, np.nan)]).opts(
            alpha=0,
            hooks=[hook],
        )


class RangeRequestPlotter(Plotter):
    """Interactive plotter for histogram range selection on spectrum plots.

    Creates presenters with RangeTool-enabled overlays that allow users
    to select a spectral range band. Edits are published to Kafka when
    the range changes (debounced).
    """

    def __init__(
        self,
        params: RangeRequestParams,
        range_publisher: RangePublisher | None = None,
    ) -> None:
        super().__init__()
        self._params = params
        self._range_publisher = range_publisher

        # Data-dependent state (set during compute())
        self._data_key: ResultKey | None = None
        self._spectral_unit: str | None = None

    def set_range_publisher(self, publisher: RangePublisher | None) -> None:
        """Set the range publisher for this plotter."""
        self._range_publisher = publisher

    def compute(self, data: dict[ResultKey, Any], **kwargs) -> dict[ResultKey, Any]:
        """Extract data-dependent info from histogram slice readback.

        Stores the ResultKey and spectral unit from the readback data.
        These are used by the edit handler callback created in create_presenter().

        Parameters
        ----------
        data:
            Dictionary with histogram slice readback data.
        **kwargs:
            Unused.

        Returns
        -------
        :
            The input data, forwarded.
        """
        del kwargs
        data_key, da = next(iter(data.items()))
        self._data_key = data_key

        # Extract spectral unit from readback data. The readback provider
        # always attaches the histogram bins unit, even when empty.
        if da.data.unit:
            self._spectral_unit = str(da.data.unit)

        self._set_cached_state(data)
        return data

    def create_presenter(self) -> RangeRequestPresenter:
        """Create a presenter for range selection."""
        on_edit = self._create_edit_handler()
        presenter = RangeRequestPresenter(
            plotter=self,
            style=self._params.style,
            on_edit=on_edit,
        )
        self._presenters.add(presenter)
        return presenter

    def _create_edit_handler(
        self,
    ) -> Callable[[tuple[float, float] | None], None]:
        """Create a per-session edit handler with closure-captured range state.

        Returns
        -------
        :
            Callback invoked with ``(low, high)`` when the range settles,
            or ``None`` when the range is cleared.
        """
        current_range: tuple[float, float] | None = None

        def handle_edit(new_range: tuple[float, float] | None) -> None:
            nonlocal current_range

            if new_range is None:
                if current_range is not None:
                    current_range = None
                    self._publish_clear()
                return

            low, high = min(new_range), max(new_range)
            normalized = (low, high)
            if normalized == current_range:
                return

            current_range = normalized
            self._publish_range(low, high)

        return handle_edit

    def _publish_range(self, low: float, high: float) -> None:
        """Publish a range update."""
        if not self._range_publisher or not self._data_key:
            return
        self._range_publisher.publish(
            self._data_key.job_id, low, high, self._spectral_unit
        )
        logger.info(
            "Published histogram slice [%s, %s] for job %s",
            low,
            high,
            self._data_key.job_id,
        )

    def _publish_clear(self) -> None:
        """Publish a range clear (restore full range)."""
        if not self._range_publisher or not self._data_key:
            return
        self._range_publisher.clear(self._data_key.job_id)
        logger.info(
            "Cleared histogram slice for job %s",
            self._data_key.job_id,
        )

    @classmethod
    def from_params(cls, params: RangeRequestParams) -> RangeRequestPlotter:
        """Create plotter from params (concrete type hint for registry)."""
        return cls(params)
