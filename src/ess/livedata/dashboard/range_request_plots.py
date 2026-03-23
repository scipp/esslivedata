# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Interactive histogram slice range plotter for spectrum plots.

This plotter creates a BoxEdit-enabled rectangle overlay on spectrum plots
that allows users to select a spectral range for filtering the detector image.
Edits are published to Kafka for backend processing.

Architecture
------------
Follows the same two-stage compute/present pattern as ROI request plotters:

1. compute(): Extracts ResultKey and spectral unit from the readback data.

2. create_presenter(): Creates a presenter with BoxEdit stream and an edit
   handler callback. The callback captures per-session range state in a closure.

3. Presenter.__init__(): Creates session-bound Pipe, DynamicMap, and BoxEdit
   stream. Each browser session gets its own presenter instance.

4. Presenter.present(): Returns the pre-created DynamicMap.

The presenter handles only HoloViews mechanics. Domain logic (range parsing,
comparison, publishing) stays in the plotter via the edit callback.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import holoviews as hv
import pydantic
import structlog

from .plots import Plotter, PresenterBase
from .static_plots import Color, LineDash

if TYPE_CHECKING:
    from ..config.workflow_spec import ResultKey
    from .range_publisher import RangePublisher

logger = structlog.get_logger(__name__)


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
    """
    Presenter for histogram range selection using BoxEdit on spectrum plot.

    Creates a session-bound Pipe, DynamicMap with Rectangles element,
    and BoxEdit stream for interactive range selection.

    Parameters
    ----------
    plotter:
        The plotter that created this presenter.
    initial_hv_data:
        Initial data in HoloViews Rectangles format.
    initial_stream_data:
        Initial data in BoxEdit stream format.
    style:
        Style parameters for the range band.
    on_edit:
        Callback for handling edit events.
    """

    def __init__(
        self,
        *,
        plotter: Plotter,
        initial_hv_data: list,
        initial_stream_data: dict,
        style: RangeRequestStyle,
        on_edit: Callable[[dict], None],
    ) -> None:
        super().__init__(plotter)
        self._style = style
        self._on_edit_callback = on_edit

        # Create session-bound components
        self._pipe = hv.streams.Pipe(data=[])
        self._dmap = hv.DynamicMap(self._create_element, streams=[self._pipe])
        self._edit_stream = hv.streams.BoxEdit(
            source=self._dmap,
            num_objects=1,
            data=initial_stream_data,
        )

        # Initialize pipe with data
        self._pipe.send(initial_hv_data)

        # Set up edit callback
        self._edit_stream.param.watch(self._handle_edit, 'data')

    def _create_element(self, data: list) -> hv.Rectangles:
        """Create HoloViews Rectangles element."""
        return hv.Rectangles(data)

    def present(self, pipe: hv.streams.Pipe) -> hv.DynamicMap:
        """
        Return pre-created DynamicMap.

        The passed pipe is ignored - range request plotters create their own
        internal pipe and don't update from external data changes.
        """
        del pipe
        return self._dmap.opts(
            color=self._style.color,
            fill_alpha=self._style.fill_alpha,
            line_width=self._style.line_width,
            line_dash=self._style.line_dash,
            # Bokeh bug: line_dash='dashed' doesn't render with WebGL backend
            backend_opts={'plot.output_backend': 'canvas'},
        )

    def _handle_edit(self, event) -> None:
        """Forward edit stream events to the plotter's callback."""
        data = event.new if hasattr(event, 'new') else event
        try:
            self._on_edit_callback(data if data is not None else {})
        except Exception as e:
            logger.error("Failed to process range edit: %s", e)


class RangeRequestPlotter(Plotter):
    """Interactive plotter for histogram range selection on spectrum plots.

    Creates presenters with BoxEdit-enabled DynamicMaps that allow users
    to draw a range band. Edits are published to Kafka when the range changes.
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
        """
        Extract data-dependent info from histogram slice readback.

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

        # Extract unit from the bound coordinate values
        if da.sizes.get('bound', 0) > 0:
            self._spectral_unit = str(da.data.unit) if da.data.unit else None

        self._set_cached_state(data)
        return data

    def create_presenter(self) -> RangeRequestPresenter:
        """Create a presenter for range selection."""
        on_edit = self._create_edit_handler()
        presenter = RangeRequestPresenter(
            plotter=self,
            initial_hv_data=[],
            initial_stream_data={'x0': [], 'x1': [], 'y0': [], 'y1': []},
            style=self._params.style,
            on_edit=on_edit,
        )
        self._presenters.add(presenter)
        return presenter

    def _create_edit_handler(self) -> Callable[[dict], None]:
        """
        Create a per-session edit handler with closure-captured range state.

        Returns
        -------
        :
            Callback function for handling edit events.
        """
        current_range: tuple[float, float] | None = None

        def handle_edit(stream_data: dict) -> None:
            nonlocal current_range

            x0_list = stream_data.get('x0', [])
            x1_list = stream_data.get('x1', [])

            if not x0_list:
                # Range was cleared
                if current_range is not None:
                    current_range = None
                    self._publish_clear()
                return

            new_range = (
                min(x0_list[0], x1_list[0]),
                max(x0_list[0], x1_list[0]),
            )
            if new_range == current_range:
                return

            current_range = new_range
            self._publish_range(*new_range)

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
