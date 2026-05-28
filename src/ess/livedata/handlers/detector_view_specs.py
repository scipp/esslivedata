# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""
Lightweight detector view spec registration.

This module provides spec registration for detector views WITHOUT importing
heavy dependencies like ess.reduce.live.raw. It should be imported by instrument
specs modules to register detector view specifications.

Factory implementations that use ess.reduce.live.raw are in detector_data_handler.py
and should only be imported by backend services.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import ClassVar, Literal

import pydantic
import scipp as sc

from .. import parameter_models
from ..config import models
from ..config.instrument import Instrument
from ..config.workflow_spec import (
    DETECTORS,
    JobId,
    OutputView,
    WorkflowOutputsBase,
)
from ..core.message import Message, StreamId, StreamKind
from ..core.timestamp import Timestamp
from ..handlers.workflow_factory import SpecHandle

CoordinateMode = Literal['toa', 'wavelength']


class CoordinateModeSettings(pydantic.BaseModel):
    """Settings for coordinate mode selection."""

    mode: CoordinateMode = pydantic.Field(
        default='toa',
        description="Coordinate system for event data: 'toa' (time-of-arrival) "
        "or 'wavelength'.",
    )


class DetectorViewParams(pydantic.BaseModel):
    coordinate_mode: CoordinateModeSettings = pydantic.Field(
        title="Coordinate Mode",
        description="Select coordinate system for detector view.",
        default_factory=CoordinateModeSettings,
    )
    pixel_weighting: models.PixelWeighting = pydantic.Field(
        title="Pixel Weighting",
        description="Whether to apply pixel weighting based on the number of pixels "
        "contributing to each screen pixel.",
        default=models.PixelWeighting(
            enabled=False, method=models.WeightingMethod.PIXEL_NUMBER
        ),
    )
    # TOA (time-of-arrival) settings
    toa_range: parameter_models.TOARange = pydantic.Field(
        title="Time of Arrival Range",
        description="Time of arrival range filter for TOA mode.",
        default=parameter_models.TOARange(),
    )
    toa_edges: parameter_models.TOAEdges = pydantic.Field(
        title="Time of Arrival Edges",
        description="Time of arrival edges for histogramming in TOA mode.",
        default=parameter_models.TOAEdges(
            start=0.0,
            stop=1000.0 / 14,
            num_bins=100,
            unit=parameter_models.TimeUnit.MS,
        ),
    )
    # Wavelength settings
    wavelength_range: parameter_models.WavelengthRangeFilter = pydantic.Field(
        title="Wavelength Range",
        description="Wavelength range filter for wavelength mode.",
        default=parameter_models.WavelengthRangeFilter(),
    )
    wavelength_edges: parameter_models.WavelengthEdges = pydantic.Field(
        title="Wavelength Edges",
        description="Wavelength edges for histogramming in wavelength mode.",
        default=parameter_models.WavelengthEdges(
            start=1.0,
            stop=10.0,
            num_bins=100,
            unit=parameter_models.WavelengthUnit.ANGSTROM,
        ),
    )

    def get_active_edges(self) -> sc.Variable:
        """Return the edges for the currently selected coordinate mode."""
        match self.coordinate_mode.mode:
            case 'toa':
                return self.toa_edges.get_edges()
            case 'wavelength':
                return self.wavelength_edges.get_edges()

    def get_active_range(self) -> tuple[sc.Variable, sc.Variable] | None:
        """Return the range for the currently selected coordinate mode, if enabled."""
        match self.coordinate_mode.mode:
            case 'toa':
                return self.toa_range.range if self.toa_range.enabled else None
            case 'wavelength':
                return (
                    self.wavelength_range.range
                    if self.wavelength_range.enabled
                    else None
                )


@dataclass(frozen=True, slots=True)
class SpectrumViewSpec:
    """Per-instrument configuration enabling a spectrum-view output.

    Parameters
    ----------
    transform:
        Callable applied to the cumulative accumulated histogram to produce
        the spectrum view. Signature is ``(histogram,) -> spectrum_view`` when
        ``params_model`` is ``None``, else ``(histogram, params) -> spectrum_view``
        where ``params`` is an instance of ``params_model``.
    output_dims:
        Spatial output dimension names, used for the initial empty template of
        the ``spectrum_view`` field. The transform preserves the spectral axis
        of the input histogram (time-of-arrival or wavelength depending on mode),
        so it is not listed here.
    output_title:
        Human-readable title for the output field.
    extra_description:
        Instrument-specific description appended as a second paragraph to the
        base description.
    params_model:
        Optional pydantic model carrying runtime parameters for the transform.
        When provided, a ``spectrum_params`` field of this type is injected into
        the generated ``DetectorViewParams`` subclass and passed to the
        transform. When ``None`` (default), the transform takes only the
        histogram and no parameter widget is shown in the UI.
    params_description:
        Description for the ``spectrum_params`` field (only used when
        ``params_model`` is set).
    """

    transform: Callable[..., sc.DataArray]
    output_dims: list[str]
    output_title: str = 'Spectrum View'
    extra_description: str = ''
    params_model: type[pydantic.BaseModel] | None = None
    params_description: str = 'Runtime parameters for the spectrum-view.'

    @property
    def output_description(self) -> str:
        base = (
            'Accumulated histogram reshaped into a per-spatial-group spectrum. '
            'The last axis is the spectral coordinate of the input histogram '
            '(time-of-arrival or wavelength, depending on the workflow mode).'
        )
        if self.extra_description:
            return f'{base}\n\n{self.extra_description}'
        return base


def _make_nd_template(ndim: int, *, with_time_coord: bool = False) -> sc.DataArray:
    """Create an empty template with the specified number of dimensions."""
    coords = {'time': sc.scalar(0, unit='ns')} if with_time_coord else {}
    if ndim == 0:
        return sc.DataArray(sc.scalar(0, unit='counts'), coords=coords)
    dims = [f'dim_{i}' for i in range(ndim)]
    return sc.DataArray(
        sc.zeros(dims=dims, shape=[0] * ndim, unit='counts'), coords=coords
    )


def _make_2d_template() -> sc.DataArray:
    """Create an empty 2D template for cumulative outputs (no time coord)."""
    return _make_nd_template(2)


def _make_2d_template_with_time() -> sc.DataArray:
    """Create an empty 2D template with time coord for current outputs."""
    return _make_nd_template(2, with_time_coord=True)


def _make_0d_template() -> sc.DataArray:
    """Create an empty 0D template for cumulative scalar outputs (no time coord)."""
    return _make_nd_template(0)


def _make_0d_template_with_time() -> sc.DataArray:
    """Create an empty 0D template with time coord for scalar outputs."""
    return _make_nd_template(0, with_time_coord=True)


_BASE_DETECTOR_VIEWS: tuple[OutputView, ...] = (
    OutputView(
        name='image',
        title='Image',
        streams={'since_start': 'cumulative', 'per_update': 'current'},
        description=(
            'Detector image. With "since run start" shows accumulated counts; '
            'with "latest update" or a window, shows recent counts.'
        ),
    ),
    OutputView(
        name='total_counts',
        title='Total',
        streams={
            'since_start': 'counts_total_cumulative',
            'per_update': 'counts_total',
        },
        description='Total number of detector events.',
    ),
    OutputView(
        name='total_in_range',
        title='Total in range',
        streams={
            'since_start': 'counts_in_toa_range_cumulative',
            'per_update': 'counts_in_toa_range',
        },
        description=('Number of detector events within the configured range filter.'),
    ),
)


class DetectorViewOutputsBase(WorkflowOutputsBase):
    """Base outputs for detector view workflows (without ROI support)."""

    output_views: ClassVar[tuple[OutputView, ...]] = _BASE_DETECTOR_VIEWS

    # Field names are kept stable as wire-format identifiers (ResultKey, da00
    # serialisation) and are referenced by ``output_views``.

    cumulative: sc.DataArray = pydantic.Field(
        title='Image',
        description='Detector image accumulated since the start of the run.',
        default_factory=_make_2d_template,
    )
    current: sc.DataArray = pydantic.Field(
        title='Image update',
        description=(
            'Detector image for the latest update interval only. '
            'Resets each update interval.'
        ),
        default_factory=_make_2d_template_with_time,
    )
    counts_total_cumulative: sc.DataArray = pydantic.Field(
        title='Total',
        description=(
            'Total number of detector events accumulated since the start of the run.'
        ),
        default_factory=_make_0d_template,
    )
    counts_total: sc.DataArray = pydantic.Field(
        title='Total (update)',
        description=(
            'Total number of detector events for the latest update interval only. '
            'Resets each update interval.'
        ),
        default_factory=_make_0d_template_with_time,
    )
    counts_in_toa_range_cumulative: sc.DataArray = pydantic.Field(
        title='Total in range',
        description=(
            'Number of detector events within the configured range filter '
            'accumulated since the start of the run.'
        ),
        default_factory=_make_0d_template,
    )
    counts_in_toa_range: sc.DataArray = pydantic.Field(
        title='Total in range (update)',
        description=(
            'Number of detector events within the configured range filter '
            'for the latest update interval only. Resets each update interval.'
        ),
        default_factory=_make_0d_template_with_time,
    )


class DetectorViewOutputs(DetectorViewOutputsBase):
    """Outputs for detector view workflows with ROI support."""

    output_views: ClassVar[tuple[OutputView, ...]] = (
        *_BASE_DETECTOR_VIEWS,
        OutputView(
            name='roi_spectra',
            title='ROI spectra',
            streams={
                'since_start': 'roi_spectra_cumulative',
                'per_update': 'roi_spectra_current',
            },
            description='Histogram for each active ROI region.',
        ),
        OutputView(
            name='roi_rectangle',
            title='ROI Rectangles (readback)',
            streams={'since_start': 'roi_rectangle'},
            description='Current rectangle ROI geometries confirmed by backend.',
        ),
        OutputView(
            name='roi_polygon',
            title='ROI Polygons (readback)',
            streams={'since_start': 'roi_polygon'},
            description='Current polygon ROI geometries confirmed by backend.',
        ),
    )

    # Stacked ROI spectra outputs (2D: roi x time_of_arrival)
    roi_spectra_cumulative: sc.DataArray = pydantic.Field(
        title='ROI spectra',
        description=(
            'Histogram for each active ROI region '
            'accumulated since the start of the run.'
        ),
        default_factory=lambda: sc.DataArray(
            sc.zeros(dims=['roi', 'time_of_arrival'], shape=[0, 0], unit='counts'),
            coords={'roi': sc.array(dims=['roi'], values=[], unit=None)},
        ),
    )
    roi_spectra_current: sc.DataArray = pydantic.Field(
        title='ROI spectra update',
        description=(
            'Histogram for each active ROI region '
            'for the latest update interval only. Resets each update interval.'
        ),
        default_factory=lambda: sc.DataArray(
            sc.zeros(dims=['roi', 'time_of_arrival'], shape=[0, 0], unit='counts'),
            coords={
                'roi': sc.array(dims=['roi'], values=[], unit=None),
                'time': sc.scalar(0, unit='ns'),
            },
        ),
    )

    # ROI geometry readbacks
    roi_rectangle: sc.DataArray = pydantic.Field(
        title='ROI Rectangles (readback)',
        description='Current rectangle ROI geometries confirmed by backend.',
        default_factory=lambda: models.RectangleROI.to_concatenated_data_array({}),
    )
    roi_polygon: sc.DataArray = pydantic.Field(
        title='ROI Polygons (readback)',
        description='Current polygon ROI geometries confirmed by backend.',
        default_factory=lambda: models.PolygonROI.to_concatenated_data_array({}),
    )


def _make_spectrum_template(output_dims: list[str]) -> sc.DataArray:
    # Append a placeholder spectral dim so the template has the right ndim for
    # plotter selection. The actual dim name is determined at runtime by the transform.
    dims = [*output_dims, '<spectral_coord>']
    return sc.DataArray(sc.zeros(dims=dims, shape=[0] * len(dims), unit='counts'))


def make_detector_view_outputs(
    output_ndim: int | None = None,
    *,
    roi_support: bool = True,
    spectrum_view: SpectrumViewSpec | None = None,
) -> type[DetectorViewOutputsBase]:
    """
    Create a DetectorViewOutputs subclass with the appropriate configuration.

    Parameters
    ----------
    output_ndim:
        Number of dimensions for spatial outputs (cumulative, current).
        The counts outputs remain 0D scalars with time coord.
        If None, uses 2D default.
    roi_support:
        Whether to include ROI-related outputs. If False, the returned class
        will not include roi_spectra_current, roi_spectra_cumulative,
        roi_rectangle, or roi_polygon fields.
    spectrum_view:
        Optional spectrum view configuration. When provided, the returned
        class includes an additional ``spectrum_view`` field with a template
        matching ``spectrum_view.output_dims``.

    Returns
    -------
    :
        A subclass of DetectorViewOutputsBase with appropriate configuration.
    """
    base_class: type[DetectorViewOutputsBase] = (
        DetectorViewOutputs if roi_support else DetectorViewOutputsBase
    )

    if output_ndim is None and spectrum_view is None:
        return base_class

    if output_ndim is not None:

        def make_cumulative_template() -> sc.DataArray:
            return _make_nd_template(output_ndim)

        def make_current_template() -> sc.DataArray:
            return _make_nd_template(output_ndim, with_time_coord=True)

        class _WithNdim(base_class):  # type: ignore[valid-type,misc]
            cumulative: sc.DataArray = pydantic.Field(
                title='Image (cumulative)',
                description='Detector image accumulated since the start of the run.',
                default_factory=make_cumulative_template,
            )
            current: sc.DataArray = pydantic.Field(
                title='Image (current)',
                description=(
                    'Detector image for the latest update interval only. '
                    'Resets each update interval.'
                ),
                default_factory=make_current_template,
            )

        base_class = _WithNdim

    if spectrum_view is not None:
        output_dims = list(spectrum_view.output_dims)

        def make_spectrum_template() -> sc.DataArray:
            return _make_spectrum_template(output_dims)

        title = spectrum_view.output_title
        description = spectrum_view.output_description
        base_views = tuple(base_class.output_views)

        class _WithSpectrum(base_class):  # type: ignore[valid-type,misc]
            output_views: ClassVar[tuple[OutputView, ...]] = (
                *base_views,
                OutputView(
                    name='spectrum_view',
                    title=title,
                    streams={'since_start': 'spectrum_view'},
                    description=description,
                ),
            )

            spectrum_view: sc.DataArray = pydantic.Field(
                title=title,
                description=description,
                default_factory=make_spectrum_template,
            )

        base_class = _WithSpectrum

    return base_class


def make_detector_view_params(
    spectrum_view: SpectrumViewSpec | None = None,
) -> type[DetectorViewParams]:
    """Return a ``DetectorViewParams`` subclass, adding spectrum-specific fields.

    When ``spectrum_view.params_model`` is set, the subclass adds a
    ``spectrum_params`` field of that model type so the runtime parameters can
    be exposed in the UI. Workflows without spectrum-view (or whose spectrum
    transform needs no runtime parameters) keep the base ``DetectorViewParams``
    unchanged.
    """
    if spectrum_view is None or spectrum_view.params_model is None:
        return DetectorViewParams

    params_model = spectrum_view.params_model
    title = spectrum_view.output_title
    description = spectrum_view.params_description

    class DetectorViewWithSpectrumParams(DetectorViewParams):
        spectrum_params: params_model = pydantic.Field(  # type: ignore[valid-type]
            title=title,
            description=description,
            default_factory=params_model,
        )

    return DetectorViewWithSpectrumParams


ProjectionType = Literal["xy_plane", "cylinder_mantle_z"]


def _roi_rectangle_seed(job_id: JobId) -> Message:
    """Cold-start "no rectangle ROI selected" message for a job's ROI context.

    Byte-identical to what the dashboard publishes when the user deletes the
    last rectangle ROI; pre-seeded at ``schedule_job`` time so the gate opens
    immediately. See ADR 0002.
    """
    return Message(
        timestamp=Timestamp.from_ns(0),
        stream=StreamId(kind=StreamKind.LIVEDATA_ROI, name=f"{job_id}/roi_rectangle"),
        value=models.RectangleROI.to_concatenated_data_array({}),
    )


def _roi_polygon_seed(job_id: JobId) -> Message:
    """Cold-start "no polygon ROI selected" message for a job's ROI context."""
    return Message(
        timestamp=Timestamp.from_ns(0),
        stream=StreamId(kind=StreamKind.LIVEDATA_ROI, name=f"{job_id}/roi_polygon"),
        value=models.PolygonROI.to_concatenated_data_array({}),
    )


def add_roi_context_inputs(handle: SpecHandle) -> None:
    """Declare the rectangle/polygon ROI context inputs for a detector-view spec.

    The wire-stream resolver prefixes the stream name with the JobId so each
    job instance owns its ROI configuration stream.
    """
    from .detector_view.types import ROIPolygonRequest, ROIRectangleRequest

    handle.add_context_input(
        stream_name='roi_rectangle',
        workflow_key=ROIRectangleRequest,
        stream_resolver=lambda job_id, name: f"{job_id}/{name}",
        seed_factory=_roi_rectangle_seed,
    )
    handle.add_context_input(
        stream_name='roi_polygon',
        workflow_key=ROIPolygonRequest,
        stream_resolver=lambda job_id, name: f"{job_id}/{name}",
        seed_factory=_roi_polygon_seed,
    )


def register_detector_view_spec(
    *,
    instrument: Instrument,
    projection: ProjectionType | dict[str, ProjectionType],
    source_names: list[str] | None = None,
    spectrum_view: SpectrumViewSpec | None = None,
) -> SpecHandle:
    """
    Register detector view specs for a given projection.

    This is a lightweight helper that registers workflow specs without creating
    the actual detector view objects.

    Parameters
    ----------
    instrument:
        Instrument to register specs with.
    projection:
        Projection type(s) to register. Either a single projection type applied to
        all sources, or a dict mapping source names to projection types. When a dict
        is provided, this creates a unified "Detector Projection" workflow that
        uses different projections for different detector banks.
    source_names:
        List of detector source names. Required when projection is a single type.
        When projection is a dict, defaults to the dict keys if not specified.
    spectrum_view:
        Optional spectrum-view configuration. When provided, the registered
        params/outputs include the spectrum-specific rebin param and the
        ``spectrum_view`` output field. The factory is still responsible for
        wiring the transform into the Sciline workflow (pass ``spectrum_view`` on
        the Sciline ``GeometricViewConfig`` / ``LogicalViewConfig`` when
        constructing ``DetectorViewFactory``).

    Returns
    -------
    :
        A SpecHandle.

    Example
    -------
    Single projection for all detectors:

    .. code-block:: python

        handle = register_detector_view_spec(
            instrument=instrument,
            projection='xy_plane',
            source_names=['detector_0', 'detector_1'],
        )

    Mixed projections (unified workflow):

    .. code-block:: python

        handle = register_detector_view_spec(
            instrument=instrument,
            projection={
                'mantle_detector': 'cylinder_mantle_z',
                'endcap_backward_detector': 'xy_plane',
                'endcap_forward_detector': 'xy_plane',
            },
        )
    """
    if isinstance(projection, dict):
        # Mixed projections - create unified "Detector Projection" workflow
        if source_names is None:
            source_names = list(projection.keys())
        name = "detector_projection"
        title = "Detector Projection"
        description = (
            "Projection of detector banks onto 2D planes. "
            "Uses the appropriate projection for each detector."
        )
    elif projection == "xy_plane":
        if source_names is None:
            raise ValueError("source_names is required when projection is a string")
        name = "detector_xy_projection"
        title = "Detector XY Projection"
        description = "Projection of a detector bank onto an XY-plane."
    elif projection == "cylinder_mantle_z":
        if source_names is None:
            raise ValueError("source_names is required when projection is a string")
        name = "detector_cylinder_mantle_z"
        title = "Detector Cylinder Mantle Z Projection"
        description = (
            "Projection of a detector bank onto a cylinder mantle along Z-axis."
        )
    else:
        raise ValueError(f"Unsupported projection: {projection}")

    handle = instrument.register_spec(
        group=DETECTORS,
        name=name,
        version=1,
        title=title,
        description=description,
        source_names=source_names,
        params=make_detector_view_params(spectrum_view=spectrum_view),
        outputs=make_detector_view_outputs(
            roi_support=True, spectrum_view=spectrum_view
        ),
    )
    add_roi_context_inputs(handle)
    return handle
