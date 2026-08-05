# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""
Lightweight detector view spec registration.

This module provides spec registration for detector views WITHOUT importing
heavy dependencies like ess.reduce.live.raw. It should be imported by instrument
specs modules to register detector view specifications.

Factory implementations that use ess.reduce.live.raw are in detector_data.py
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
from ..config.workflow_spec import (
    AuxSources,
    CumulativeOutput,
    JobId,
    OutputView,
    WindowOutput,
    WorkflowOutputsBase,
)

CoordinateMode = Literal['toa', 'wavelength']


class CoordinateModeSettings(pydantic.BaseModel):
    """Settings for coordinate mode selection."""

    mode: CoordinateMode = pydantic.Field(
        default='toa',
        description="Coordinate system for event data: 'toa' (time-of-arrival) "
        "or 'wavelength'.",
        json_schema_extra={
            'labels': {
                'toa': 'Time of arrival (TOA)',
                'wavelength': 'Wavelength',
            }
        },
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
        description=(
            "Time of arrival (TOA) is the time elapsed since the most recent "
            "source pulse. These edges define the histogram bins in TOA mode. "
            "The default range spans one pulse period of the 14 Hz ESS source "
            f"(0 to {parameter_models.ESS_PULSE_PERIOD_MS} ms); events outside "
            "the range are excluded from the histogram."
        ),
        default=parameter_models.TOAEdges(
            start=0.0,
            stop=parameter_models.ESS_PULSE_PERIOD_MS,
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


def _make_nd_template(ndim: int) -> sc.DataArray:
    """Create an empty template with the specified number of dimensions."""
    if ndim == 0:
        return sc.DataArray(sc.scalar(0, unit='counts'))
    dims = [f'dim_{i}' for i in range(ndim)]
    return sc.DataArray(sc.zeros(dims=dims, shape=[0] * ndim, unit='counts'))


def _make_2d_template() -> sc.DataArray:
    """Create an empty 2D template."""
    return _make_nd_template(2)


def _make_0d_template() -> sc.DataArray:
    """Create an empty 0D template for scalar outputs."""
    return _make_nd_template(0)


def _make_roi_spectra_template() -> sc.DataArray:
    """Create an empty template for stacked per-ROI spectra."""
    return sc.DataArray(
        sc.zeros(dims=['roi', 'time_of_arrival'], shape=[0, 0], unit='counts'),
        coords={'roi': sc.array(dims=['roi'], values=[], unit=None)},
    )


_BASE_DETECTOR_VIEWS: tuple[OutputView, ...] = (
    OutputView(
        name='image',
        title='Image',
        fields=('cumulative', 'current'),
        description=(
            'Detector image. With "since run start" shows accumulated counts; '
            'with "latest update" or a window, shows recent counts.'
        ),
        params=(
            'coordinate_mode',
            'pixel_weighting',
            'toa_edges',
            'wavelength_edges',
        ),
    ),
    OutputView(
        name='total_counts',
        title='Total',
        fields=('counts_total_cumulative', 'counts_total'),
        description='Total number of detector events.',
    ),
    OutputView(
        name='total_in_range',
        title='Total in range',
        fields=('counts_in_toa_range_cumulative', 'counts_in_toa_range'),
        description=('Number of detector events within the configured range filter.'),
        params=('coordinate_mode', 'toa_range', 'wavelength_range'),
    ),
)


class DetectorViewOutputsBase(WorkflowOutputsBase):
    """Base outputs for detector view workflows (without ROI support)."""

    output_views: ClassVar[tuple[OutputView, ...]] = _BASE_DETECTOR_VIEWS

    # Field names are kept stable as wire-format identifiers (ResultKey, da00
    # serialisation) and are referenced by ``output_views``.

    cumulative: CumulativeOutput = pydantic.Field(
        title='Image',
        description='Detector image accumulated since the start of the run.',
        default_factory=_make_2d_template,
    )
    current: WindowOutput = pydantic.Field(
        title='Image update',
        description=(
            'Detector image for the latest update interval only. '
            'Resets each update interval.'
        ),
        default_factory=_make_2d_template,
    )
    counts_total_cumulative: CumulativeOutput = pydantic.Field(
        title='Total',
        description=(
            'Total number of detector events accumulated since the start of the run.'
        ),
        default_factory=_make_0d_template,
    )
    counts_total: WindowOutput = pydantic.Field(
        title='Total (update)',
        description=(
            'Total number of detector events for the latest update interval only. '
            'Resets each update interval.'
        ),
        default_factory=_make_0d_template,
    )
    counts_in_toa_range_cumulative: CumulativeOutput = pydantic.Field(
        title='Total in range',
        description=(
            'Number of detector events within the configured range filter '
            'accumulated since the start of the run.'
        ),
        default_factory=_make_0d_template,
    )
    counts_in_toa_range: WindowOutput = pydantic.Field(
        title='Total in range (update)',
        description=(
            'Number of detector events within the configured range filter '
            'for the latest update interval only. Resets each update interval.'
        ),
        default_factory=_make_0d_template,
    )


class DetectorViewOutputs(DetectorViewOutputsBase):
    """Outputs for detector view workflows with ROI support."""

    output_views: ClassVar[tuple[OutputView, ...]] = (
        *_BASE_DETECTOR_VIEWS,
        OutputView(
            name='roi_spectra',
            title='ROI spectra',
            fields=('roi_spectra_cumulative', 'roi_spectra_current'),
            description='Histogram for each active ROI region.',
            params=('coordinate_mode', 'toa_edges', 'wavelength_edges'),
        ),
        OutputView(
            name='roi_rectangle',
            title='ROI Rectangles (readback)',
            fields=('roi_rectangle',),
            description='Current rectangle ROI geometries confirmed by backend.',
        ),
        OutputView(
            name='roi_polygon',
            title='ROI Polygons (readback)',
            fields=('roi_polygon',),
            description='Current polygon ROI geometries confirmed by backend.',
        ),
    )

    # Stacked ROI spectra outputs (2D: roi x time_of_arrival)
    roi_spectra_cumulative: CumulativeOutput = pydantic.Field(
        title='ROI spectra',
        description=(
            'Histogram for each active ROI region '
            'accumulated since the start of the run.'
        ),
        default_factory=_make_roi_spectra_template,
    )
    roi_spectra_current: WindowOutput = pydantic.Field(
        title='ROI spectra update',
        description=(
            'Histogram for each active ROI region '
            'for the latest update interval only. Resets each update interval.'
        ),
        default_factory=_make_roi_spectra_template,
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
        The counts outputs remain 0D scalars. If None, uses 2D default.
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

        def make_template() -> sc.DataArray:
            return _make_nd_template(output_ndim)

        class _WithNdim(base_class):  # type: ignore[valid-type,misc]
            cumulative: CumulativeOutput = pydantic.Field(
                title='Image (cumulative)',
                description=('Detector image accumulated since the start of the run.'),
                default_factory=make_template,
            )
            current: WindowOutput = pydantic.Field(
                title='Image (current)',
                description=(
                    'Detector image for the latest update interval only. '
                    'Resets each update interval.'
                ),
                default_factory=make_template,
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
                    fields=('spectrum_view',),
                    description=description,
                    params=(
                        'coordinate_mode',
                        'spectrum_params',
                        'toa_edges',
                        'wavelength_edges',
                    ),
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


class DetectorROIAuxSources(AuxSources):
    """Auxiliary source spec for ROI configuration in detector workflows.

    Subscribes to the supported ROI geometry streams (rectangle, polygon).
    :meth:`render` prefixes each stream name with the ``job_id`` so every job
    instance owns its own ROI configuration stream.

    ROI is an auxiliary source, not a gated context binding: the ROI providers
    treat a missing or empty request as "no ROI selected" (an empty result),
    so there is nothing to gate on and no cold-start seed is required. The
    detector-view factory wires the ROI streams into ``set_context`` itself
    (see :meth:`DetectorViewFactory.make_workflow`).
    """

    def __init__(self) -> None:
        super().__init__(
            {
                'roi_rectangle': 'roi_rectangle',
                'roi_polygon': 'roi_polygon',
            }
        )

    def render(
        self,
        job_id: JobId,
        selections: dict[str, str] | None = None,
    ) -> dict[str, str]:
        """Render ROI stream names with a job-specific prefix.

        Parameters
        ----------
        job_id:
            Job identifier containing source_name and job_number.
        selections:
            Ignored — ROI streams are always job-specific.

        Returns
        -------
        :
            Mapping from ROI geometry keys to job-specific stream names
            (e.g. ``'{job_id}/roi_rectangle'``).
        """
        return {
            'roi_rectangle': f"{job_id}/roi_rectangle",
            'roi_polygon': f"{job_id}/roi_polygon",
        }
