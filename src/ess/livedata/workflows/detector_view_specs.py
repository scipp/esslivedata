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
from typing import ClassVar

import pydantic
import scipp as sc

from .. import parameter_models
from ..config import models
from ..config.roi_names import DETECTOR_PIXELS_COORD
from ..config.spectral_coords import (
    spectral_edges_template_coords,
    spectral_range_template_coords,
)
from ..config.workflow_spec import (
    CumulativeOutput,
    OutputView,
    WindowOutput,
    WorkflowOutputsBase,
)


class DetectorViewParams(pydantic.BaseModel):
    """Detector-view parameters offering both spectral axes."""

    spectral_axis: parameter_models.SpectralAxis = pydantic.Field(
        title="Spectral Axis",
        description="Axis along which the detector view histograms events.",
        default_factory=parameter_models.TOASpectralAxis,
    )
    pixel_weighting: models.PixelWeighting = pydantic.Field(
        title="Pixel Weighting",
        description=(
            "Divide each image pixel by the number of detector pixels that map to "
            "it, so that the image shows counts per detector pixel. In geometric "
            "projections (xy-plane, cylinder) this number varies across the image "
            "and is averaged over the position noise, so it can be fractional at "
            "module edges. Without weighting, the image shows pixel density as "
            "well as count rate. Image pixels that no detector pixel maps to, such "
            "as gaps between detector modules, have no value (NaN). In logical "
            "views every image pixel has the same number of detector pixels, so "
            "weighting has no effect or divides the whole image by a constant. ROI "
            "outputs stay summed counts. To show them per detector pixel, use "
            "'Per Detector Pixel' in the plot settings."
        ),
        default_factory=models.PixelWeighting,
    )


class TOAOnlyDetectorViewParams(DetectorViewParams):
    """Detector-view parameters restricted to the time-of-arrival axis.

    For specs that cannot convert to wavelength: the instrument has no lookup
    table, or the workflow is not wired to receive one.
    """

    spectral_axis: parameter_models.TOASpectralAxis = pydantic.Field(
        title="Spectral Axis",
        description="Axis along which the detector view histograms events. Only "
        "time of arrival (TOA) is available for this workflow.",
        default_factory=parameter_models.TOASpectralAxis,
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
        of the input histogram (time-of-arrival or wavelength), so it is not
        listed here.
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
            '(time-of-arrival or wavelength, depending on the spectral axis).'
        )
        if self.extra_description:
            return f'{base}\n\n{self.extra_description}'
        return base


def _make_nd_template(ndim: int) -> sc.DataArray:
    """Create an empty template for an output summed over the spectral axis."""
    coords = spectral_range_template_coords()
    if ndim == 0:
        return sc.DataArray(sc.scalar(0, unit='counts'), coords=coords)
    dims = [f'dim_{i}' for i in range(ndim)]
    return sc.DataArray(
        sc.zeros(dims=dims, shape=[0] * ndim, unit='counts'), coords=coords
    )


def _make_2d_template() -> sc.DataArray:
    """Create an empty 2D template."""
    return _make_nd_template(2)


def _make_0d_template() -> sc.DataArray:
    """Create an empty 0D template for scalar outputs."""
    return _make_nd_template(0)


def _make_roi_coords() -> dict[str, sc.Variable]:
    """Create the empty coords shared by all per-ROI outputs."""
    return {
        'roi': sc.array(dims=['roi'], values=[], unit=None, dtype='int32'),
        DETECTOR_PIXELS_COORD: sc.array(dims=['roi'], values=[], dtype='float64'),
    }


def _make_roi_scalars_template() -> sc.DataArray:
    """Create an empty template for one scalar per ROI."""
    return sc.DataArray(
        sc.zeros(dims=['roi'], shape=[0], unit='counts'),
        coords={**_make_roi_coords(), **spectral_range_template_coords()},
    )


def _make_roi_spectra_template() -> sc.DataArray:
    """Create an empty template for stacked per-ROI spectra."""
    return sc.DataArray(
        sc.zeros(dims=['roi', 'time_of_arrival'], shape=[0, 0], unit='counts'),
        coords={**_make_roi_coords(), **spectral_edges_template_coords()},
    )


_DETECTOR_PIXELS_NOTE = (
    ' Each ROI carries its number of detector pixels, so plots can show counts '
    'per detector pixel. Image pixels without a detector pixel behind them do not '
    'count. For detectors downsampled at ingest, a detector pixel is a pixel of the '
    'downsampled image.'
)


_BASE_DETECTOR_VIEWS: tuple[OutputView, ...] = (
    OutputView(
        name='image',
        title='Image',
        fields=('cumulative', 'current'),
        description=(
            'Detector image. With "since run start" shows accumulated counts; '
            'with "latest update" or a window, shows recent counts. With pixel '
            'weighting, counts are per detector pixel.'
        ),
        params=('spectral_axis', 'pixel_weighting'),
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
        params=('spectral_axis',),
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


def _make_area_image_template() -> sc.DataArray:
    """Create an empty template for an area-detector image."""
    return sc.DataArray(sc.zeros(dims=['dim_0', 'dim_1'], shape=[0, 0], unit='counts'))


class AreaDetectorViewOutputs(DetectorViewOutputsBase):
    """Outputs for area-detector (camera) views.

    Camera images have no spectral axis, so unlike the images of other detector
    views they carry no spectral range coords.
    """

    cumulative: CumulativeOutput = pydantic.Field(
        title='Image',
        description='Detector image accumulated since the start of the run.',
        default_factory=_make_area_image_template,
    )
    current: WindowOutput = pydantic.Field(
        title='Image update',
        description=(
            'Detector image for the latest update interval only. '
            'Resets each update interval.'
        ),
        default_factory=_make_area_image_template,
    )


class DetectorViewOutputs(DetectorViewOutputsBase):
    """Outputs for detector view workflows with ROI support."""

    output_views: ClassVar[tuple[OutputView, ...]] = (
        *_BASE_DETECTOR_VIEWS,
        OutputView(
            name='roi_spectra',
            title='ROI spectra',
            fields=('roi_spectra_cumulative', 'roi_spectra_current'),
            description='Histogram for each active ROI region.' + _DETECTOR_PIXELS_NOTE,
            params=('spectral_axis',),
        ),
        OutputView(
            name='roi_total_in_range',
            title='Σ ROI',
            fields=('roi_counts_in_range_cumulative', 'roi_counts_in_range_current'),
            description=(
                'Counts summed over each ROI and over the range filter, one value '
                'per ROI. Unlike the ROI spectra, this respects the range filter.'
                + _DETECTOR_PIXELS_NOTE
            ),
            params=('spectral_axis',),
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

    # One scalar per ROI (1D: roi)
    roi_counts_in_range_cumulative: CumulativeOutput = pydantic.Field(
        title='Σ ROI',
        description=(
            'Counts summed over each ROI and over the range filter, '
            'accumulated since the start of the run.'
        ),
        default_factory=_make_roi_scalars_template,
    )
    roi_counts_in_range_current: WindowOutput = pydantic.Field(
        title='Σ ROI (update)',
        description=(
            'Counts summed over each ROI and over the range filter '
            'for the latest update interval only. Resets each update interval.'
        ),
        default_factory=_make_roi_scalars_template,
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
    output_template: sc.DataArray | None = None,
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
    output_template:
        Template for the spatial outputs, in place of the generic one of
        ``output_ndim`` dims. For outputs whose dims and coords are known when
        the spec is registered; the dashboard selects plotters by them.
    roi_support:
        Whether to include the ROI outputs.
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

    if output_ndim is not None and output_template is not None:
        raise ValueError("Pass either output_ndim or output_template, not both.")
    if output_ndim is None and output_template is None and spectrum_view is None:
        return base_class

    if output_ndim is not None or output_template is not None:

        def make_template() -> sc.DataArray:
            if output_template is not None:
                return output_template.copy().assign_coords(
                    spectral_range_template_coords()
                )
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
                    params=('spectral_axis', 'spectrum_params'),
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
    *,
    base: type[DetectorViewParams] = DetectorViewParams,
) -> type[DetectorViewParams]:
    """Return a detector-view params model, adding spectrum-specific fields.

    When ``spectrum_view.params_model`` is set, the returned model adds a
    ``spectrum_params`` field of that model type so the runtime parameters can
    be exposed in the UI. Workflows without spectrum-view (or whose spectrum
    transform needs no runtime parameters) get ``base`` unchanged.

    Parameters
    ----------
    spectrum_view:
        Optional spectrum-view configuration contributing a params field.
    base:
        Params model to extend, selecting which spectral axes the spec
        offers. Defaults to :class:`DetectorViewParams`, which offers both.
    """
    if spectrum_view is None or spectrum_view.params_model is None:
        return base

    params_model = spectrum_view.params_model
    title = spectrum_view.output_title
    description = spectrum_view.params_description

    class DetectorViewWithSpectrumParams(base):  # type: ignore[valid-type,misc]
        spectrum_params: params_model = pydantic.Field(  # type: ignore[valid-type]
            title=title,
            description=description,
            default_factory=params_model,
        )

    return DetectorViewWithSpectrumParams
