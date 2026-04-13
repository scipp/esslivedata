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

from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from .detector_view.types import TransformValueStream

import pydantic
import scipp as sc

from .. import parameter_models
from ..config import models
from ..config.instrument import Instrument
from ..config.workflow_spec import AuxInput, AuxSources, JobId, WorkflowOutputsBase
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


class DetectorViewOutputsBase(WorkflowOutputsBase):
    """Base outputs for detector view workflows (without ROI support)."""

    # Field names are legacy identifiers kept for compatibility with existing
    # workflow templates and serialized configs. Titles are user-facing names.

    cumulative: sc.DataArray = pydantic.Field(
        title='Image (cumulative)',
        description='Detector image accumulated since the start of the run.',
        default_factory=_make_2d_template,
    )
    current: sc.DataArray = pydantic.Field(
        title='Image (current)',
        description=(
            'Detector image for the latest update interval only. '
            'Resets each update interval.'
        ),
        default_factory=_make_2d_template_with_time,
    )
    counts_total_cumulative: sc.DataArray = pydantic.Field(
        title='Total (cumulative)',
        description=(
            'Total number of detector events accumulated since the start of the run.'
        ),
        default_factory=_make_0d_template,
    )
    counts_total: sc.DataArray = pydantic.Field(
        title='Total (current)',
        description=(
            'Total number of detector events for the latest update interval only. '
            'Resets each update interval.'
        ),
        default_factory=_make_0d_template_with_time,
    )
    counts_in_toa_range_cumulative: sc.DataArray = pydantic.Field(
        title='Total in interval (cumulative)',
        description=(
            'Number of detector events within the configured range filter '
            'accumulated since the start of the run.'
        ),
        default_factory=_make_0d_template,
    )
    counts_in_toa_range: sc.DataArray = pydantic.Field(
        title='Total in interval (current)',
        description=(
            'Number of detector events within the configured range filter '
            'for the latest update interval only. Resets each update interval.'
        ),
        default_factory=_make_0d_template_with_time,
    )


class DetectorViewOutputs(DetectorViewOutputsBase):
    """Outputs for detector view workflows with ROI support."""

    # Stacked ROI spectra outputs (2D: roi x time_of_arrival)
    roi_spectra_cumulative: sc.DataArray = pydantic.Field(
        title='ROI spectra (cumulative)',
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
        title='ROI spectra (current)',
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


def make_detector_view_outputs(
    output_ndim: int | None = None,
    *,
    roi_support: bool = True,
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

    Returns
    -------
    :
        A subclass of DetectorViewOutputsBase with appropriate configuration.
    """
    base_class = DetectorViewOutputs if roi_support else DetectorViewOutputsBase

    if output_ndim is None:
        return base_class

    def make_cumulative_template() -> sc.DataArray:
        return _make_nd_template(output_ndim)

    def make_current_template() -> sc.DataArray:
        return _make_nd_template(output_ndim, with_time_coord=True)

    class CustomDetectorViewOutputs(base_class):  # type: ignore[valid-type]
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

    return CustomDetectorViewOutputs


class DetectorROIAuxSources(AuxSources):
    """Auxiliary source spec for ROI configuration in detector workflows.

    Subscribes to all supported ROI geometry streams (rectangle, polygon).
    The render() method prefixes ROI stream names with the job_id to create
    job-specific ROI configuration streams, since each job instance needs its
    own ROIs.

    Optionally also advertises one or more global f144 streams that drive
    runtime-dynamic NeXus transformation values for specific source_names.
    These streams are physical properties of the instrument (not job-
    specific), so they are rendered un-prefixed and only routed to the jobs
    whose source_name actually consumes them.
    """

    def __init__(
        self,
        dynamic_transforms: dict[str, TransformValueStream] | None = None,
    ) -> None:
        self._dynamic_transforms = dynamic_transforms or {}
        inputs: dict[str, str | AuxInput] = {
            'roi_rectangle': 'roi_rectangle',
            'roi_polygon': 'roi_polygon',
        }
        # Advertise each unique global aux stream so the dashboard schema
        # and spec validation know it exists. Routing is per-source via
        # render().
        for binding in self._dynamic_transforms.values():
            inputs.setdefault(binding.aux_stream, binding.aux_stream)
        super().__init__(inputs)

    def render(
        self,
        job_id: JobId,
        selections: dict[str, str] | None = None,
    ) -> dict[str, str]:
        """Render ROI stream names with job-specific prefix, plus any
        source-specific global aux streams.

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
            (e.g., ``'{source_name}/{job_number}/roi_rectangle'``), plus
            any global aux streams bound to this source's NeXus transforms,
            rendered un-prefixed.
        """
        rendered: dict[str, str] = {
            'roi_rectangle': f"{job_id}/roi_rectangle",
            'roi_polygon': f"{job_id}/roi_polygon",
        }
        binding = self._dynamic_transforms.get(job_id.source_name)
        if binding is not None:
            rendered[binding.aux_stream] = binding.aux_stream
        return rendered


ProjectionType = Literal["xy_plane", "cylinder_mantle_z"]


def register_detector_view_spec(
    *,
    instrument: Instrument,
    projection: ProjectionType | dict[str, ProjectionType],
    source_names: list[str] | None = None,
    aux_sources: AuxSources | None = None,
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
    aux_sources:
        Optional auxiliary source specification. If None (default), uses
        DetectorROIAuxSources for ROI geometry streams. Instruments that need
        both ROI and position streams can subclass DetectorROIAuxSources.

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

    return instrument.register_spec(
        namespace="detector_data",
        name=name,
        version=1,
        title=title,
        description=description,
        source_names=source_names,
        aux_sources=aux_sources if aux_sources is not None else DetectorROIAuxSources(),
        params=DetectorViewParams,
        outputs=DetectorViewOutputs,
    )
