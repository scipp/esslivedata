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

from typing import Literal

import pydantic
import scipp as sc

from .. import parameter_models
from ..config import models
from ..config.instrument import Instrument
from ..config.workflow_spec import AuxSourcesBase, JobId, WorkflowOutputsBase
from ..handlers.workflow_factory import SpecHandle


class DetectorViewParams(pydantic.BaseModel):
    pixel_weighting: models.PixelWeighting = pydantic.Field(
        title="Pixel Weighting",
        description="Whether to apply pixel weighting based on the number of pixels "
        "contributing to each screen pixel.",
        default=models.PixelWeighting(
            enabled=False, method=models.WeightingMethod.PIXEL_NUMBER
        ),
    )
    # TODO split out the enabled flag?
    toa_range: parameter_models.TOARange = pydantic.Field(
        title="Time of Arrival Range",
        description="Time of arrival range for detector data.",
        default=parameter_models.TOARange(),
    )
    toa_edges: parameter_models.TOAEdges = pydantic.Field(
        title="Time of Arrival Edges",
        description="Time of arrival edges for histogramming.",
        default=parameter_models.TOAEdges(
            start=0.0,
            stop=1000.0 / 14,
            num_bins=100,
            unit=parameter_models.TimeUnit.MS,
        ),
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


def _make_0d_template_with_time() -> sc.DataArray:
    """Create an empty 0D template with time coord for scalar outputs."""
    return _make_nd_template(0, with_time_coord=True)


class DetectorViewOutputs(WorkflowOutputsBase):
    """Outputs for detector view workflows."""

    cumulative: sc.DataArray = pydantic.Field(
        title='Cumulative Counts',
        description='Time-integrated detector counts accumulated over all time.',
        default_factory=_make_2d_template,
    )
    current: sc.DataArray = pydantic.Field(
        title='Current Counts',
        description='Detector counts for the current time window since last update.',
        default_factory=_make_2d_template_with_time,
    )
    counts_total: sc.DataArray = pydantic.Field(
        title='Total Event Count',
        description='Total number of detector events in the current time window.',
        default_factory=_make_0d_template_with_time,
    )
    counts_in_toa_range: sc.DataArray = pydantic.Field(
        title='Event Count in TOA Range',
        description='Number of detector events within the configured TOA range filter.',
        default_factory=_make_0d_template_with_time,
    )

    # Stacked ROI spectra outputs (2D: roi x time_of_arrival)
    roi_spectra_current: sc.DataArray = pydantic.Field(
        title='ROI Spectra (Current)',
        description='Time-of-arrival spectra for active ROIs in current time window. '
        'Stacked 2D array with roi coordinate containing ROI indices.',
        default_factory=lambda: sc.DataArray(
            sc.zeros(dims=['roi', 'time_of_arrival'], shape=[0, 0], unit='counts'),
            coords={
                'roi': sc.array(dims=['roi'], values=[], unit=None),
                'time': sc.scalar(0, unit='ns'),
            },
        ),
    )
    roi_spectra_cumulative: sc.DataArray = pydantic.Field(
        title='ROI Spectra (Cumulative)',
        description='Cumulative time-of-arrival spectra for active ROIs. '
        'Stacked 2D array with roi coordinate containing ROI indices.',
        default_factory=lambda: sc.DataArray(
            sc.zeros(dims=['roi', 'time_of_arrival'], shape=[0, 0], unit='counts'),
            coords={'roi': sc.array(dims=['roi'], values=[], unit=None)},
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


def make_detector_view_outputs(output_ndim: int) -> type[DetectorViewOutputs]:
    """
    Create a DetectorViewOutputs subclass with spatial outputs of the given ndim.

    Parameters
    ----------
    output_ndim:
        Number of dimensions for spatial outputs (cumulative, current).
        The counts outputs remain 0D scalars with time coord.

    Returns
    -------
    :
        A subclass of DetectorViewOutputs with appropriate default_factory templates.
    """

    def make_cumulative_template() -> sc.DataArray:
        return _make_nd_template(output_ndim)

    def make_current_template() -> sc.DataArray:
        return _make_nd_template(output_ndim, with_time_coord=True)

    class CustomDetectorViewOutputs(DetectorViewOutputs):
        cumulative: sc.DataArray = pydantic.Field(
            title='Cumulative Counts',
            description='Time-integrated detector counts accumulated over all time.',
            default_factory=make_cumulative_template,
        )
        current: sc.DataArray = pydantic.Field(
            title='Current Counts',
            description=(
                'Detector counts for the current time window since last update.'
            ),
            default_factory=make_current_template,
        )

    return CustomDetectorViewOutputs


class DetectorROIAuxSources(AuxSourcesBase):
    """
    Auxiliary source model for ROI configuration in detector workflows.

    Subscribes to all supported ROI geometry streams (rectangle, polygon).
    The render() method prefixes stream names with the job_id to create job-specific
    ROI configuration streams, since each job instance needs its own ROIs.
    """

    def render(self, job_id: JobId) -> dict[str, str]:
        """
        Render ROI stream names with job-specific prefix.

        Parameters
        ----------
        job_id:
            Job identifier containing source_name and job_number.

        Returns
        -------
        :
            Mapping from ROI geometry keys to job-specific stream names.
            Keys are 'roi_rectangle', 'roi_polygon', etc.
            Values are in the format '{source_name}/{job_number}/roi_{shape}'
            (e.g., 'mantle/abc-123/roi_rectangle').
        """
        return {
            'roi_rectangle': f"{job_id}/roi_rectangle",
            'roi_polygon': f"{job_id}/roi_polygon",
        }


ProjectionType = Literal["xy_plane", "cylinder_mantle_z"]


def register_detector_view_spec(
    *,
    instrument: Instrument,
    projection: ProjectionType | dict[str, ProjectionType],
    source_names: list[str] | None = None,
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
        aux_sources=DetectorROIAuxSources,
        params=DetectorViewParams,
        outputs=DetectorViewOutputs,
    )


def register_logical_detector_view_spec(
    *,
    instrument: Instrument,
    name: str,
    title: str,
    description: str,
    source_names: list[str],
    roi_support: bool = True,
    output_ndim: int | None = None,
) -> SpecHandle:
    """
    Register a logical detector view spec with custom metadata.

    Use this helper for logical detector views that don't use the standard
    projection pattern. Unlike projection-based views, logical views are
    bespoke and require custom titles and descriptions.

    Parameters
    ----------
    instrument:
        Instrument to register specs with.
    name:
        Unique name for the spec within the detector_data namespace.
    title:
        Human-readable title for the view.
    description:
        Description of the view.
    source_names:
        List of detector source names.
    roi_support:
        Whether ROI selection is supported for this view. If True, includes
        DetectorROIAuxSources which enables the ROI detector plotter.
        Set to False for views where ROI doesn't make sense (e.g., views
        that sum over dimensions internally).
    output_ndim:
        Number of dimensions for spatial outputs (cumulative, current).
        Defaults to 2 for standard detector views. Set to 3 for views
        that produce 3D output (e.g., ESTIA's strip/blade/wire folded view).

    Returns
    -------
    :
        A SpecHandle.
    """
    outputs = (
        make_detector_view_outputs(output_ndim)
        if output_ndim is not None
        else DetectorViewOutputs
    )
    return instrument.register_spec(
        namespace="detector_data",
        name=name,
        version=1,
        title=title,
        description=description,
        source_names=source_names,
        aux_sources=DetectorROIAuxSources if roi_support else None,
        params=DetectorViewParams,
        outputs=outputs,
    )
